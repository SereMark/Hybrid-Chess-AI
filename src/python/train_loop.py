"""Core training loop that drives self-play sampling and optimisation."""

from __future__ import annotations

import time
from typing import Any

import config as C
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler

__all__ = ["run_training_iteration", "train_step"]

# ---------------------------------------------------------------------------#
# Schedules


def _entropy_coefficient(iteration: int) -> float:
    """Linearly anneal the entropy coefficient towards the configured floor."""
    start = float(C.TRAIN.loss_entropy_coef)
    floor = float(C.TRAIN.loss_entropy_min_coef)
    horizon = max(1, int(C.TRAIN.loss_entropy_iters))
    progress = max(0.0, min(1.0, (iteration - 1) / horizon))
    return start + progress * (floor - start)


def _recent_sample_ratio() -> float:
    """Return the recent-sample ratio constrained to configured bounds."""
    base = float(C.SAMPLING.recent_ratio)
    lo = float(C.SAMPLING.recent_ratio_min)
    hi = float(C.SAMPLING.recent_ratio_max)
    return float(np.clip(base, lo, hi))


def _games_for_iteration(iteration: int) -> int:
    """Scale games per iteration during warm-up to keep startup responsive."""
    base = int(max(1, C.TRAIN.games_per_iter))
    min_scale = float(np.clip(getattr(C.TRAIN, "games_per_iter_scale_min", 1.0), 0.1, 1.0))
    warmup = int(max(0, getattr(C.TRAIN, "games_per_iter_warmup_iters", 0)))
    if warmup <= 0 or min_scale >= 1.0:
        return base
    progress = float(np.clip(iteration / max(1, warmup), 0.0, 1.0))
    scale = min_scale + (1.0 - min_scale) * progress
    return max(1, int(round(base * scale)))


# ---------------------------------------------------------------------------#
# Public API


def run_training_iteration(trainer: Any) -> dict[str, float | int | str]:
    """Execute one training iteration of self-play generation and optimisation."""
    buf_capacity = max(1, trainer.selfplay_engine.get_capacity())
    buf_size = trainer.selfplay_engine.size()
    buf_fill = buf_size / buf_capacity

    recent_ratio = _recent_sample_ratio()
    recent_window = float(np.clip(C.SAMPLING.recent_window_frac, 0.0, 1.0))

    # Resignation policy
    if not C.RESIGN.enabled:
        trainer.selfplay_engine.enable_resign(False)
        trainer.selfplay_engine.resign_consecutive = 0
        resign_status = "disabled"
    elif trainer.iteration < C.RESIGN.cooldown_iters:
        trainer.selfplay_engine.enable_resign(False)
        trainer.selfplay_engine.resign_consecutive = 0
        resign_status = "warmup"
    else:
        trainer.selfplay_engine.enable_resign(True)
        trainer.selfplay_engine.set_resign_params(C.RESIGN.value_threshold, C.RESIGN.min_plies)
        trainer.selfplay_engine.resign_consecutive = int(max(1, C.RESIGN.consecutive_required))
        resign_status = "enabled"

    trainer.selfplay_engine.update_adjudication(trainer.iteration)

    # Self-play
    t0 = time.time()
    games = _games_for_iteration(trainer.iteration)
    sp_stats = trainer.selfplay_engine.play_games(games)
    selfplay_time = time.time() - t0
    games_generated = int(sp_stats.get("games", 0))

    # Planned optimiser steps this iteration
    batches_target = max(
        int(C.TRAIN.update_steps_min),
        min(
            int(C.TRAIN.update_steps_max),
            int(round(games_generated * float(C.TRAIN.samples_per_new_game))),
        ),
    )

    # Training
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    grad_norms: list[float] = []

    optimizer_steps = 0
    t1 = time.time()
    for _ in range(batches_target):
        batch = trainer.selfplay_engine.sample_batch(trainer.train_batch_size, recent_ratio, recent_window)
        if not batch[0]:
            break
        result = train_step(trainer, batch)
        if result is None:
            continue
        p_loss, v_loss, g_norm, entr = result
        policy_losses.append(p_loss)
        value_losses.append(v_loss)
        grad_norms.append(g_norm)
        entropies.append(entr)
        optimizer_steps += 1
    train_time = time.time() - t1

    # Aggregates
    if optimizer_steps == 0:
        avgP = avgV = avgE = avgG = stdP = stdV = stdE = stdG = 0.0
    else:
        avgP = float(np.mean(policy_losses))
        avgV = float(np.mean(value_losses))
        avgE = float(np.mean(entropies))
        avgG = float(np.mean(grad_norms))
        stdP = float(np.std(policy_losses, ddof=0))
        stdV = float(np.std(value_losses, ddof=0))
        stdE = float(np.std(entropies, ddof=0))
        stdG = float(np.std(grad_norms, ddof=0))

    lr = float(trainer.optimizer.param_groups[0]["lr"])
    ent_coef = _entropy_coefficient(trainer.iteration)
    train_samples = optimizer_steps * trainer.train_batch_size
    sps = train_samples / train_time if train_time > 0 and train_samples > 0 else 0.0

    return {
        "train_steps_planned": batches_target,
        "train_steps_actual": optimizer_steps,
        "optimizer_steps": optimizer_steps,
        "train_samples": train_samples,
        "policy_loss": avgP,
        "policy_loss_std": stdP,
        "value_loss": avgV,
        "value_loss_std": stdV,
        "entropy": avgE,
        "entropy_std": stdE,
        "avg_grad_norm": avgG,
        "grad_norm_std": stdG,
        "entropy_coef": ent_coef,
        "learning_rate": lr,
        "train_time_s": train_time,
        "selfplay_time_s": selfplay_time,
        "buffer_percent": buf_fill * 100.0,
        "train_recent_pct": recent_ratio * 100.0,
        "buffer_size": buf_size,
        "buffer_capacity": buf_capacity,
        "resign_status": resign_status,
        "resign_threshold": trainer.selfplay_engine.resign_threshold,
        "resign_min_plies": trainer.selfplay_engine.resign_min_plies,
        "samples_per_sec": sps,
        "selfplay_stats": sp_stats,
        "adjudication_phase": trainer.selfplay_engine.adjudication_phase,
        "adjudication_enabled": int(bool(trainer.selfplay_engine.adjudication_enabled)),
        "adjudication_min_plies": int(trainer.selfplay_engine.adjudication_min_plies),
        "adjudication_value_margin": float(trainer.selfplay_engine.adjudication_value_margin),
        "adjudication_persist_plies": int(trainer.selfplay_engine.adjudication_persist),
        "adjudication_material_margin": float(trainer.selfplay_engine.adjudication_material_margin),
    }


# ---------------------------------------------------------------------------#
# Training step
def train_step(
    trainer: Any,
    batch_data: tuple[list[Any], list[np.ndarray], list[np.ndarray], list[np.ndarray]],
) -> tuple[float, float, float, float] | None:
    """Perform one optimisation step and return `(policy, value, grad, entropy)`."""
    states_u8, indices_i32, counts_u16, values_i8 = batch_data
    if not states_u8:
        return None

    # Inputs
    x_u8 = np.stack(states_u8).astype(np.uint8, copy=False)
    x = torch.from_numpy(x_u8).to(trainer.device, non_blocking=True).to(dtype=torch.float32)
    x = (x / float(C.DATA.u8_scale)).contiguous(memory_format=torch.channels_last)

    v_i8 = np.asarray(values_i8, dtype=np.int8)
    v_target = torch.from_numpy(v_i8).to(trainer.device, non_blocking=True).to(dtype=torch.float32)
    v_target = v_target / float(C.DATA.value_i8_scale)

    # One-time augmentation setup on trainer
    if not hasattr(trainer, "_aug_mirror_idx"):
        from augmentation import Augment as _Aug

        def _invert(perm: np.ndarray) -> np.ndarray:
            out = np.empty_like(perm, dtype=np.int64)
            out[perm.astype(np.int64)] = np.arange(perm.size, dtype=np.int64)
            return out

        mirror_idx = _Aug._policy_index_permutation("mirror")
        rot_idx = _Aug._policy_index_permutation("rot180")
        vflip_idx = _Aug._policy_index_permutation("vflip_cs")
        plane_perm = _Aug._vflip_cs_plane_permutation(x.shape[1])
        feat_idx = _Aug._feature_plane_indices()

        trainer._turn_plane_idx = int(feat_idx.get("turn_plane", x.shape[1]))
        trainer._aug_mirror_idx = torch.tensor(mirror_idx, dtype=torch.long, device=trainer.device)
        trainer._aug_rot_idx = torch.tensor(rot_idx, dtype=torch.long, device=trainer.device)
        trainer._aug_vflip_idx = torch.tensor(vflip_idx, dtype=torch.long, device=trainer.device)
        trainer._aug_plane_perm = torch.tensor(plane_perm, dtype=torch.long, device=trainer.device)
        trainer._aug_mirror_inv = _invert(mirror_idx)
        trainer._aug_rot_inv = _invert(rot_idx)
        trainer._aug_vflip_inv = _invert(vflip_idx)

    # Stochastic spatial+policy augmentations
    if np.random.rand() < C.AUGMENT.mirror_prob:
        x = torch.flip(x, dims=[-1])
        indices_i32 = [trainer._aug_mirror_inv[np.asarray(idx, dtype=np.int64)].astype(np.int32) for idx in indices_i32]
    if np.random.rand() < C.AUGMENT.rot180_prob:
        x = torch.flip(x, dims=[-2, -1])
        indices_i32 = [trainer._aug_rot_inv[np.asarray(idx, dtype=np.int64)].astype(np.int32) for idx in indices_i32]
    if np.random.rand() < C.AUGMENT.vflip_prob:
        x = torch.flip(x, dims=[-2])
        x = x.index_select(1, trainer._aug_plane_perm)
        if 0 <= trainer._turn_plane_idx < x.shape[1]:
            x[:, trainer._turn_plane_idx] = 1.0 - x[:, trainer._turn_plane_idx]
        indices_i32 = [trainer._aug_vflip_inv[np.asarray(idx, dtype=np.int64)].astype(np.int32) for idx in indices_i32]
        v_target = -v_target

    # Forward + losses (AMP aware)
    trainer.model.train()
    autocast_device = trainer.device.type
    autocast_enabled = bool(getattr(trainer, "_amp_enabled", False))
    autocast_dtype = getattr(trainer, "_autocast_dtype", torch.float16)
    if autocast_device != "cuda":
        autocast_dtype = torch.bfloat16

    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=autocast_enabled):
        policy_logits, value_pred = trainer.model(x)
        value_pred = value_pred.reshape(-1)
        log_probs = F.log_softmax(policy_logits, dim=1)

        # Sparse cross-entropy via indexed rows/cols with counts as weights
        rows_np: list[np.ndarray] = []
        cols_np: list[np.ndarray] = []
        cnts_np: list[np.ndarray] = []
        for row_idx, (idx_arr, cnt_arr) in enumerate(zip(indices_i32, counts_u16, strict=False)):
            idx_np = np.asarray(idx_arr, dtype=np.int64)
            cnt_np = np.asarray(cnt_arr, dtype=np.float32)
            if idx_np.size == 0:
                continue
            valid = (idx_np >= 0) & (idx_np < log_probs.shape[1]) & (cnt_np > 0)
            if not np.any(valid):
                continue
            idx_np = idx_np[valid]
            cnt_np = cnt_np[valid]
            cols_np.append(idx_np)
            cnts_np.append(cnt_np)
            rows_np.append(np.full(idx_np.shape[0], row_idx, dtype=np.int64))

        if rows_np:
            cols = torch.from_numpy(np.concatenate(cols_np)).to(trainer.device, non_blocking=True, dtype=torch.long)
            cnts = torch.from_numpy(np.concatenate(cnts_np)).to(trainer.device, non_blocking=True, dtype=torch.float32)
            rows = torch.from_numpy(np.concatenate(rows_np)).to(trainer.device, non_blocking=True, dtype=torch.long)

            denom_rows = torch.zeros((x.shape[0],), device=trainer.device, dtype=torch.float32)
            denom_rows.index_add_(0, rows, cnts)
            denom = denom_rows.index_select(0, rows).clamp_min(1e-9)
            weights = cnts / denom

            eps = float(max(0.0, C.TRAIN.loss_policy_label_smooth))
            if eps > 0.0:
                m_rows = torch.zeros((x.shape[0],), device=trainer.device, dtype=torch.float32)
                m_rows.index_add_(0, rows, torch.ones_like(weights))
                m = m_rows.index_select(0, rows).clamp_min(1.0)
                weights = (1.0 - eps) * weights + (eps / m)

            selected = log_probs[rows, cols]
            per_entry = -weights * selected
            loss_rows = torch.zeros((x.shape[0],), device=trainer.device, dtype=torch.float32)
            loss_rows.index_add_(0, rows, per_entry)
            policy_loss = loss_rows.mean()
        else:
            policy_loss = torch.tensor(0.0, device=trainer.device, dtype=torch.float32)

        value_loss = F.mse_loss(value_pred, v_target)
        entropy = (-(F.softmax(policy_logits, dim=1) * log_probs).sum(dim=1)).mean()

        total_loss = (
            C.TRAIN.loss_policy_weight * policy_loss
            + C.TRAIN.loss_value_weight * value_loss
            - _entropy_coefficient(trainer.iteration) * entropy
        )

    # Backward + step (AMP aware)
    trainer.optimizer.zero_grad(set_to_none=True)
    scaler_obj = getattr(trainer, "scaler", None)
    scaler: GradScaler | None = None
    if isinstance(scaler_obj, GradScaler) and scaler_obj.is_enabled():
        scaler = scaler_obj

    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(trainer.optimizer)
    else:
        total_loss.backward()

    try:
        grad_total = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), float(C.TRAIN.grad_clip_norm))
    except RuntimeError:
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.scheduler.step()
        raise

    if torch.isnan(grad_total) or torch.isinf(grad_total):
        trainer.optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.update()  # keep scaler moving even if step is skipped
        trainer.scheduler.step()
        return None

    if scaler is not None:
        scaler.step(trainer.optimizer)
        scaler.update()
    else:
        trainer.optimizer.step()
    trainer.scheduler.step()

    if getattr(trainer, "ema", None) is not None:
        trainer.ema.update(trainer.model)

    return (
        float(policy_loss.detach()),
        float(value_loss.detach()),
        float(grad_total.detach().cpu()),
        float(entropy.detach().cpu()),
    )
