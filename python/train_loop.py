from __future__ import annotations

import contextlib
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from reporting import (
    format_gb,
    format_si,
    format_time,
    get_mem_info,
    get_sys_info,
)


def run_training_iteration(trainer: Any) -> dict[str, int | float]:
    if trainer.iteration < int(C.RESIGN.DISABLE_UNTIL_ITERS):
        trainer.selfplay_engine.resign_consecutive = 0
    else:
        trainer.selfplay_engine.resign_consecutive = max(int(C.RESIGN.CONSECUTIVE_PLIES), int(C.RESIGN.CONSECUTIVE_MIN))
    if trainer.iteration == C.ARENA.GATE_Z_SWITCH_ITER:
        trainer._gate.z = float(C.ARENA.GATE_Z_LATE)
    stats: dict[str, int | float] = {}
    torch.cuda.reset_peak_memory_stats(trainer.device)
    header_lr = float(trainer.scheduler.peek_next_lr())
    mem_info = get_mem_info(trainer._proc, trainer.device, trainer.device_total_gb)
    sys_info = get_sys_info(trainer._proc)
    try:
        buffer_len = int(trainer.selfplay_engine.size())
        buffer_cap = int(trainer.selfplay_engine.get_capacity())
    except Exception:
        buffer_len = 0
        buffer_cap = 1
    buffer_pct = (buffer_len / buffer_cap) * 100
    
    total_elapsed = time.time() - trainer.start_time
    pct_done = 100.0 * (trainer.iteration - 1) / max(1, C.TRAIN.TOTAL_ITERATIONS)
    trainer.log.info(
        f"[Iter    ] i {trainer.iteration:>3}/{C.TRAIN.TOTAL_ITERATIONS} ({pct_done:>4.1f}%) | LRnext {header_lr:.2e} | t {format_time(total_elapsed)} | "
        f"buf {format_si(buffer_len)}/{format_si(buffer_cap)} ({int(buffer_pct)}%) | "
        f"GPU {format_gb(mem_info['allocated_gb'])}/{format_gb(mem_info['reserved_gb'])}/{format_gb(mem_info['total_gb'])} | "
        f"RSS {format_gb(mem_info['rss_gb'])} | CPU {sys_info['cpu_sys_pct']:.0f}/{sys_info['cpu_proc_pct']:.0f}% | "
        f"RAM {format_gb(sys_info['ram_used_gb'])}/{format_gb(sys_info['ram_total_gb'])} ({sys_info['ram_pct']:.0f}%) | load {sys_info['load1']:.2f}"
    )

    t0 = time.time()
    game_stats = trainer.selfplay_engine.play_games(C.TRAIN.GAMES_PER_ITER)
    trainer.total_games += int(game_stats["games"])
    sp_elapsed = time.time() - t0

    games_per_min = game_stats["games"] / max(1e-9, sp_elapsed / 60)
    moves_per_sec = game_stats["moves"] / max(1e-9, sp_elapsed)
    games_count = int(game_stats["games"])
    white_wins = int(game_stats["white_wins"])
    black_wins = int(game_stats["black_wins"])
    draws_true = int(game_stats.get("draws_true", 0))
    draws_cap = int(game_stats.get("draws_cap", 0))
    draws_count = int(draws_true + draws_cap)
    white_win_pct = 100.0 * white_wins / max(1, games_count)
    draw_pct = 100.0 * draws_count / max(1, games_count)
    black_win_pct = 100.0 * black_wins / max(1, games_count)
    avg_game_length = game_stats["moves"] / max(1, games_count)

    eval_metrics = trainer.evaluator.get_metrics()
    requests_total = int(eval_metrics.get("requests_total", 0))
    cache_hits_total = int(eval_metrics.get("cache_hits_total", 0))
    batches_total = int(eval_metrics.get("batches_total", 0))
    eval_positions_total = int(eval_metrics.get("eval_positions_total", 0))
    max_batch_size = int(eval_metrics.get("batch_size_max", 0))
    prev_metrics = getattr(trainer, "_prev_eval_m", {}) or {}
    delta_requests = int(requests_total - int(prev_metrics.get("requests_total", 0)))
    delta_hits = int(cache_hits_total - int(prev_metrics.get("cache_hits_total", 0)))
    delta_batches = int(batches_total - int(prev_metrics.get("batches_total", 0)))
    delta_eval_positions = int(eval_positions_total - int(prev_metrics.get("eval_positions_total", 0)))
    hit_rate = (100.0 * cache_hits_total / max(1, requests_total)) if requests_total else 0.0
    hit_rate_d = (100.0 * delta_hits / max(1, delta_requests)) if delta_requests > 0 else 0.0

    try:
        if delta_batches > 0 and delta_eval_positions > 0:
            avg_batch = delta_eval_positions / max(1, delta_batches)
            cap = int(trainer._eval_batch_cap)
            max_cap_allowed = int(min(8192, max(1024, int(2048 * (trainer.device_total_gb / 16.0)))))
            mem_now = get_mem_info(trainer._proc, trainer.device, trainer.device_total_gb)
            reserved_frac = float(mem_now["reserved_gb"]) / max(1e-9, float(mem_now["total_gb"]))
            if avg_batch >= 0.90 * cap and cap < max_cap_allowed and reserved_frac < 0.90:
                new_cap = int(min(max_cap_allowed, cap + 512))
                if new_cap != cap:
                    trainer._eval_batch_cap = new_cap
                    trainer.evaluator.set_batching_params(batch_size_max=new_cap)
                    trainer.log.info(f"[AUTO    ] eval_batch_size_max {cap} -> {new_cap}")
            elif avg_batch <= 0.25 * cap and trainer._eval_coalesce_ms > 4:
                old_ms = int(trainer._eval_coalesce_ms)
                new_ms = int(max(4, int(old_ms * 0.8)))
                if new_ms != old_ms:
                    trainer._eval_coalesce_ms = new_ms
                    trainer.evaluator.set_batching_params(coalesce_ms=new_ms)
                    trainer.log.info(f"[AUTO    ] eval_coalesce_ms {old_ms} -> {new_ms}")
            elif avg_batch >= 0.80 * cap and trainer._eval_coalesce_ms < 50:
                old_ms = int(trainer._eval_coalesce_ms)
                new_ms = int(min(50, int(old_ms * 1.2 + 1)))
                if new_ms != old_ms:
                    trainer._eval_coalesce_ms = new_ms
                    trainer.evaluator.set_batching_params(coalesce_ms=new_ms)
                    trainer.log.info(f"[AUTO    ] eval_coalesce_ms {old_ms} -> {new_ms}")
    except Exception:
        pass

    sp_line = (
        f"games {games_count:,} | W/D/B {white_wins}/{draws_count}/{black_wins} "
        f"({white_win_pct:.0f}%/{draw_pct:.0f}%/{black_win_pct:.0f}%) | len {avg_game_length:>4.1f} | "
        f"gpm {games_per_min:>5.1f} | mps {moves_per_sec / 1000:>4.1f}k | t {format_time(sp_elapsed)} | "
        f"new {format_si(int(game_stats.get('moves', 0)))} | cap {draws_cap}"
    )

    ev_short = (
        f"req {format_si(requests_total)}(+{format_si(delta_requests)}) | hit {hit_rate:>4.1f}% (+{hit_rate_d:>4.1f}%) | "
        f"batches {format_si(batches_total)}(+{format_si(delta_batches)}) | "
        f"evalN {format_si(eval_positions_total)}(+{format_si(delta_eval_positions)}) | bmax {max_batch_size}"
    )
    trainer.log.info("[SP] " + sp_line + "\n" + " " * 6 + "[EV] " + ev_short)

    stats.update(game_stats)
    stats["selfplay_time"] = sp_elapsed
    stats["games_per_min"] = games_per_min
    stats["moves_per_sec"] = moves_per_sec

    t1 = time.time()
    if (C.LOG.CUDA_EMPTY_CACHE_EVERY_ITERS > 0) and (trainer.iteration % C.LOG.CUDA_EMPTY_CACHE_EVERY_ITERS == 0):
        torch.cuda.empty_cache()

    losses: list[tuple[torch.Tensor, torch.Tensor]] = []
    min_batch_samples = max(1, trainer.train_batch_size // 2)
    new_examples = int(game_stats.get("moves", 0))
    target_train_samples = int(C.TRAIN.TARGET_TRAIN_SAMPLES_PER_NEW * max(1, new_examples))
    num_steps = int(np.ceil(target_train_samples / max(1, trainer.train_batch_size)))
    num_steps = max(C.TRAIN.UPDATE_STEPS_MIN, min(C.TRAIN.UPDATE_STEPS_MAX, num_steps))
    samples_ratio = 0.0
    try:
        buffer_size_now = int(trainer.selfplay_engine.size())
    except Exception:
        buffer_size_now = 0
    if buffer_size_now < min_batch_samples:
        num_steps = 0
    else:
        samples_ratio = (num_steps * trainer.train_batch_size) / max(1, new_examples)

    grad_norm_sum: float = 0.0
    entropy_sum: float = 0.0
    for _step in range(num_steps):
        batch = trainer.selfplay_engine.sample_batch(
            trainer.train_batch_size, recent_ratio=C.SAMPLING.TRAIN_RECENT_SAMPLE_RATIO
        )
        if not batch:
            continue
        states, indices_sparse, counts_sparse, values = batch
        try:
            policy_loss_t, value_loss_t, grad_norm_val, pred_entropy = trainer.train_step(
                (states, indices_sparse, counts_sparse, values)
            )
            grad_norm_sum += float(grad_norm_val)
            entropy_sum += float(pred_entropy)
            losses.append((policy_loss_t, value_loss_t))
        except torch.OutOfMemoryError:
            prev_bs_local = int(trainer.train_batch_size)
            new_bs_local = int(max(int(C.TRAIN.BATCH_SIZE_MIN), prev_bs_local - 1024))
            if new_bs_local < prev_bs_local:
                trainer.train_batch_size = new_bs_local
                trainer.log.info(
                    f"[AUTO    ] OOM encountered; reducing train_batch_size {prev_bs_local} -> {new_bs_local}"
                )
            trainer._oom_cooldown_iters = max(int(trainer._oom_cooldown_iters), 3)
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            with contextlib.suppress(Exception):
                torch.cuda.reset_peak_memory_stats(trainer.device)
            break

    train_elapsed_s = time.time() - t1
    actual_update_steps = len(losses)
    avg_policy_loss = float(torch.stack([pair[0] for pair in losses]).mean().detach().cpu()) if losses else float("nan")
    avg_value_loss = float(torch.stack([pair[1] for pair in losses]).mean().detach().cpu()) if losses else float("nan")
    try:
        buffer_size = int(trainer.selfplay_engine.size())
        buffer_capacity2 = int(trainer.selfplay_engine.get_capacity())
    except Exception:
        buffer_size = 0
        buffer_capacity2 = 1
    buffer_pct2 = (buffer_size / buffer_capacity2) * 100
    batches_per_sec = (len(losses) / max(1e-9, train_elapsed_s)) if losses else 0.0
    samples_per_sec = ((len(losses) * trainer.train_batch_size) / max(1e-9, train_elapsed_s)) if losses else 0.0
    learning_rate = trainer.optimizer.param_groups[0]["lr"]
    avg_grad_norm = (grad_norm_sum / max(1, len(losses))) if losses else 0.0
    avg_entropy = (entropy_sum / max(1, len(losses))) if losses else 0.0
    entropy_coef = 0.0
    if C.TRAIN.LOSS_ENTROPY_COEF_INIT > 0 and trainer.iteration <= C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS:
        entropy_coef = C.TRAIN.LOSS_ENTROPY_COEF_INIT * (
            1.0 - (trainer.iteration - 1) / max(1, C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS)
        )
    if C.TRAIN.LOSS_ENTROPY_COEF_MIN > 0:
        entropy_coef = max(float(entropy_coef), float(C.TRAIN.LOSS_ENTROPY_COEF_MIN))
    sched_drift_pct = 0.0
    if C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST > 0:
        sched_drift_pct = (
            100.0 * (actual_update_steps - C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST) / C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST
            if actual_update_steps > 0
            else 0.0
        )
    if (
        trainer.iteration == 1
        and actual_update_steps > 0
        and abs(sched_drift_pct) > C.TRAIN.LR_SCHED_DRIFT_ADJUST_THRESHOLD
    ):
        remaining_iters = max(0, C.TRAIN.TOTAL_ITERATIONS - trainer.iteration)
        new_total = int(trainer.scheduler.t + remaining_iters * actual_update_steps)
        new_total = max(trainer.scheduler.t + 1, new_total)
        trainer.scheduler.set_total_steps(new_total)
        trainer.log.info(
            f"[LR      ] adjust total_steps -> {trainer.scheduler.total} (iter1 measured {actual_update_steps} vs est {C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST}, drift {sched_drift_pct:+.1f}%)"
        )

    stats.update(
        {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "learning_rate": float(learning_rate),
            "buffer_size": buffer_size,
            "buffer_percent": buffer_pct2,
            "training_time": train_elapsed_s,
            "batches_per_sec": batches_per_sec,
            "optimizer_steps": actual_update_steps,
            "lr_sched_t": trainer.scheduler.t,
            "lr_sched_total": trainer.scheduler.total,
        }
    )

    recent_pct = round(100.0 * C.SAMPLING.TRAIN_RECENT_SAMPLE_RATIO)
    old_pct = max(0, 100 - recent_pct)
    tr_plan_line = (
        f"plan {num_steps} | tgt {C.TRAIN.TARGET_TRAIN_SAMPLES_PER_NEW:.1f}x | act {samples_ratio:>3.1f}x | mix {recent_pct}/{old_pct}%"
        if num_steps > 0
        else "plan 0 skip (buffer underfilled)"
    )
    tr_line = (
        f"steps {len(losses):>3} | b/s {batches_per_sec:>4.1f} | sps {format_si(samples_per_sec, digits=1)} | t {format_time(train_elapsed_s)} | "
        f"P {avg_policy_loss:>6.4f} | V {avg_value_loss:>6.4f} | LR {learning_rate:.2e} | grad {avg_grad_norm:>5.3f} | ent {avg_entropy:>5.3f} | "
        f"entc {entropy_coef:.2e} | clip {C.TRAIN.GRAD_CLIP_NORM:.1f} | buf {int(buffer_pct2):>3}% ({format_si(buffer_size)}) | batch {trainer.train_batch_size}"
    )
    lr_sched_fragment = f"sched {C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST}->{actual_update_steps} | drift {sched_drift_pct:+.0f}% | pos {trainer.scheduler.t}/{format_si(trainer.scheduler.total)}"

    tr_line_parts = tr_line.split(" | ")
    tr_line_step = tr_line_parts[0] if tr_line_parts else ""
    tr_line_rest = " | ".join(tr_line_parts[1:]) if len(tr_line_parts) > 1 else ""
    tr_block = (
        "[Train   ] "
        + tr_plan_line
        + "\n"
        + " " * 10
        + tr_line_step
        + "\n"
        + " " * 10
        + tr_line_rest
        + "\n"
        + " " * 10
        + lr_sched_fragment
    )
    trainer.log.info(tr_block)
    trainer._prev_eval_m = eval_metrics

    return stats


def train_step(
    trainer: Any, batch_data: tuple[list[Any], list[np.ndarray], list[np.ndarray], list[np.ndarray]]
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    states_u8_list, indices_i32_list, counts_u16_list, values_i8_list = batch_data
    x_u8_np = np.stack(states_u8_list).astype(np.uint8, copy=False)
    x_u8_t = torch.from_numpy(x_u8_np)
    if C.TORCH.TRAIN_PIN_MEMORY:
        with contextlib.suppress(Exception):
            x_u8_t = x_u8_t.pin_memory()
    x = x_u8_t.to(trainer.device, non_blocking=True)
    v_i8_np = np.asarray(values_i8_list, dtype=np.int8)
    v_i8_t = torch.from_numpy(v_i8_np)
    if C.TORCH.TRAIN_PIN_MEMORY:
        with contextlib.suppress(Exception):
            v_i8_t = v_i8_t.pin_memory()
    v_target = v_i8_t.to(trainer.device, non_blocking=True).to(dtype=torch.float32) / float(C.DATA.VALUE_I8_SCALE)
    x = x.to(dtype=torch.float32) / float(C.DATA.U8_SCALE)
    x = x.contiguous(memory_format=torch.channels_last)
    if not hasattr(trainer, "_aug_mirror_idx"):
        from augmentation import Augment as _Aug

        mirror_idx = _Aug._policy_index_permutation("mirror")
        rot180_idx = _Aug._policy_index_permutation("rot180")
        vflip_idx = _Aug._policy_index_permutation("vflip_cs")
        plane_perm = _Aug._vflip_cs_plane_permutation(x.shape[1])
        feat_idx = _Aug._feature_plane_indices()
        _tp = feat_idx.get("turn_plane")
        trainer._turn_plane_idx = int(_tp if _tp is not None else int(x.shape[1]))
        trainer._aug_mirror_idx = torch.tensor(mirror_idx, dtype=torch.long, device=trainer.device)
        trainer._aug_rot180_idx = torch.tensor(rot180_idx, dtype=torch.long, device=trainer.device)
        trainer._aug_vflip_idx = torch.tensor(vflip_idx, dtype=torch.long, device=trainer.device)
        trainer._aug_vflip_plane_perm = torch.tensor(plane_perm, dtype=torch.long, device=trainer.device)

        def _inv_perm(perm: list[int]) -> np.ndarray:
            perm_np = np.asarray(perm, dtype=np.int64)
            inv = np.empty_like(perm_np)
            inv[perm_np] = np.arange(perm_np.size, dtype=np.int64)
            return inv

        trainer._aug_mirror_inv_np = _inv_perm(mirror_idx)
        trainer._aug_rot180_inv_np = _inv_perm(rot180_idx)
        trainer._aug_vflip_inv_np = _inv_perm(vflip_idx)

    if np.random.rand() < C.AUGMENT.MIRROR_PROB:
        x = torch.flip(x, dims=[-1])
        indices_i32_list = [
            trainer._aug_mirror_inv_np[np.asarray(idx, dtype=np.int64)].astype(np.int32) for idx in indices_i32_list
        ]
    if np.random.rand() < C.AUGMENT.ROT180_PROB:
        x = torch.flip(x, dims=[-1, -2])
        indices_i32_list = [
            trainer._aug_rot180_inv_np[np.asarray(idx, dtype=np.int64)].astype(np.int32) for idx in indices_i32_list
        ]
    if np.random.rand() < C.AUGMENT.VFLIP_CS_PROB:
        x = torch.flip(x, dims=[-2])
        x = x.index_select(1, trainer._aug_vflip_plane_perm)
        if 0 <= getattr(trainer, "_turn_plane_idx", x.shape[1]) < x.shape[1]:
            x[:, trainer._turn_plane_idx] = 1.0 - x[:, trainer._turn_plane_idx]
        indices_i32_list = [
            trainer._aug_vflip_inv_np[np.asarray(idx, dtype=np.int64)].astype(np.int32) for idx in indices_i32_list
        ]
        v_target = -v_target
    trainer.model.train()
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=C.TORCH.AMP_ENABLED,
    ):
        policy_logits, value_pred = trainer.model(x)
        value_pred = value_pred.squeeze(-1)
        log_probs = F.log_softmax(policy_logits, dim=1)
        rows_np_list: list[np.ndarray] = []
        cols_np_list: list[np.ndarray] = []
        cnts_np_list: list[np.ndarray] = []
        for i, (idx_arr, cnt_arr) in enumerate(zip(indices_i32_list, counts_u16_list, strict=False)):
            idx_np = np.asarray(idx_arr, dtype=np.int64)
            cnt_np = np.asarray(cnt_arr, dtype=np.float32)
            if idx_np.size == 0:
                continue
            valid = (idx_np >= 0) & (idx_np < log_probs.shape[1]) & (cnt_np > 0)
            if not np.any(valid):
                continue
            idx_np = idx_np[valid]
            cnt_np = cnt_np[valid]
            cols_np_list.append(idx_np)
            cnts_np_list.append(cnt_np)
            rows_np_list.append(np.full((idx_np.size,), i, dtype=np.int64))
        if rows_np_list:
            cols_cpu = torch.from_numpy(np.concatenate(cols_np_list, axis=0))
            cnts_cpu = torch.from_numpy(np.concatenate(cnts_np_list, axis=0).astype(np.float32, copy=False))
            rows_cpu = torch.from_numpy(np.concatenate(rows_np_list, axis=0))
            if C.TORCH.TRAIN_PIN_MEMORY:
                with contextlib.suppress(Exception):
                    cols_cpu = cols_cpu.pin_memory()
                    cnts_cpu = cnts_cpu.pin_memory()
                    rows_cpu = rows_cpu.pin_memory()
            cols = cols_cpu.to(trainer.device, non_blocking=True, dtype=torch.long)
            cnts = cnts_cpu.to(trainer.device, non_blocking=True, dtype=torch.float32)
            rows = rows_cpu.to(trainer.device, non_blocking=True, dtype=torch.long)
            denom_rows = torch.zeros((x.shape[0],), device=trainer.device, dtype=torch.float32)
            denom_rows.index_add_(0, rows, cnts)
            denom = denom_rows.index_select(0, rows).clamp_min(1e-9)
            weights = cnts / denom
            eps = float(C.TRAIN.LOSS_POLICY_LABEL_SMOOTH)
            if eps > 0.0:
                ones = torch.ones_like(weights)
                m_rows = torch.zeros((x.shape[0],), device=trainer.device, dtype=torch.float32)
                m_rows.index_add_(0, rows, ones)
                m = m_rows.index_select(0, rows).clamp_min(1.0)
                weights = (1.0 - eps) * weights + (eps / m)
            sel = log_probs[rows, cols]
            per_entry = -weights * sel
            loss_rows = torch.zeros((x.shape[0],), device=trainer.device, dtype=torch.float32)
            loss_rows.index_add_(0, rows, per_entry)
            policy_loss = loss_rows.mean()
        else:
            policy_loss = torch.tensor(0.0, device=trainer.device, dtype=torch.float32)
        value_loss = F.mse_loss(value_pred, v_target)
        entropy_coef = 0.0
        if C.TRAIN.LOSS_ENTROPY_COEF_INIT > 0 and trainer.iteration <= C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS:
            entropy_coef = C.TRAIN.LOSS_ENTROPY_COEF_INIT * (
                1.0 - (trainer.iteration - 1) / max(1, C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS)
            )
        if C.TRAIN.LOSS_ENTROPY_COEF_MIN > 0:
            entropy_coef = max(float(entropy_coef), float(C.TRAIN.LOSS_ENTROPY_COEF_MIN))
        entropy = (-(F.softmax(policy_logits, dim=1) * F.log_softmax(policy_logits, dim=1)).sum(dim=1)).mean()
        value_loss_weight = (
            C.TRAIN.LOSS_VALUE_WEIGHT_LATE
            if trainer.iteration >= C.TRAIN.LOSS_VALUE_WEIGHT_SWITCH_ITER
            else C.TRAIN.LOSS_VALUE_WEIGHT
        )
        total_loss = C.TRAIN.LOSS_POLICY_WEIGHT * policy_loss + value_loss_weight * value_loss - entropy_coef * entropy
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.scaler.scale(total_loss).backward()
    trainer.scaler.unscale_(trainer.optimizer)
    grad_total_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), C.TRAIN.GRAD_CLIP_NORM)
    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    trainer.scheduler.step()
    if trainer.ema is not None:
        trainer.ema.update(trainer.model)
    return (
        policy_loss.detach(),
        value_loss.detach(),
        float(grad_total_norm.detach().cpu()),
        float(entropy.detach().cpu()),
    )
