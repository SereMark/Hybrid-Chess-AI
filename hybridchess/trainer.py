from __future__ import annotations

import logging
import os
import time
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler

from .config import CONFIG
from .model import (BatchedEvaluator, ChessNet, get_module_state_dict,
                    load_module_state_dict)
from .selfplay import Augment, SelfPlayEngine

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = ChessNet().to(self.device)
        if CONFIG.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)  # type: ignore[reportCallIssue]
        if CONFIG.use_torch_compile:
            self.model = cast(
                torch.nn.Module,
                torch.compile(
                    self.model,
                    backend=getattr(CONFIG, "compile_backend", "inductor"),
                    mode=getattr(CONFIG, "compile_mode_train", "default"),
                    fullgraph=getattr(CONFIG, "compile_fullgraph_train", False),
                    dynamic=getattr(CONFIG, "compile_dynamic", False),
                ),
            )
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=CONFIG.learning_rate_init,
            momentum=CONFIG.momentum,
            weight_decay=CONFIG.weight_decay,
            nesterov=True,
        )
        schedule_map = {m: lr for m, lr in CONFIG.learning_rate_schedule}

        def lr_lambda(epoch: int) -> float:
            target = CONFIG.learning_rate_init
            for m in sorted(schedule_map):
                if epoch >= m:
                    target = schedule_map[m]
            return target / CONFIG.learning_rate_init

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler: GradScaler | None = None
        if self.device.type == "cuda":
            self.scaler = GradScaler(enabled=True)
        self.evaluator = BatchedEvaluator(self.device)
        self.evaluator.refresh_from(self.model)
        self.best_model = self._clone_model()
        self.selfplay_engine = SelfPlayEngine(self.evaluator)
        self.iteration = 0
        self.total_games = 0
        self.start_time = time.time()
        self.iter_times: list[float] = []
        self.iter_ema_time: float | None = None
        self.device_name: str | None = None
        self.device_total_gb = 0.0
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            self.device_name = props.name
            self.device_total_gb = props.total_memory / 1024**3

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def _get_mem_info(self) -> dict[str, float]:
        if self.device.type == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated(self.device) / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved(self.device) / 1024**3,
                "total_gb": self.device_total_gb,
            }
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0}

    def _clone_model(self) -> torch.nn.Module:
        clone = ChessNet().to(self.device)
        load_module_state_dict(clone, get_module_state_dict(self.model), strict=True)
        clone.eval()
        return clone

    def train_step(
        self, batch_data: tuple[list[Any], list[np.ndarray], list[float]]
    ) -> tuple[float, float]:
        states, policies, values = batch_data
        x_np = np.stack(states).astype(np.float32, copy=False)
        x_cpu = torch.from_numpy(x_np)
        if self.device.type == "cuda":
            x_cpu = x_cpu.pin_memory()
        x = x_cpu.to(self.device, non_blocking=True)
        if CONFIG.use_channels_last:
            x = x.to(memory_format=torch.channels_last)  # type: ignore[reportCallIssue]

        pi_cpu = torch.from_numpy(np.stack(policies).astype(np.float32))
        if self.device.type == "cuda":
            pi_cpu = pi_cpu.pin_memory()
        pi_target = pi_cpu.to(self.device, non_blocking=True)

        v_cpu = torch.tensor(values, dtype=torch.float32)
        if self.device.type == "cuda":
            v_cpu = v_cpu.pin_memory()
        v_target = v_cpu.to(self.device, non_blocking=True)

        self.model.train()
        with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
            pi_pred, v_pred = self.model(x)
            policy_loss = F.kl_div(
                F.log_softmax(pi_pred, dim=1), pi_target, reduction="batchmean"
            )
            value_loss = F.mse_loss(v_pred, v_target)
            total_loss = (
                CONFIG.policy_weight * policy_loss + CONFIG.value_weight * value_loss
            )

        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), CONFIG.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), CONFIG.gradient_clip
            )
            self.optimizer.step()
        return float(policy_loss.item()), float(value_loss.item())

    def training_iteration(self) -> dict[str, int | float]:
        stats: dict[str, int | float] = {}
        total_iter_start = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        current_lr = self.optimizer.param_groups[0]["lr"]
        mem = self._get_mem_info()
        buffer_pct_pre = (
            self.selfplay_engine.buffer.__len__() / CONFIG.buffer_size
        ) * 100
        buffer_len_pre = self.selfplay_engine.buffer.__len__()
        total_elapsed = time.time() - self.start_time
        log.info(
            "\n[Iter %d/%d] lr %.2e | gpu %.1f/%.1f/%.1f GB | buf %d%% (%s) | elapsed %s",
            self.iteration,
            CONFIG.iterations,
            current_lr,
            mem["allocated_gb"],
            mem["reserved_gb"],
            mem["total_gb"],
            int(buffer_pct_pre),
            f"{buffer_len_pre:,}",
            self._format_time(total_elapsed),
        )

        selfplay_start = time.time()
        game_stats = self.selfplay_engine.play_games(CONFIG.games_per_iteration)
        self.total_games += int(game_stats["games"])  # type: ignore[index]
        elapsed = time.time() - selfplay_start
        gpm = (game_stats["games"] / (elapsed / 60)) if elapsed > 0 else 0
        mps = (game_stats["moves"] / elapsed) if elapsed > 0 else 0
        avg_len = (
            (game_stats["moves"] / max(1, game_stats["games"])) if elapsed > 0 else 0
        )
        ww = int(game_stats["white_wins"])  # type: ignore[index]
        bb = int(game_stats["black_wins"])  # type: ignore[index]
        dd = int(game_stats["draws"])  # type: ignore[index]
        gc = int(game_stats["games"])  # type: ignore[index]
        if gc > 0:
            wpct = 100.0 * ww / gc
            dpct = 100.0 * dd / gc
            bpct = 100.0 * bb / gc
        else:
            wpct = dpct = bpct = 0.0
        log.info(
            "Self-play: games %d | gpm %.1f | mps %.1fK | avg_len %.1f | W/D/B %d/%d/%d (%.0f%%/%.0f%%/%.0f%%) | time %s",
            gc,
            gpm,
            mps / 1000,
            avg_len,
            ww,
            dd,
            bb,
            wpct,
            dpct,
            bpct,
            self._format_time(elapsed),
        )

        stats.update(game_stats)
        stats["selfplay_time"] = elapsed
        stats["games_per_min"] = gpm
        stats["moves_per_sec"] = mps

        train_start = time.time()
        losses: list[tuple[float, float]] = []
        buffer_snapshot = self.selfplay_engine.snapshot()
        for step in range(CONFIG.train_steps_per_iter):
            batch = self.selfplay_engine.sample_from_snapshot(
                buffer_snapshot, CONFIG.batch_size
            )
            if batch:
                s, p, v = batch
                did_aug = False
                if (
                    CONFIG.augment_mirror
                    and np.random.rand() < CONFIG.augment_mirror_prob
                ):
                    s, p, _ = Augment.apply(s, p, "mirror")
                    did_aug = True
                if (
                    CONFIG.augment_rotate180
                    and np.random.rand() < CONFIG.augment_rot180_prob
                ):
                    s, p, _ = Augment.apply(s, p, "rot180")
                    did_aug = True
                if (
                    CONFIG.augment_vflip_cs
                    and np.random.rand() < CONFIG.augment_vflip_cs_prob
                ):
                    s, p, cs = Augment.apply(s, p, "vflip_cs")
                    if cs:
                        v = [-val for val in v]
                    did_aug = True
                if did_aug:
                    batch = (s, p, v)
                loss_vals = self.train_step(batch)
                losses.append(loss_vals)
            if step % CONFIG.eval_refresh_steps == 0:
                self.evaluator.refresh_from(self.model)
        train_elapsed = time.time() - train_start
        if losses:
            pol_loss = float(np.mean([loss[0] for loss in losses]))
            val_loss = float(np.mean([loss[1] for loss in losses]))
            buffer_size = self.selfplay_engine.buffer.__len__()
            buffer_pct = (buffer_size / CONFIG.buffer_size) * 100
            batches_per_sec = len(losses) / train_elapsed if train_elapsed > 0 else 0
            samples_per_sec = (
                (len(losses) * CONFIG.batch_size) / train_elapsed
                if train_elapsed > 0
                else 0
            )
            log.info(
                "Train: steps %d | %.1f batch/s | %d samp/s | P %.4f | V %.4f | lr %.2e | buf %d%% (%s) | time %s",
                len(losses),
                batches_per_sec,
                int(samples_per_sec),
                pol_loss,
                val_loss,
                current_lr,
                int(buffer_pct),
                f"{buffer_size:,}",
                self._format_time(train_elapsed),
            )
            stats.update(
                {
                    "policy_loss": pol_loss,
                    "value_loss": val_loss,
                    "learning_rate": current_lr,
                    "buffer_size": buffer_size,
                    "buffer_percent": buffer_pct,
                    "training_time": train_elapsed,
                    "batches_per_sec": batches_per_sec,
                    "total_iteration_time": time.time() - total_iter_start,
                }
            )

        self.evaluator.refresh_from(self.model)
        self.scheduler.step()

        return stats

    def _arena_match(
        self, challenger: torch.nn.Module, incumbent: torch.nn.Module
    ) -> tuple[float, int, int, int]:
        import chesscore as _ccore

        from .model import BatchedEvaluator as _BatchedEval

        wins = draws = losses = 0

        challenger_eval = _BatchedEval(self.device)
        load_module_state_dict(
            challenger_eval.eval_model, get_module_state_dict(challenger), strict=True
        )
        challenger_eval.eval_model.eval()
        incumbent_eval = _BatchedEval(self.device)
        load_module_state_dict(
            incumbent_eval.eval_model, get_module_state_dict(incumbent), strict=True
        )
        incumbent_eval.eval_model.eval()

        openings: list[str] = []
        if CONFIG.arena_openings_path:
            try:
                with open(CONFIG.arena_openings_path, "r", encoding="utf-8") as f:
                    openings = [line.strip() for line in f if line.strip()]
            except Exception:
                openings = []
        if not openings:
            openings = [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
                "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",
                "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
                "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            ]

        def play(e1: _BatchedEval, e2: _BatchedEval, start_fen: str | None) -> int:
            pos = _ccore.Position()
            if start_fen:
                pos.from_fen(start_fen)
            noise_w = (
                0.0 if not CONFIG.arena_use_noise else CONFIG.arena_dirichlet_weight
            )
            mcts1 = _ccore.MCTS(
                CONFIG.simulations_eval,
                CONFIG.c_puct,
                CONFIG.dirichlet_alpha,
                noise_w,
            )
            if hasattr(mcts1, "set_c_puct_params"):
                mcts1.set_c_puct_params(CONFIG.c_puct_base, CONFIG.c_puct_init)
            mcts2 = _ccore.MCTS(
                CONFIG.simulations_eval,
                CONFIG.c_puct,
                CONFIG.dirichlet_alpha,
                noise_w,
            )
            if hasattr(mcts2, "set_c_puct_params"):
                mcts2.set_c_puct_params(CONFIG.c_puct_base, CONFIG.c_puct_init)
            turn = 0
            while pos.result() == _ccore.ONGOING and turn < CONFIG.max_game_moves:
                if turn % 2 == 0:
                    visits = mcts1.search_batched(
                        pos, e1.infer_positions, CONFIG.eval_max_batch
                    )
                else:
                    visits = mcts2.search_batched(
                        pos, e2.infer_positions, CONFIG.eval_max_batch
                    )
                if not visits:
                    break
                moves = pos.legal_moves()
                if turn < CONFIG.arena_opening_random_plies:
                    v = np.asarray(visits, dtype=np.float64)
                    v = np.maximum(v, 0)
                    if v.sum() <= 0:
                        idx = int(np.argmax(visits))
                    else:
                        temp = max(
                            CONFIG.arena_opening_temperature_epsilon,
                            float(CONFIG.arena_opening_temperature),
                        )
                        probs = v ** (1.0 / temp)
                        s = probs.sum()
                        if s <= 0:
                            idx = int(np.argmax(visits))
                        else:
                            probs /= s
                            idx = int(np.random.choice(len(moves), p=probs))
                elif (
                    CONFIG.arena_use_noise
                    and CONFIG.arena_temperature > 0.0
                    and turn < CONFIG.arena_temp_moves
                ):
                    v = np.asarray(visits, dtype=np.float64)
                    v = np.maximum(v, 0)
                    if v.sum() <= 0:
                        idx = int(np.argmax(visits))
                    else:
                        probs = v ** (1.0 / float(CONFIG.arena_temperature))
                        s = probs.sum()
                        if s <= 0:
                            idx = int(np.argmax(visits))
                        else:
                            probs /= s
                            idx = int(np.random.choice(len(moves), p=probs))
                else:
                    idx = int(np.argmax(visits))
                pos.make_move(moves[idx])
                turn += 1
            res = pos.result()
            if res == _ccore.WHITE_WIN:
                return 1
            if res == _ccore.BLACK_WIN:
                return -1
            return 0

        for g in range(CONFIG.arena_games):
            start_fen: str | None = None
            if openings:
                if CONFIG.arena_openings_random:
                    start_fen = openings[np.random.randint(0, len(openings))]
                else:
                    start_fen = openings[g % len(openings)]
            if g % 2 == 0:
                r = play(challenger_eval, incumbent_eval, start_fen)
            else:
                r = -play(incumbent_eval, challenger_eval, start_fen)
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1
        challenger_eval.shutdown()
        incumbent_eval.shutdown()
        score = (wins + CONFIG.arena_draw_score * draws) / max(1, CONFIG.arena_games)
        return score, wins, draws, losses

    def train(self) -> None:
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        log.info("\nStarting training...")
        if self.device.type == "cuda":
            log.info(
                "\nDevice: %s (%s) | GPU total %.1fGB | AMP %s",
                self.device,
                self.device_name,
                self.device_total_gb,
                "on" if self.scaler else "off",
            )
        else:
            log.info("Device: %s | AMP off", self.device)
        log.info(
            "Model: %.1fM parameters | %d blocks x %d channels",
            total_params,
            CONFIG.blocks,
            CONFIG.channels,
        )
        log.info(
            "Training: %d iterations | %d games/iter",
            CONFIG.iterations,
            CONFIG.games_per_iteration,
        )
        sched_pairs = ", ".join(
            [f"{m} -> {lr:.1e}" for m, lr in CONFIG.learning_rate_schedule]
        )
        log.info("LR: init %.2e | schedule: %s", CONFIG.learning_rate_init, sched_pairs)
        log.info(
            "Expected: %d total games", CONFIG.iterations * CONFIG.games_per_iteration
        )

        for iteration in range(1, CONFIG.iterations + 1):
            self.iteration = iteration
            iter_start = time.time()
            iter_stats = self.training_iteration()
            if iteration % CONFIG.checkpoint_freq == 0:

                checkpoint = {
                    "iteration": self.iteration,
                    "total_games": self.total_games,
                    "model_state_dict": get_module_state_dict(self.model),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "training_time": time.time() - self.start_time,
                }
                filepath = f"checkpoint_{self.iteration}.pt"
                torch.save(checkpoint, filepath)
                size_mb = os.path.getsize(filepath) / 1024**2
                log.info(
                    "Checkpoint: %s | %.1fMB | Games %s | Time %s",
                    filepath,
                    size_mb,
                    f"{self.total_games:,}",
                    self._format_time(time.time() - self.start_time),
                )

            do_eval = (iteration % CONFIG.arena_eval_every) == 0
            arena_elapsed = 0.0
            if do_eval:
                challenger = self._clone_model()
                arena_start = time.time()
                win_rate, aw, ad, al = self._arena_match(challenger, self.best_model)
                arena_elapsed = time.time() - arena_start

                if CONFIG.arena_accumulate:
                    self._arena_acc_w = getattr(self, "_arena_acc_w", 0) + aw
                    self._arena_acc_d = getattr(self, "_arena_acc_d", 0) + ad
                    self._arena_acc_l = getattr(self, "_arena_acc_l", 0) + al
                    gw = self._arena_acc_w
                    gd = self._arena_acc_d
                    gl = self._arena_acc_l
                    gn = max(1, gw + gd + gl)
                    acc_mu = (gw + CONFIG.arena_draw_score * gd) / gn
                    gating_mu = acc_mu
                    gating_n = gn
                else:
                    gating_mu = (aw + CONFIG.arena_draw_score * ad) / max(
                        1, CONFIG.arena_games
                    )
                    gating_n = CONFIG.arena_games

                promote = False
                threshold = max(CONFIG.arena_threshold_base, CONFIG.arena_win_rate)
                if CONFIG.arena_confidence:
                    n = gating_n
                    mu = gating_mu
                    e_x2 = (
                        (self._arena_acc_w if CONFIG.arena_accumulate else aw)
                        + 0.25 * (self._arena_acc_d if CONFIG.arena_accumulate else ad)
                    ) / max(1, n)
                    var = max(0.0, e_x2 - mu * mu)
                    se = (var / max(1, n)) ** 0.5
                    lb = mu - CONFIG.arena_confidence_z * se
                    promote = lb >= threshold
                else:
                    promote = gating_mu >= threshold

                if CONFIG.arena_confidence:
                    cur_n = CONFIG.arena_games
                    cur_mu = (aw + CONFIG.arena_draw_score * ad) / max(1, cur_n)
                    cur_e_x2 = (aw + 0.25 * ad) / max(1, cur_n)
                    cur_var = max(0.0, cur_e_x2 - cur_mu * cur_mu)
                    cur_se = (cur_var / max(1, cur_n)) ** 0.5
                    cur_lb = cur_mu - CONFIG.arena_confidence_z * cur_se

                    if CONFIG.arena_accumulate:
                        acc_n = gating_n
                        acc_mu = gating_mu
                        acc_e_x2 = (self._arena_acc_w + 0.25 * self._arena_acc_d) / max(
                            1, acc_n
                        )
                        acc_var = max(0.0, acc_e_x2 - acc_mu * acc_mu)
                        acc_se = (acc_var / max(1, acc_n)) ** 0.5
                        acc_lb = acc_mu - CONFIG.arena_confidence_z * acc_se
                        log.info(
                            "Arena: score %.1f%% (cur LB %.1f%%, acc LB %.1f%% over %d) | W/D/L %d/%d/%d | games %d | time %s",
                            100.0 * win_rate,
                            100.0 * cur_lb,
                            100.0 * acc_lb,
                            acc_n,
                            aw,
                            ad,
                            al,
                            CONFIG.arena_games,
                            self._format_time(arena_elapsed),
                        )
                    else:
                        log.info(
                            "Arena: score %.1f%% (LB %.1f%%) | W/D/L %d/%d/%d | games %d | time %s",
                            100.0 * win_rate,
                            100.0 * cur_lb,
                            aw,
                            ad,
                            al,
                            CONFIG.arena_games,
                            self._format_time(arena_elapsed),
                        )
                else:
                    log.info(
                        "Arena: score %.1f%% | W/D/L %d/%d/%d | games %d | time %s",
                        100.0 * win_rate,
                        aw,
                        ad,
                        al,
                        CONFIG.arena_games,
                        self._format_time(arena_elapsed),
                    )

                if promote:
                    load_module_state_dict(
                        self.best_model,
                        get_module_state_dict(challenger),
                        strict=True,
                    )
                    self.best_model.eval()
                    self.evaluator.refresh_from(self.best_model)
                    try:
                        torch.save(
                            get_module_state_dict(self.best_model), "best_model.pt"
                        )
                        log.info("Saved best model to best_model.pt")
                    except Exception as e:
                        log.info("Warning: failed to save best model: %s", e)
                    if hasattr(self, "_arena_acc_w"):
                        self._arena_acc_w = 0
                        self._arena_acc_d = 0
                        self._arena_acc_l = 0
                    if hasattr(self, "_sprt_state"):
                        self._sprt_state = {"llr": 0.0, "games": 0}
            else:
                log.info(
                    "Arena: skipped | games 0 | time %s",
                    self._format_time(arena_elapsed),
                )

            full_iter_time = time.time() - iter_start
            self.iter_times.append(full_iter_time)
            if self.iter_ema_time is None:
                self.iter_ema_time = full_iter_time
            else:
                alpha = CONFIG.iteration_ema_alpha
                self.iter_ema_time = (
                    alpha * full_iter_time + (1 - alpha) * self.iter_ema_time
                )
            avg_iter = (
                self.iter_ema_time if self.iter_ema_time is not None else full_iter_time
            )
            remaining = max(0, CONFIG.iterations - self.iteration)
            eta_sec = avg_iter * remaining
            sp_time = float(iter_stats.get("selfplay_time", 0.0))
            tr_time = float(iter_stats.get("training_time", 0.0))
            brk = f"(sp {self._format_time(sp_time)} + tr {self._format_time(tr_time)} + ar {self._format_time(arena_elapsed)})"
            if self.device.type == "cuda":
                peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1024**3
                peak_res = torch.cuda.max_memory_reserved(self.device) / 1024**3
                log.info(
                    "Totals: iter %s %s | avg %s | elapsed %s | ETA %s | games %s | peak_gpu %.1f/%.1f GB",
                    self._format_time(full_iter_time),
                    brk,
                    self._format_time(avg_iter),
                    self._format_time(time.time() - self.start_time),
                    self._format_time(eta_sec),
                    f"{self.total_games:,}",
                    peak_alloc,
                    peak_res,
                )
            else:
                log.info(
                    "Totals: iter %s %s | avg %s | elapsed %s | ETA %s | games %s",
                    self._format_time(full_iter_time),
                    brk,
                    self._format_time(avg_iter),
                    self._format_time(time.time() - self.start_time),
                    self._format_time(eta_sec),
                    f"{self.total_games:,}",
                )
