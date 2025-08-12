from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import CONFIG
from .model import BatchedEvaluator, ChessNet
from .selfplay import Augment, SelfPlayEngine


class Trainer:
    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            device = "cuda"
        if not torch.cuda.is_available() or str(device) != "cuda":
            raise RuntimeError("CUDA device required")
        self.device = torch.device("cuda")
        self.model = ChessNet().to(self.device)
        if CONFIG.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=CONFIG.learning_rate_init,
            momentum=CONFIG.momentum,
            weight_decay=CONFIG.weight_decay,
            nesterov=True,
        )
        schedule_map = {m: lr for m, lr in CONFIG.learning_rate_schedule}
        self._schedule_map = schedule_map

        def lr_lambda(epoch: int) -> float:
            target = CONFIG.learning_rate_init
            for m in sorted(schedule_map):
                if epoch >= m:
                    target = schedule_map[m]
            return target / CONFIG.learning_rate_init

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        self.evaluator = BatchedEvaluator(self.device)
        self.evaluator.refresh_from(self.model)
        self.best_model = self._clone_model()
        self.selfplay_engine = SelfPlayEngine(self.evaluator)
        self.iteration = 0
        self.total_games = 0
        self.start_time = time.time()
        self.iter_ema_time: float | None = None
        self.ema_sp_time: float | None = None
        self.ema_tr_time: float | None = None
        self.ema_ar_time: float | None = None
        self.ema_ck_time: float | None = None
        self.device_name: str | None = None
        self.device_total_gb = 0.0
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
        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1024**3,
            "total_gb": self.device_total_gb,
        }

    def _clone_model(self) -> torch.nn.Module:
        clone = ChessNet().to(self.device)
        clone.load_state_dict(self.model.state_dict(), strict=True)
        clone.eval()
        return clone

    def train_step(
        self, batch_data: tuple[list[Any], list[np.ndarray], list[float]]
    ) -> tuple[float, float]:
        states, policies, values = batch_data
        x_np = np.stack(states).astype(np.float32, copy=False)
        x_cpu = torch.from_numpy(x_np)
        x_cpu = x_cpu.pin_memory()
        x = x_cpu.to(self.device, non_blocking=True)
        if CONFIG.use_channels_last:
            x = x.to(memory_format=torch.channels_last)

        pi_cpu = torch.from_numpy(np.stack(policies).astype(np.float32))
        pi_cpu = pi_cpu.pin_memory()
        pi_target = pi_cpu.to(self.device, non_blocking=True)

        v_cpu = torch.tensor(values, dtype=torch.float32)
        v_cpu = v_cpu.pin_memory()
        v_target = v_cpu.to(self.device, non_blocking=True)

        self.model.train()
        with torch.autocast(device_type="cuda", enabled=True):
            pi_pred, v_pred = self.model(x)
            policy_loss = F.kl_div(
                F.log_softmax(pi_pred, dim=1), pi_target, reduction="batchmean"
            )
            value_loss = F.mse_loss(v_pred, v_target)
            total_loss = (
                CONFIG.policy_weight * policy_loss + CONFIG.value_weight * value_loss
            )

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(policy_loss.item()), float(value_loss.item())

    def training_iteration(self) -> dict[str, int | float]:
        stats: dict[str, int | float] = {}
        torch.cuda.reset_peak_memory_stats(self.device)
        current_lr = self.optimizer.param_groups[0]["lr"]
        mem = self._get_mem_info()
        buffer_len_pre = len(self.selfplay_engine.buffer)
        buffer_pct_pre = (buffer_len_pre / CONFIG.buffer_size) * 100
        total_elapsed = time.time() - self.start_time
        next_lr_iter = None
        next_lr_val = None
        if self._schedule_map:
            future = [m for m in sorted(self._schedule_map) if m > self.iteration]
            if future:
                next_lr_iter = future[0]
                next_lr_val = self._schedule_map[next_lr_iter]
        pct_done = 100.0 * (self.iteration - 1) / max(1, CONFIG.iterations)
        print(
            "\n[Iter %4d/%d | %5.1f%%] | LR % .2e%s | Elapsed %s"
            % (
                self.iteration,
                CONFIG.iterations,
                pct_done,
                current_lr,
                (
                    (" -> %d:%.2e" % (next_lr_iter, next_lr_val))
                    if next_lr_iter is not None
                    else ""
                ),
                self._format_time(total_elapsed),
            )
        )
        print(
            "GPU %4.1f/%4.1f/%4.1f GB | Buffer %10s/%-10s (%3d%%)"
            % (
                mem["allocated_gb"],
                mem["reserved_gb"],
                mem["total_gb"],
                f"{buffer_len_pre:,}",
                f"{CONFIG.buffer_size:,}",
                int(buffer_pct_pre),
            )
        )
        print(
            "CFG  workers %2d | sims T/E %3d/%-3d | eval B %4d/%2dms | arena every %2d"
            % (
                CONFIG.selfplay_workers,
                CONFIG.simulations_train,
                CONFIG.simulations_eval,
                CONFIG.eval_max_batch,
                CONFIG.eval_batch_timeout_ms,
                CONFIG.arena_eval_every,
            )
        )

        selfplay_start = time.time()
        game_stats = self.selfplay_engine.play_games(CONFIG.games_per_iteration)
        self.total_games += int(game_stats["games"])
        elapsed = time.time() - selfplay_start
        gpm = (game_stats["games"] / (elapsed / 60)) if elapsed > 0 else 0
        mps = (game_stats["moves"] / elapsed) if elapsed > 0 else 0
        avg_len = (
            (game_stats["moves"] / max(1, game_stats["games"])) if elapsed > 0 else 0
        )
        ww = int(game_stats["white_wins"])
        bb = int(game_stats["black_wins"])
        dd = int(game_stats["draws"])
        gc = int(game_stats["games"])
        if gc > 0:
            wpct = 100.0 * ww / gc
            dpct = 100.0 * dd / gc
            bpct = 100.0 * bb / gc
        else:
            wpct = dpct = bpct = 0.0
        print(
            "SP   games %5d | gpm %4.1f | mps %5.1fK | avg_len %4.1f | W/D/B %4d/%-4d/%-4d (%3.0f%%/%-3.0f%%/%-3.0f%%) | time %s"
            % (
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
            buffer_size = len(self.selfplay_engine.buffer)
            buffer_pct = (buffer_size / CONFIG.buffer_size) * 100
            batches_per_sec = len(losses) / train_elapsed if train_elapsed > 0 else 0
            samples_per_sec = (
                (len(losses) * CONFIG.batch_size) / train_elapsed
                if train_elapsed > 0
                else 0
            )
            print(
                "TR   steps %5d | batch/s %5.1f | samp/s %7d | P %7.4f | V %7.4f | LR % .2e | buf %3d%% (%s) | time %s"
                % (
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
        challenger_eval.eval_model.load_state_dict(challenger.state_dict(), strict=True)
        challenger_eval.eval_model.eval()
        incumbent_eval = _BatchedEval(self.device)
        incumbent_eval.eval_model.load_state_dict(incumbent.state_dict(), strict=True)
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
            mcts1.set_c_puct_params(CONFIG.c_puct_base, CONFIG.c_puct_init)
            mcts2 = _ccore.MCTS(
                CONFIG.simulations_eval,
                CONFIG.c_puct,
                CONFIG.dirichlet_alpha,
                noise_w,
            )
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
        score = (wins + CONFIG.arena_draw_score * draws) / max(1, CONFIG.arena_games)
        return score, wins, draws, losses

    def train(self) -> None:
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print("\nStarting training...")
        print(
            "\nDevice: %s (%s) | GPU total %.1fGB | AMP on"
            % (
                self.device,
                self.device_name,
                self.device_total_gb,
            )
        )
        print(
            "Model: %.1fM parameters | %d blocks x %d channels"
            % (total_params, CONFIG.blocks, CONFIG.channels)
        )
        print(
            "Training: %d iterations | %d games/iter"
            % (CONFIG.iterations, CONFIG.games_per_iteration)
        )
        sched_pairs = ", ".join(
            [f"{m} -> {lr:.1e}" for m, lr in CONFIG.learning_rate_schedule]
        )
        print("LR: init %.2e | schedule: %s" % (CONFIG.learning_rate_init, sched_pairs))
        print(
            "Expected: %d total games"
            % (CONFIG.iterations * CONFIG.games_per_iteration)
        )
        print(
            "Config: workers %d | sims_train %d | sims_eval %d | eval_batch %d/%dms | cache %d | arena %d/every %d | gate z %.2f thr %.2f | channels_last %s"
            % (
                CONFIG.selfplay_workers,
                CONFIG.simulations_train,
                CONFIG.simulations_eval,
                CONFIG.eval_max_batch,
                CONFIG.eval_batch_timeout_ms,
                CONFIG.eval_cache_size,
                CONFIG.arena_games,
                CONFIG.arena_eval_every,
                CONFIG.arena_confidence_z,
                max(CONFIG.arena_threshold_base, CONFIG.arena_win_rate),
                str(bool(CONFIG.use_channels_last)),
            )
        )

        for iteration in range(1, CONFIG.iterations + 1):
            self.iteration = iteration
            iter_stats = self.training_iteration()
            if iteration % CONFIG.checkpoint_freq == 0:
                ck_start = time.time()
                checkpoint = {
                    "iteration": self.iteration,
                    "total_games": self.total_games,
                    "model_state_dict": (
                        self.model.module.state_dict()
                        if hasattr(self.model, "module")
                        else self.model.state_dict()
                    ),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "training_time": time.time() - self.start_time,
                }
                filepath = f"checkpoint_{self.iteration}.pt"
                torch.save(checkpoint, filepath)
                size_mb = os.path.getsize(filepath) / 1024**2
                ck_elapsed = time.time() - ck_start
                if self.ema_ck_time is None:
                    self.ema_ck_time = ck_elapsed
                else:
                    a = CONFIG.iteration_ema_alpha
                    self.ema_ck_time = a * ck_elapsed + (1 - a) * self.ema_ck_time
                print(
                    "Checkpoint: %s | %.1fMB | Games %s | Time %s"
                    % (
                        filepath,
                        size_mb,
                        f"{self.total_games:,}",
                        self._format_time(time.time() - self.start_time),
                    )
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
                        print(
                            "AR   score %5.1f%% | cur LB %5.1f%% | acc LB %5.1f%% (%4d) | W/D/L %4d/%-4d/%-4d | games %4d | time %s"
                            % (
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
                        )
                    else:
                        print(
                            "AR   score %5.1f%% | LB %5.1f%% | W/D/L %4d/%-4d/%-4d | games %4d | time %s"
                            % (
                                100.0 * win_rate,
                                100.0 * cur_lb,
                                aw,
                                ad,
                                al,
                                CONFIG.arena_games,
                                self._format_time(arena_elapsed),
                            )
                        )
                else:
                    print(
                        "AR   score %5.1f%% | W/D/L %4d/%-4d/%-4d | games %4d | time %s"
                        % (
                            100.0 * win_rate,
                            aw,
                            ad,
                            al,
                            CONFIG.arena_games,
                            self._format_time(arena_elapsed),
                        )
                    )

                if promote:
                    self.best_model.load_state_dict(
                        challenger.state_dict(), strict=True
                    )
                    self.best_model.eval()
                    self.evaluator.refresh_from(self.best_model)
                    try:
                        torch.save(self.best_model.state_dict(), "best_model.pt")
                        print("Saved best model to best_model.pt")
                    except Exception as e:
                        print("Warning: failed to save best model: %s" % (e,))
                    if hasattr(self, "_arena_acc_w"):
                        self._arena_acc_w = 0
                        self._arena_acc_d = 0
                        self._arena_acc_l = 0

            else:
                print(
                    "AR   skipped | games    0 | time %s"
                    % (self._format_time(arena_elapsed),)
                )

            full_iter_time = (
                float(iter_stats.get("selfplay_time", 0.0))
                + float(iter_stats.get("training_time", 0.0))
                + arena_elapsed
            )

            sp_time = float(iter_stats.get("selfplay_time", 0.0))
            tr_time = float(iter_stats.get("training_time", 0.0))
            if self.iter_ema_time is None:
                self.iter_ema_time = full_iter_time
            else:
                alpha = CONFIG.iteration_ema_alpha
                self.iter_ema_time = (
                    alpha * full_iter_time + (1 - alpha) * self.iter_ema_time
                )
            if self.ema_sp_time is None:
                self.ema_sp_time = sp_time
            else:
                a = CONFIG.iteration_ema_alpha
                self.ema_sp_time = a * sp_time + (1 - a) * self.ema_sp_time
            if self.ema_tr_time is None:
                self.ema_tr_time = tr_time
            else:
                a = CONFIG.iteration_ema_alpha
                self.ema_tr_time = a * tr_time + (1 - a) * self.ema_tr_time
            if arena_elapsed > 0.0:
                if self.ema_ar_time is None:
                    self.ema_ar_time = arena_elapsed
                else:
                    a = CONFIG.iteration_ema_alpha
                    self.ema_ar_time = a * arena_elapsed + (1 - a) * self.ema_ar_time
            avg_iter = (
                self.iter_ema_time if self.iter_ema_time is not None else full_iter_time
            )
            remaining = max(0, CONFIG.iterations - self.iteration)
            eta_sec = avg_iter * remaining
            if CONFIG.arena_eval_every > 0:
                k = CONFIG.arena_eval_every
                rem_arena = ((self.iteration + remaining) // k) - (self.iteration // k)
                sp_est = self.ema_sp_time if self.ema_sp_time is not None else sp_time
                tr_est = self.ema_tr_time if self.ema_tr_time is not None else tr_time
                ar_est = (
                    self.ema_ar_time if self.ema_ar_time is not None else arena_elapsed
                )
                eta_sec = remaining * (sp_est + tr_est) + max(0, rem_arena) * max(
                    0.0, ar_est
                )
            if CONFIG.checkpoint_freq > 0:
                k = CONFIG.checkpoint_freq
                rem_ck = ((self.iteration + remaining) // k) - (self.iteration // k)
                ck_est = self.ema_ck_time if self.ema_ck_time is not None else 0.0
                eta_sec += max(0, rem_ck) * max(0.0, ck_est)
            next_ar = 0
            if CONFIG.arena_eval_every > 0:
                k = CONFIG.arena_eval_every
                r = self.iteration % k
                next_ar = (k - r) if r != 0 else k
            next_ck = 0
            if CONFIG.checkpoint_freq > 0:
                k = CONFIG.checkpoint_freq
                r = self.iteration % k
                next_ck = (k - r) if r != 0 else k
            brk = f"(sp {self._format_time(sp_time)} + tr {self._format_time(tr_time)} + ar {self._format_time(arena_elapsed)})"
            peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1024**3
            peak_res = torch.cuda.max_memory_reserved(self.device) / 1024**3
            print("SUM  iter %8s | %s" % (self._format_time(full_iter_time), brk))
            print(
                "     avg %9s | elapsed %9s | ETA %9s | next_ar %3d | next_ck %3d | EMA sp/tr/ar %s/%s/%s"
                % (
                    self._format_time(avg_iter),
                    self._format_time(time.time() - self.start_time),
                    self._format_time(eta_sec),
                    next_ar,
                    next_ck,
                    self._format_time(self.ema_sp_time) if self.ema_sp_time else "-",
                    self._format_time(self.ema_tr_time) if self.ema_tr_time else "-",
                    self._format_time(self.ema_ar_time) if self.ema_ar_time else "-",
                )
            )
            print(
                "     games %11s | peak GPU %4.1f/%-4.1f GB"
                % (
                    f"{self.total_games:,}",
                    peak_alloc,
                    peak_res,
                )
            )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = CONFIG.use_cudnn_benchmark
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = bool(getattr(CONFIG, "allow_tf32", True))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(CONFIG.matmul_precision)
    torch.set_num_threads(CONFIG.torch_num_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(getattr(CONFIG, "torch_num_interop_threads", 1))
    if getattr(CONFIG, "use_cuda_alloc_tuning", False):
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "max_split_size_mb:128,garbage_collection_threshold:0.8",
        )
    Trainer().train()
