from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

import chesscore as chessai
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler

from .config import CONFIG
from .model import (
    BatchedEvaluator,
    ChessNet,
    get_module_state_dict,
    load_module_state_dict,
)

log = logging.getLogger(__name__)


class Augment:
    @staticmethod
    def mirror_policy(pi: np.ndarray) -> np.ndarray:
        arr = pi.reshape((73, 8, 8))
        arr = arr[:, :, ::-1].copy()
        out = np.empty_like(arr)
        dir_map = [2, 1, 0, 4, 3, 7, 6, 5]
        for d in range(8):
            for r in range(7):
                old_idx = d * 7 + r
                new_idx = dir_map[d] * 7 + r
                out[new_idx] = arr[old_idx]
        kmap = [1, 0, 3, 2, 5, 4, 7, 6]
        for i in range(8):
            out[56 + kmap[i]] = arr[56 + i]
        pmap = [0, 2, 1]
        for p in range(3):
            base = 64 + p * 3
            out[base + pmap[0]] = arr[base + 0]
            out[base + pmap[1]] = arr[base + 1]
            out[base + pmap[2]] = arr[base + 2]
        return out.reshape(-1)

    @staticmethod
    def rotate180_policy(pi: np.ndarray) -> np.ndarray:
        arr = pi.reshape((73, 8, 8))
        arr = arr[:, ::-1, ::-1].copy()
        out = np.empty_like(arr)
        dir_map = [7, 6, 5, 4, 3, 2, 1, 0]
        for d in range(8):
            for r in range(7):
                old_idx = d * 7 + r
                new_idx = dir_map[d] * 7 + r
                out[new_idx] = arr[old_idx]
        kmap = [7, 6, 5, 4, 3, 2, 1, 0]
        for i in range(8):
            out[56 + kmap[i]] = arr[56 + i]
        for p in range(3):
            base = 64 + p * 3
            out[base + 0] = arr[base + 0]
            out[base + 1] = arr[base + 1]
            out[base + 2] = arr[base + 2]
        return out.reshape(-1)

    @staticmethod
    def vflip_colorswap_policy(pi: np.ndarray) -> np.ndarray:
        arr = pi.reshape((73, 8, 8))
        arr = arr[:, ::-1, :].copy()
        out = np.empty_like(arr)
        dir_map = [5, 6, 7, 4, 3, 2, 1, 0]
        for d in range(8):
            for r in range(7):
                old_idx = d * 7 + r
                new_idx = dir_map[d] * 7 + r
                out[new_idx] = arr[old_idx]
        kmap = [6, 7, 4, 5, 2, 3, 0, 1]
        for i in range(8):
            out[56 + kmap[i]] = arr[56 + i]
        for p in range(3):
            base = 64 + p * 3
            out[base + 0] = arr[base + 0]
            out[base + 1] = arr[base + 1]
            out[base + 2] = arr[base + 2]
        return out.reshape(-1)

    @staticmethod
    def apply(
        states: list[np.ndarray], policies: list[np.ndarray], which: str
    ) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
        if which == "mirror":
            states = [s[..., ::-1].copy() for s in states]
            policies = [Augment.mirror_policy(p) for p in policies]
            return states, policies, False
        elif which == "rot180":
            states = [s[..., ::-1, ::-1].copy() for s in states]
            policies = [Augment.rotate180_policy(p) for p in policies]
            return states, policies, False
        elif which == "vflip_cs":
            out_states: list[np.ndarray] = []
            for s in states:
                x = s[..., ::-1, :].copy()
                planes_per_pos = CONFIG.planes_per_position
                hist_len = CONFIG.history_length
                for t in range(hist_len):
                    base = t * planes_per_pos
                    for piece in range(6):
                        a = base + piece * 2 + 0
                        b = base + piece * 2 + 1
                        xa = x[a].copy()
                        x[a] = x[b]
                        x[b] = xa
                turn_plane = hist_len * planes_per_pos
                x[turn_plane] = 1.0 - x[turn_plane]
                cs_base = turn_plane + 2
                xa = x[cs_base + 0].copy()
                x[cs_base + 0] = x[cs_base + 2]
                x[cs_base + 2] = xa
                xb = x[cs_base + 1].copy()
                x[cs_base + 1] = x[cs_base + 3]
                x[cs_base + 3] = xb
                out_states.append(x)
            policies = [Augment.vflip_colorswap_policy(p) for p in policies]
            return out_states, policies, True
        return states, policies, False


class SelfPlayEngine:
    def __init__(self, evaluator: BatchedEvaluator) -> None:
        self.evaluator = evaluator
        self.buffer: deque[tuple[Any, np.ndarray, float]] = deque(
            maxlen=CONFIG.buffer_size
        )
        self.buffer_lock = threading.Lock()

    def _temp_select(
        self, moves: list[Any], visits: list[int], move_number: int
    ) -> Any:
        if move_number < CONFIG.temp_moves:
            temperature = CONFIG.temp_high
        else:
            temperature = CONFIG.temp_low
        if temperature > 0.01:
            probs = np.array(visits, dtype=np.float64) ** (1.0 / temperature)
            probs /= probs.sum()
            move_idx = np.random.choice(len(moves), p=probs)
        else:
            move_idx = int(np.argmax(visits))
        return moves[move_idx]

    def _process_result(self, data: list[tuple[Any, np.ndarray]], result: int) -> None:
        if result == chessai.WHITE_WIN:
            values = [1.0 if i % 2 == 0 else -1.0 for i in range(len(data))]
        elif result == chessai.BLACK_WIN:
            values = [-1.0 if i % 2 == 0 else 1.0 for i in range(len(data))]
        else:
            values = [0.0] * len(data)
        with self.buffer_lock:
            for (position, policy), value in zip(data, values):
                self.buffer.append((position, policy, value))

    def play_single_game(self) -> tuple[int, int]:
        position = chessai.Position()
        setattr(self, "_resign_count", 0)
        mcts = chessai.MCTS(
            CONFIG.simulations_train,
            CONFIG.c_puct,
            CONFIG.dirichlet_alpha,
            CONFIG.dirichlet_weight,
        )
        data: list[tuple[Any, np.ndarray]] = []
        history: list[Any] = []
        move_count = 0
        while position.result() == chessai.ONGOING and move_count < 512:
            pos_copy = chessai.Position(position)
            sims = max(
                CONFIG.mcts_min_sims, CONFIG.simulations_train // (1 + move_count // 40)
            )
            mcts.set_simulations(sims)
            visits = mcts.search_batched(
                position, self.evaluator.infer_positions, CONFIG.eval_max_batch
            )
            if not visits:
                break

            moves = position.legal_moves()
            target = np.zeros(CONFIG.policy_output, dtype=np.float32)
            for move, visit_count in zip(moves, visits):
                move_index = chessai.encode_move_index(move)
                if (move_index is not None) and (
                    0 <= int(move_index) < CONFIG.policy_output
                ):
                    target[move_index] = visit_count
            policy_sum = target.sum()
            if policy_sum > 0:
                target /= policy_sum

            if history:
                histories = history[-CONFIG.history_length :] + [pos_copy]
                encoded = chessai.encode_batch([histories])[0]
            else:
                encoded = chessai.encode_position(pos_copy)
            encoded = encoded.astype(np.float16, copy=False)
            target = target.astype(np.float16, copy=False)
            data.append((encoded, target))
            if CONFIG.resign_consecutive > 0:
                _, val_arr = self.evaluator.infer_positions([pos_copy])
                value_est = float(val_arr[0])
                if value_est <= CONFIG.resign_threshold:
                    consecutive = getattr(self, "_resign_count", 0) + 1
                    setattr(self, "_resign_count", consecutive)
                    if consecutive >= CONFIG.resign_consecutive:
                        break
                else:
                    setattr(self, "_resign_count", 0)

            move = self._temp_select(moves, visits, move_count)
            position.make_move(move)
            history.append(pos_copy)
            if len(history) > CONFIG.history_length:
                history.pop(0)
            move_count += 1

        self._process_result(data, position.result())
        return move_count, position.result()

    def snapshot(self) -> list[tuple[Any, np.ndarray, float]]:
        with self.buffer_lock:
            return list(self.buffer)

    def sample_from_snapshot(
        self, snapshot: list[tuple[Any, np.ndarray, float]], batch_size: int
    ):
        if len(snapshot) < batch_size:
            return None
        idx = np.random.randint(0, len(snapshot), size=batch_size)
        batch = [snapshot[int(i)] for i in idx]
        s, p, v = zip(*batch)
        return list(s), list(p), list(v)

    def play_games(self, num_games: int) -> dict[str, int | float]:
        results: dict[str, int | float] = {
            "games": 0,
            "moves": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
        }
        workers = max(1, CONFIG.selfplay_workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self.play_single_game) for _ in range(num_games)]
            for idx, fut in enumerate(as_completed(futures), 1):
                moves, result = fut.result()
                results["games"] += 1
                results["moves"] += moves
                if result == chessai.WHITE_WIN:
                    results["white_wins"] += 1
                elif result == chessai.BLACK_WIN:
                    results["black_wins"] += 1
                else:
                    results["draws"] += 1
        return results


class AlphaZeroTrainer:
    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = ChessNet().to(self.device)
        if CONFIG.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)  # type: ignore[reportCallIssue]
        if CONFIG.use_torch_compile:
            self.model = cast(torch.nn.Module, torch.compile(self.model))
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
        with torch.amp.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
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
            if step % 256 == 0:
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

    def save_checkpoint(self, filepath: str | None = None) -> None:
        if filepath is None:
            filepath = f"checkpoint_{self.iteration}.pt"
        checkpoint = {
            "iteration": self.iteration,
            "total_games": self.total_games,
            "model_state_dict": get_module_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_time": time.time() - self.start_time,
        }
        torch.save(checkpoint, filepath)
        checkpoint_size = os.path.getsize(filepath) / 1024**2
        log.info(
            "Checkpoint: %s | %.1fMB | Games %s | Time %s",
            filepath,
            checkpoint_size,
            f"{self.total_games:,}",
            self._format_time(time.time() - self.start_time),
        )

    def _clone_model(self) -> torch.nn.Module:
        clone = ChessNet().to(self.device)
        load_module_state_dict(clone, get_module_state_dict(self.model), strict=True)
        clone.eval()
        return clone

    def _arena_match(
        self, challenger: torch.nn.Module, incumbent: torch.nn.Module
    ) -> float:
        wins = 0
        import chesscore as _chessai

        challenger_eval = BatchedEvaluator(self.device)
        load_module_state_dict(
            challenger_eval.eval_model, get_module_state_dict(challenger), strict=True
        )
        challenger_eval.eval_model.eval()
        incumbent_eval = BatchedEvaluator(self.device)
        load_module_state_dict(
            incumbent_eval.eval_model, get_module_state_dict(incumbent), strict=True
        )
        incumbent_eval.eval_model.eval()

        def play(e1: BatchedEvaluator, e2: BatchedEvaluator) -> int:
            pos = _chessai.Position()
            mcts1 = _chessai.MCTS(
                CONFIG.simulations_eval,
                CONFIG.c_puct,
                CONFIG.dirichlet_alpha,
                CONFIG.arena_dirichlet_weight,
            )
            mcts2 = _chessai.MCTS(
                CONFIG.simulations_eval,
                CONFIG.c_puct,
                CONFIG.dirichlet_alpha,
                CONFIG.arena_dirichlet_weight,
            )
            turn = 0
            while pos.result() == _chessai.ONGOING and turn < 512:
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
                if CONFIG.arena_temperature > 0.0 and turn < CONFIG.arena_temp_moves:
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
            if res == _chessai.WHITE_WIN:
                return 1
            if res == _chessai.BLACK_WIN:
                return -1
            return 0

        for g in range(CONFIG.arena_games):
            if g % 2 == 0:
                r = play(challenger_eval, incumbent_eval)
            else:
                r = -play(incumbent_eval, challenger_eval)
            if r > 0:
                wins += 1
        challenger_eval.shutdown()
        incumbent_eval.shutdown()
        return wins / max(1, CONFIG.arena_games)

    def load_checkpoint(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.iteration = int(checkpoint["iteration"])  # type: ignore[index]
        self.total_games = int(checkpoint["total_games"])  # type: ignore[index]
        load_module_state_dict(self.model, checkpoint["model_state_dict"])  # type: ignore[index]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # type: ignore[index]
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # type: ignore[index]
        log.info("Loaded checkpoint from %s (iteration %d)", filepath, self.iteration)

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
                self.save_checkpoint()
            challenger = self._clone_model()
            arena_start = time.time()
            win_rate = self._arena_match(challenger, self.best_model)
            arena_elapsed = time.time() - arena_start
            log.info(
                "Arena: win_rate %.1f%% over %d games | time %s",
                100.0 * win_rate,
                CONFIG.arena_games,
                self._format_time(arena_elapsed),
            )
            if win_rate >= CONFIG.arena_win_rate:
                load_module_state_dict(
                    self.best_model,
                    get_module_state_dict(challenger),
                    strict=True,
                )
                self.best_model.eval()
                self.evaluator.refresh_from(self.best_model)
            full_iter_time = time.time() - iter_start
            self.iter_times.append(full_iter_time)
            if self.iter_ema_time is None:
                self.iter_ema_time = full_iter_time
            else:
                alpha = 0.3
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    trainer = AlphaZeroTrainer()
    trainer.train()
