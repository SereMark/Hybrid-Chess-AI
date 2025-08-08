from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import chessai

from .config import CONFIG
from .inference import BatchedEvaluator, PositionEncoder
from .nn import ChessNet


log = logging.getLogger(__name__)


class SelfPlayEngine:
    def __init__(self, evaluator: BatchedEvaluator) -> None:
        self.evaluator = evaluator
        self.buffer: deque[tuple[Any, np.ndarray, float]] = deque(
            maxlen=CONFIG.buffer_size
        )
        self.buffer_lock = threading.Lock()

    def evaluate_position(self, position: Any) -> tuple[np.ndarray, float]:
        policy, value = self.evaluator.evaluate(position)
        return policy, value

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
        mcts = chessai.MCTS(
            CONFIG.simulations_train,
            CONFIG.c_puct,
            CONFIG.dirichlet_alpha,
            CONFIG.dirichlet_weight,
        )
        data: list[tuple[Any, np.ndarray]] = []
        move_count = 0
        while position.result() == chessai.ONGOING and move_count < 512:
            pos_copy = chessai.Position(position)
            policy, value = self.evaluate_position(position)
            sims = max(
                CONFIG.mcts_min_sims, CONFIG.simulations_train // (1 + move_count // 40)
            )
            mcts.set_simulations(sims)
            visits = mcts.search(position, policy, value)
            if not visits:
                break

            moves = position.legal_moves()
            target = np.zeros(CONFIG.policy_output, dtype=np.float32)
            for move, visit_count in zip(moves, visits):
                move_index = chessai.encode_move_index(move)
                if move_index is not None and move_index < CONFIG.policy_output:
                    target[move_index] = visit_count
            policy_sum = target.sum()
            if policy_sum > 0:
                target /= policy_sum

            data.append((pos_copy, target))
            move = self._temp_select(moves, visits, move_count)
            position.make_move(move)
            move_count += 1

        self._process_result(data, position.result())
        return move_count, position.result()

    def generate_batch(self, batch_size: int | None = None):
        if batch_size is None:
            batch_size = CONFIG.batch_size
        with self.buffer_lock:
            snapshot = list(self.buffer)
        if len(snapshot) < batch_size:
            return None
        indices = np.random.choice(len(snapshot), batch_size)
        batch = [snapshot[int(i)] for i in indices]
        states, policies, values = zip(*batch)
        return list(states), list(policies), list(values)

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
                if (idx) % max(1, num_games // 20) == 0 and num_games > 100:
                    progress = (idx) / num_games * 100
                    log.info("Progress: %d/%d (%.0f%%)", idx, num_games, progress)
        return results


class AlphaZeroTrainer:
    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        if CONFIG.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)  # type: ignore[reportCallIssue]
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
        self.scaler: torch.cuda.amp.GradScaler | None = None
        if self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.evaluator = BatchedEvaluator(self.device)
        self.evaluator.refresh_from(self.model)
        self.selfplay_engine = SelfPlayEngine(self.evaluator)
        self.encoder = PositionEncoder()
        self.iteration = 0
        self.total_games = 0
        self.start_time = time.time()
        self.iter_times: list[float] = []
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
        x = torch.from_numpy(self.encoder.encode_batch(states)).to(self.device)
        if CONFIG.use_channels_last:
            x = x.to(memory_format=torch.channels_last)  # type: ignore[reportCallIssue]
        pi_target = torch.from_numpy(np.stack(policies).astype(np.float32)).to(
            self.device
        )
        v_target = torch.tensor(values, dtype=torch.float32).to(self.device)

        self.model.train()
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            pi_pred, v_pred = self.model(x)
            policy_loss = F.kl_div(
                F.log_softmax(pi_pred, dim=1), pi_target, reduction="batchmean"
            )
            value_loss = F.mse_loss(v_pred, v_target)
            total_loss = (
                CONFIG.policy_weight * policy_loss + CONFIG.value_weight * value_loss
            )

        self.optimizer.zero_grad()
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
        log.info(
            "[Iter %d/%d] LR %.2e | GPU %.1f/%.1fGB | Buffer %d%%",
            self.iteration,
            CONFIG.iterations,
            current_lr,
            mem["allocated_gb"],
            mem["reserved_gb"],
            int(buffer_pct_pre),
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
        log.info(
            "Self-play: %d games | %.1f games/min | %.1fK moves/sec | avg %.1f moves",
            game_stats["games"],
            gpm,
            mps / 1000,
            avg_len,
        )

        stats.update(game_stats)
        stats["selfplay_time"] = elapsed
        stats["games_per_min"] = gpm
        stats["moves_per_sec"] = mps

        train_start = time.time()
        losses: list[tuple[float, float]] = []
        for step in range(CONFIG.train_steps_per_iter):
            batch = self.selfplay_engine.generate_batch(CONFIG.batch_size)
            if batch:
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
                "Training: %d steps | %.1f batches/sec | %d samples/sec | Loss P %.4f V %.4f | LR %.2e | Buffer %d%%",
                len(losses),
                batches_per_sec,
                int(samples_per_sec),
                pol_loss,
                val_loss,
                current_lr,
                int(buffer_pct),
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
        self.scheduler.step()

        iter_total_time = time.time() - total_iter_start
        self.iter_times.append(iter_total_time)
        if self.device.type == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1024**3
            peak_res = torch.cuda.max_memory_reserved(self.device) / 1024**3
            log.info(
                "Totals: Iter %s | Cumulative games %s | Peak GPU %.1f/%.1fGB",
                self._format_time(iter_total_time),
                f"{self.total_games:,}",
                peak_alloc,
                peak_res,
            )
        else:
            log.info(
                "Totals: Iter %s | Cumulative games %s",
                self._format_time(iter_total_time),
                f"{self.total_games:,}",
            )
        return stats

    def save_checkpoint(self, filepath: str | None = None) -> None:
        if filepath is None:
            filepath = f"checkpoint_{self.iteration}.pt"
        checkpoint = {
            "iteration": self.iteration,
            "total_games": self.total_games,
            "model_state_dict": self.model.state_dict(),
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

    def load_checkpoint(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.iteration = int(checkpoint["iteration"])  # type: ignore[index]
        self.total_games = int(checkpoint["total_games"])  # type: ignore[index]
        self.model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[index]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # type: ignore[index]
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # type: ignore[index]
        log.info("Loaded checkpoint from %s (iteration %d)", filepath, self.iteration)

    def train(self) -> None:
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        log.info("Starting training...")
        if self.device.type == "cuda":
            log.info(
                "Device: %s (%s) | GPU total %.1fGB | AMP %s",
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
            _ = self.training_iteration()
            if iteration % CONFIG.checkpoint_freq == 0:
                self.save_checkpoint()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def main() -> None:
    setup_logging()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    trainer = AlphaZeroTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
