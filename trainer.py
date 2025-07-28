import contextlib
import time
from pathlib import Path

import chess
import torch
from config import (
    BATCH_SIZE,
    BOARD_SIZE,
    BUFFER_SIZE,
    GAMES_PER_ITER,
    GRADIENT_ACCUMULATION,
    GRADIENT_CLIP_NORM,
    ITERATIONS,
    LEARNING_RATE,
    LEARNING_RATE_MIN_FACTOR,
    MAX_MOVES,
    MOVE_COUNT,
    POLICY_EPSILON,
    TEMP_MOVES,
    USE_MIXED_PRECISION,
)
from mcts import MCTS
from model import ChessModel
from move_encoder import MoveEncoder
from torch import optim
from torch.nn import functional
from torch.optim.lr_scheduler import CosineAnnealingLR
from worker import (
    distribute_games_to_workers,
    execute_worker_games,
    init_worker_process,
)


class ChessTrainer:
    def __init__(self, device: str):
        self.device = device
        self.move_encoder = MoveEncoder()
        self.model = ChessModel(device).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=ITERATIONS,
            eta_min=LEARNING_RATE * LEARNING_RATE_MIN_FACTOR,
        )
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if USE_MIXED_PRECISION and device == "cuda"
            else None
        )

        self.buffer_size = 0
        self.buffer_pos = 0
        self.buffer_boards = torch.zeros(
            (BUFFER_SIZE, BOARD_SIZE), dtype=torch.float32, device=device
        )
        self.buffer_values = torch.zeros(
            BUFFER_SIZE, dtype=torch.float32, device=device
        )
        self.buffer_policies = torch.zeros(
            (BUFFER_SIZE, MOVE_COUNT), dtype=torch.float32, device=device
        )

        self.mcts = MCTS(self.model, self.move_encoder)
        self.games_played = 0
        self.accumulation_step = 0
        self.training_start_time = time.time()
        self.iteration_times: list[float] = []

    def process_worker_results(
        self, worker_results
    ) -> tuple[list[tuple], dict[str, int | float]]:
        combined_stats: dict[str, int | float] = {
            "completed": 0,
            "total": GAMES_PER_ITER,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "resigned": 0,
            "move_limit": 0,
            "temp_transitions": 0,
        }
        all_game_lengths = []
        all_processed_data = []

        for worker_result in worker_results:
            data_idx = 0

            for game_result in worker_result["game_results"]:
                if game_result.get("error", False):
                    continue

                expected_count = game_result["move_count"]
                actual_count = min(
                    expected_count, len(worker_result["game_data"]) - data_idx
                )

                if game_result["is_game_over"]:
                    combined_stats["completed"] += 1
                    result_str = game_result["result"]
                    if result_str == "1-0":
                        combined_stats["wins"] += 1
                    elif result_str == "0-1":
                        combined_stats["losses"] += 1
                    elif result_str == "1/2-1/2":
                        combined_stats["draws"] += 1
                    result_value = {"1-0": 1.0, "0-1": -1.0}.get(result_str, 0.0)
                elif game_result["resigned"]:
                    combined_stats["resigned"] += 1
                    combined_stats["completed"] += 1

                    if actual_count == 0:
                        last_turn_white = True
                    else:
                        last_turn_white = (
                            worker_result["game_data"][data_idx + actual_count - 1][2]
                            == chess.WHITE
                        )

                    if last_turn_white:
                        combined_stats["losses"] += 1
                    else:
                        combined_stats["wins"] += 1
                    result_value = -1.0 if last_turn_white else 1.0
                elif game_result["move_count"] >= MAX_MOVES:
                    combined_stats["move_limit"] += 1
                    result_value = 0.0
                else:
                    result_value = 0.0

                actual_game_length = min(
                    game_result["move_count"],
                    len(worker_result["game_data"]) - data_idx,
                )
                all_game_lengths.append(actual_game_length)
                if game_result["move_count"] >= TEMP_MOVES:
                    combined_stats["temp_transitions"] += 1

                for _ in range(actual_count):
                    if data_idx < len(worker_result["game_data"]):
                        tensor, probs, turn = worker_result["game_data"][data_idx]
                        value = result_value if turn == chess.WHITE else -result_value
                        all_processed_data.append((tensor, probs, value))
                        data_idx += 1

        if all_game_lengths:
            combined_stats["avg_moves"] = sum(all_game_lengths) / len(all_game_lengths)
            combined_stats["min_moves"] = min(all_game_lengths)
            combined_stats["max_moves"] = max(all_game_lengths)
        else:
            combined_stats["avg_moves"] = 0
            combined_stats["min_moves"] = 0
            combined_stats["max_moves"] = 0

        total_simulations = sum(
            wr["mcts_stats"]["total_simulations"] for wr in worker_results
        )
        total_nodes_expanded = sum(
            wr["mcts_stats"]["nodes_expanded"] for wr in worker_results
        )
        terminal_nodes_hit = sum(
            wr["mcts_stats"]["terminal_nodes_hit"] for wr in worker_results
        )
        model_forward_calls = sum(
            wr["mcts_stats"]["model_forward_calls"] for wr in worker_results
        )
        searches_performed = sum(
            wr["mcts_stats"]["searches_performed"] for wr in worker_results
        )

        self.mcts.total_simulations = total_simulations
        self.mcts.total_nodes_expanded = total_nodes_expanded
        self.mcts.terminal_nodes_hit = terminal_nodes_hit
        self.mcts.model_forward_calls = model_forward_calls
        self.mcts.searches_performed = searches_performed

        total_forward_calls = sum(
            wr["model_stats"]["forward_calls"] for wr in worker_results
        )
        total_cache_hits = sum(wr["model_stats"]["cache_hits"] for wr in worker_results)
        total_cache_misses = sum(
            wr["model_stats"]["cache_misses"] for wr in worker_results
        )

        self.model.forward_calls = total_forward_calls
        self.model.cache_hits = total_cache_hits
        self.model.cache_misses = total_cache_misses

        return all_processed_data, combined_stats

    def self_play_parallel(self) -> tuple[list[tuple], dict[str, float]]:
        try:
            with contextlib.suppress(RuntimeError):
                torch.multiprocessing.set_start_method("spawn", force=True)

            model_state_dict = self.model.state_dict()
            device_str = self.device

            game_assignments = distribute_games_to_workers()
            optimal_processes = len(game_assignments)

            with torch.multiprocessing.Pool(
                processes=optimal_processes,
                initializer=init_worker_process,
                initargs=(model_state_dict, device_str),
            ) as pool:
                worker_results = pool.map(execute_worker_games, game_assignments)

            processed_data, aggregated_stats = self.process_worker_results(
                worker_results
            )

            self.games_played += GAMES_PER_ITER

            return processed_data, aggregated_stats

        except Exception as e:
            raise RuntimeError(
                "Parallel self-play failed and no fallback available"
            ) from e

    def self_play(self) -> tuple[list[tuple], dict[str, float]]:
        return self.self_play_parallel()

    def add_to_buffer(self, data: list[tuple]) -> None:
        for board_tensor, move_probs, value in data:
            if not move_probs:
                continue

            self.buffer_boards[self.buffer_pos] = board_tensor
            self.buffer_values[self.buffer_pos] = value

            policy_tensor = self.buffer_policies[self.buffer_pos]
            policy_tensor.zero_()
            valid_indices = []
            valid_probs = []
            for move, prob in move_probs.items():
                idx = self.move_encoder.encode_move(move)
                if 0 <= idx < MOVE_COUNT:
                    valid_indices.append(idx)
                    valid_probs.append(prob)

            if valid_indices:
                policy_tensor[valid_indices] = torch.tensor(
                    valid_probs, device=self.device
                )

            self.buffer_pos = (self.buffer_pos + 1) % BUFFER_SIZE
            self.buffer_size = min(self.buffer_size + 1, BUFFER_SIZE)

    def train_step(self) -> dict[str, float]:
        if self.buffer_size == 0:
            return {}

        actual_batch_size = min(BATCH_SIZE, self.buffer_size)
        indices = torch.randint(
            0, self.buffer_size, (actual_batch_size,), device=self.device
        )
        boards = self.buffer_boards[indices]
        values = self.buffer_values[indices]
        policies = self.buffer_policies[indices]

        if self.accumulation_step == 0:
            self.optimizer.zero_grad()

        self.model.train()

        with torch.amp.autocast("cuda", enabled=self.scaler is not None):
            outputs = self.model(boards)
            value_loss = functional.mse_loss(outputs.value.squeeze(), values)
            policy_loss = functional.kl_div(
                torch.log(outputs.policy + POLICY_EPSILON),
                policies,
                reduction="batchmean",
            )
            total_loss = (value_loss + policy_loss) / GRADIENT_ACCUMULATION

        if self.scaler:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        grad_norm = 0.0
        optimizer_stepped = False
        self.accumulation_step += 1
        if self.accumulation_step >= GRADIENT_ACCUMULATION:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=GRADIENT_CLIP_NORM
                ).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=GRADIENT_CLIP_NORM
                ).item()
                self.optimizer.step()
            self.accumulation_step = 0
            optimizer_stepped = True

        return {
            "loss": (value_loss + policy_loss).item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "actual_batch_size": actual_batch_size,
            "accumulation_step": self.accumulation_step,
            "optimizer_stepped": optimizer_stepped,
        }

    def iteration(self) -> dict[str, float]:
        start = time.time()

        self.mcts.reset_search_stats()
        self.model.reset_inference_stats()

        self_play_start = time.time()
        game_data, game_stats = self.self_play()
        self_play_time = time.time() - self_play_start

        buffer_start = time.time()
        self.add_to_buffer(game_data)
        buffer_time = time.time() - buffer_start

        train_start = time.time()
        losses = self.train_step()
        train_time = time.time() - train_start

        if self.accumulation_step == 0:
            self.scheduler.step()

        iteration_time = time.time() - start
        self.iteration_times.append(iteration_time)

        timing_breakdown = {
            "time": iteration_time,
            "self_play_time": self_play_time,
            "buffer_time": buffer_time,
            "train_time": train_time,
            "other_time": iteration_time - (self_play_time + buffer_time + train_time),
            "self_play_pct": (self_play_time / max(iteration_time, 0.001)) * 100,
            "train_pct": (train_time / max(iteration_time, 0.001)) * 100,
        }

        result: dict[str, float] = {
            "buffer": float(self.buffer_size),
        }
        result.update(timing_breakdown)
        result.update(losses)
        result.update(game_stats)
        return result

    def save(self, path: Path) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "games_played": self.games_played,
            },
            path,
        )
