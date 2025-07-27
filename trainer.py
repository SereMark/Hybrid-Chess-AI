import time
from pathlib import Path

import chess
import numpy as np
import torch
from config import (
    BATCH_SIZE,
    BOARD_SIZE,
    BUFFER_SIZE,
    CACHE_SIZE,
    GAMES_PER_ITER,
    GRADIENT_ACCUMULATION,
    ITERATIONS,
    LEARNING_RATE,
    MAX_MOVES,
    MOVE_COUNT,
    RESIGN_THRESHOLD,
    TEMP_MOVES,
    USE_MIXED_PRECISION,
)
from mcts import MCTS
from model import ChessModel
from move_encoder import MoveEncoder
from torch import optim
from torch.nn import functional
from torch.optim.lr_scheduler import CosineAnnealingLR


class ChessTrainer:
    def __init__(self, device: str):
        self.device = device
        self.move_encoder = MoveEncoder()
        self.model = ChessModel(device).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=ITERATIONS, eta_min=LEARNING_RATE * 0.1
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
        self.iteration_times = []

    def self_play(self) -> tuple[list[tuple], dict[str, float]]:
        games_data = [[] for _ in range(GAMES_PER_ITER)]
        boards = [chess.Board() for _ in range(GAMES_PER_ITER)]
        move_counts = [0] * GAMES_PER_ITER
        resigned_games = [False] * GAMES_PER_ITER
        active = list(range(GAMES_PER_ITER))

        while active:
            remaining = []
            for i in active:
                if boards[i].is_game_over() or move_counts[i] >= MAX_MOVES:
                    continue
                elif self._should_resign(boards[i]):
                    resigned_games[i] = True
                    continue
                remaining.append(i)
            active = remaining

            if not active:
                break

            active_boards = [boards[i] for i in active]
            policies = self.mcts.search_batch(active_boards)

            for idx, game_idx in enumerate(active):
                board = boards[game_idx]
                policy = policies[idx]

                if not policy:
                    continue

                board_tensor = self.model.encode_board(board)
                games_data[game_idx].append(
                    (board_tensor.clone(), policy.copy(), board.turn)
                )

                temp = 1.0 if move_counts[game_idx] < TEMP_MOVES else 0.1
                move = self._sample_move(policy, temp)

                if move in board.legal_moves:
                    board.push(move)
                    move_counts[game_idx] += 1

        self.games_played += GAMES_PER_ITER

        all_data = []
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        completed_games = sum(1 for board in boards if board.is_game_over())
        resigned_count = sum(resigned_games)
        move_limit_count = sum(
            1
            for i in range(GAMES_PER_ITER)
            if move_counts[i] >= MAX_MOVES
            and not boards[i].is_game_over()
            and not resigned_games[i]
        )
        total_moves = sum(len(game_data) for game_data in games_data)
        individual_game_lengths = [len(game_data) for game_data in games_data]
        temp_transitions = sum(
            1 for i in range(GAMES_PER_ITER) if move_counts[i] >= TEMP_MOVES
        )

        for game_idx, board in enumerate(boards):
            result_str = board.result()
            if board.is_game_over():
                results[result_str] += 1

            result_value = {"1-0": 1.0, "0-1": -1.0}.get(result_str, 0.0)
            for tensor, probs, turn in games_data[game_idx]:
                value = result_value if turn == chess.WHITE else -result_value
                all_data.append((tensor, probs, value))

        game_stats = {
            "completed": completed_games,
            "total": GAMES_PER_ITER,
            "wins": results["1-0"],
            "losses": results["0-1"],
            "draws": results["1/2-1/2"],
            "avg_moves": total_moves / GAMES_PER_ITER,
            "resigned": resigned_count,
            "move_limit": move_limit_count,
            "temp_transitions": temp_transitions,
            "individual_lengths": individual_game_lengths,
            "min_moves": min(individual_game_lengths) if individual_game_lengths else 0,
            "max_moves": max(individual_game_lengths) if individual_game_lengths else 0,
        }

        return all_data, game_stats

    def _should_resign(self, board: chess.Board) -> bool:
        if board.is_game_over():
            return False

        with torch.no_grad():
            tensor = self.model.encode_board(board).unsqueeze(0)
            value = self.model(tensor).value.item()
            return (
                (value < RESIGN_THRESHOLD)
                if board.turn
                else (value > -RESIGN_THRESHOLD)
            )

    def _sample_move(
        self, probs: dict[chess.Move, float], temperature: float
    ) -> chess.Move:
        moves = list(probs.keys())
        if not moves:
            raise ValueError("No moves available for sampling")

        values = np.array(list(probs.values()), dtype=np.float32)

        if temperature != 1.0:
            values = np.maximum(values, 1e-10)
            np.power(values, 1.0 / temperature, out=values)

        values_sum = values.sum()
        if values_sum == 0 or not np.isfinite(values_sum):
            values.fill(1.0 / len(values))
        else:
            values /= values_sum

        idx = np.random.choice(len(moves), p=values)
        return moves[idx]

    def add_to_buffer(self, data: list[tuple]) -> None:
        for board_tensor, move_probs, value in data:
            if not move_probs:
                continue

            self.buffer_boards[self.buffer_pos] = board_tensor
            self.buffer_values[self.buffer_pos] = value

            policy_tensor = self.buffer_policies[self.buffer_pos]
            policy_tensor.zero_()
            for move, prob in move_probs.items():
                idx = self.move_encoder.encode_move(move)
                if 0 <= idx < MOVE_COUNT:
                    policy_tensor[idx] = prob

            self.buffer_pos = (self.buffer_pos + 1) % BUFFER_SIZE
            self.buffer_size = min(self.buffer_size + 1, BUFFER_SIZE)

    def train_step(self) -> dict[str, float]:
        if self.buffer_size == 0:
            return {}

        actual_batch_size = min(BATCH_SIZE, self.buffer_size)
        indices = torch.randint(
            0,
            self.buffer_size,
            (actual_batch_size,),
            device=self.device,
        )
        boards = self.buffer_boards[indices]
        values = self.buffer_values[indices]
        policies = self.buffer_policies[indices]

        if self.accumulation_step == 0:
            self.optimizer.zero_grad()

        self.model.train()

        if self.scaler:
            with torch.amp.autocast("cuda"):
                outputs = self.model(boards)
                value_loss = functional.mse_loss(outputs.value.squeeze(), values)
                policy_loss = functional.kl_div(
                    torch.log(outputs.policy + 1e-8), policies, reduction="batchmean"
                )
                total_loss = (value_loss + policy_loss) / GRADIENT_ACCUMULATION
            self.scaler.scale(total_loss).backward()
        else:
            outputs = self.model(boards)
            value_loss = functional.mse_loss(outputs.value.squeeze(), values)
            policy_loss = functional.kl_div(
                torch.log(outputs.policy + 1e-8), policies, reduction="batchmean"
            )
            total_loss = (value_loss + policy_loss) / GRADIENT_ACCUMULATION
            total_loss.backward()

        grad_norm = 0.0
        optimizer_stepped = False
        self.accumulation_step += 1
        if self.accumulation_step >= GRADIENT_ACCUMULATION:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                ).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
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
            "self_play_pct": (self_play_time / iteration_time) * 100,
            "train_pct": (train_time / iteration_time) * 100,
        }

        result = {
            "buffer": self.buffer_size,
        }
        result.update(timing_breakdown)
        result.update(losses)
        result.update(game_stats)
        return result

    def get_detailed_state(self) -> dict[str, float]:
        mcts_stats = self.mcts.get_search_stats()
        model_stats = self.model.get_inference_stats()
        
        recent_times = self.iteration_times[-10:] if len(self.iteration_times) >= 10 else self.iteration_times
        time_trend = 0.0
        if len(recent_times) >= 2:
            time_trend = (recent_times[-1] - recent_times[0]) / len(recent_times)
        
        base_stats = {
            "buffer_usage_pct": (self.buffer_size / BUFFER_SIZE) * 100,
            "buffer_position": self.buffer_pos,
            "buffer_size": self.buffer_size,
            "cache_size": len(self.model.cache),
            "cache_usage_pct": (len(self.model.cache) / CACHE_SIZE) * 100,
            "accumulation_progress_pct": (
                self.accumulation_step / GRADIENT_ACCUMULATION
            )
            * 100,
            "games_played": self.games_played,
            "total_iterations": len(self.iteration_times),
            "time_trend_per_iter": time_trend,
            "avg_recent_iter_time": sum(recent_times) / max(len(recent_times), 1),
        }
        
        result = {}
        result.update(base_stats)
        result.update({f"mcts_{k}": v for k, v in mcts_stats.items()})
        result.update({f"model_{k}": v for k, v in model_stats.items()})
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
