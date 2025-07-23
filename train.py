import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import chess
import numpy as np
import torch
from config import config
from engine import MCTS, ChessModel, MoveEncoder, uniform_probs
from torch import nn, optim
from torch.nn import functional as f
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass
class IterationMetrics:
    training_loss: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0

    games_completed: int = 0
    games_timeout: int = 0
    games_total: int = 0
    avg_game_length: float = 0.0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    buffer_size: int = 0
    games_played_total: int = 0
    training_steps_total: int = 0
    iteration_time: float = 0.0
    gpu_memory_gb: float = 0.0
    mcts_searches: int = 0
    zero_move_games: int = 0

    def __post_init__(self):
        self.completion_rate = (
            (self.games_completed / self.games_total * 100)
            if self.games_total > 0
            else 0.0
        )
        self.win_rate = (
            (self.wins / self.games_total * 100) if self.games_total > 0 else 0.0
        )
        self.buffer_utilization = (
            (self.buffer_size / config.training.buffer_size * 100)
            if config.training.buffer_size > 0
            else 0.0
        )

    def log_summary(self, iteration: int) -> None:
        sections = [
            f"=== ITERATION {iteration:3d}/{config.training.iterations} ===",
            f"Loss: {self.training_loss:.4f} (val:{self.value_loss:.4f} pol:{self.policy_loss:.4f}) | Grad: {self.grad_norm:.3f} | LR: {self.learning_rate:.6f}",
            f"Games: {self.games_completed}/{self.games_total} completed ({self.completion_rate:.1f}%) | Timeout: {self.games_timeout} | Avg Length: {self.avg_game_length:.1f}",
        ]

        if self.wins + self.losses + self.draws > 0:
            sections.append(
                f"Results: W:{self.wins} L:{self.losses} D:{self.draws} | Win Rate: {self.win_rate:.1f}%"
            )

        sections.extend(
            [
                f"Buffer: {self.buffer_size:,}/{config.training.buffer_size:,} ({self.buffer_utilization:.1f}%) | Steps: {self.training_steps_total:,} | Games Total: {self.games_played_total:,}",
                self._format_timing_info(),
            ]
        )

        print("\n" + "\n".join(sections))

    def _format_timing_info(self) -> str:
        info = [f"Time: {self.iteration_time:.1f}s"]
        if self.gpu_memory_gb > 0:
            info.append(f"GPU: {self.gpu_memory_gb:.1f}GB")
        if self.mcts_searches > 0:
            info.append(f"MCTS: {self.mcts_searches} searches")
        if self.zero_move_games > 0:
            info.append(f"Zero-move: {self.zero_move_games}")
        return " | ".join(info)


class State(TypedDict):
    board_tensor: torch.Tensor
    move_probs: dict[chess.Move, float]
    move_played: chess.Move | None
    player: chess.Color


class Position(TypedDict):
    board_tensor: torch.Tensor
    move_probs: dict[chess.Move, float]
    value: float
    move_played: chess.Move | None


def _calculate_temperature(
    move_count: int, temp_moves: int, high_temp: float, low_temp: float
) -> float:
    if move_count < temp_moves:
        progress = move_count / temp_moves
        return high_temp * (1 - progress) + low_temp * progress
    return low_temp


def get_temp(move_count: int) -> float:
    return _calculate_temperature(
        move_count,
        config.game.temp_moves,
        config.game.high_temperature,
        config.game.low_temperature,
    )


def _sample_move_index(probs: np.ndarray, temperature: float) -> int:
    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)

    probs_sum = np.sum(probs)
    if probs_sum > 0:
        probs = probs / probs_sum
    else:
        probs = np.ones_like(probs) / len(probs)

    cumsum = np.cumsum(probs)
    rand = np.random.random()
    return np.searchsorted(cumsum, rand)


def sample_move(
    move_probs: dict[chess.Move, float], temperature: float = 1.0
) -> chess.Move:
    if not move_probs:
        raise ValueError("No moves available for selection")

    moves = list(move_probs.keys())
    probs = np.array(list(move_probs.values()), dtype=np.float32)

    idx = _sample_move_index(probs, temperature)
    return moves[idx]


class Game:
    def __init__(self, fen: str | None = None) -> None:
        self.board: chess.Board = chess.Board(fen) if fen else chess.Board()
        self.history: list[State | None] = [None] * config.game.max_moves_per_game
        self.history_size = 0

    def get_result(self) -> float:
        if not self.board.is_game_over():
            return 0.0
        return {"1-0": 1.0, "0-1": -1.0}.get(self.board.result(), 0.0)

    def add_position(
        self,
        board_tensor: torch.Tensor,
        move_probs: dict[chess.Move, float],
        move_played: chess.Move | None,
    ) -> None:
        if self.history_size < len(self.history):
            self.history[self.history_size] = {
                "board_tensor": board_tensor.clone(),
                "move_probs": move_probs.copy(),
                "move_played": move_played,
                "player": self.board.turn,
            }
            self.history_size += 1

    def make_move(self, move: chess.Move) -> bool:
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    def training_data(self, model: "ChessModel | None" = None) -> list[Position]:
        if not self.board.is_game_over():
            if model is not None:
                try:
                    with torch.no_grad():
                        board_tensor = model.encode_board(self.board).unsqueeze(0)
                        output = model(board_tensor)
                        final_result = float(output.value.item())
                except (RuntimeError, ValueError):
                    final_result = 0.0
            else:
                final_result = 0.0
        else:
            final_result = self.get_result()

        training_data = []
        for i in range(self.history_size):
            position = self.history[i]
            value = final_result if position["player"] == chess.WHITE else -final_result
            training_data.append(
                {
                    "board_tensor": position["board_tensor"],
                    "move_probs": position["move_probs"],
                    "value": value,
                    "move_played": position["move_played"],
                }
            )
        return training_data


class Buffer:
    def __init__(
        self,
        max_size: int = config.training.buffer_size,
        device: str = "cuda",
        move_encoder: MoveEncoder | None = None,
        model: "ChessModel | None" = None,
    ):
        self.max_size = max_size
        self.device = device
        self.move_encoder = move_encoder
        self.model = model
        self.size = 0
        self.pos = 0

        self.boards = torch.zeros(
            (max_size, config.board_encoding_size), dtype=torch.float32, device=device
        )
        self.values = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.policies = torch.zeros(
            (max_size, config.model.move_space_size), dtype=torch.float32, device=device
        )
        self.legal_masks = torch.zeros(
            (max_size, config.model.move_space_size), dtype=torch.float32, device=device
        )

    def add_batch(self, games: list[Game]) -> int:
        if not games:
            return 0

        positions_added = 0
        for game in games:
            training_data = game.training_data(self.model)
            for position in training_data:
                if self._add_position(
                    position["board_tensor"], position["value"], position["move_probs"]
                ):
                    positions_added += 1
        return positions_added

    def _add_position(
        self,
        board_tensor: torch.Tensor,
        value: float,
        move_probs: dict[chess.Move, float],
    ) -> bool:
        if self.move_encoder is None or not move_probs:
            return False

        try:
            if board_tensor.device != torch.device(self.device):
                board_tensor = board_tensor.to(self.device)

            self.boards[self.pos] = board_tensor.flatten()
            self.values[self.pos] = value

            self.policies[self.pos].zero_()
            self.legal_masks[self.pos].zero_()

            for move, prob in move_probs.items():
                idx = self.move_encoder.encode_move(move)
                if 0 <= idx < self.policies.shape[1]:
                    self.legal_masks[self.pos, idx] = 1.0
                    self.policies[self.pos, idx] = prob

            self.pos = (self.pos + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            return True

        except (RuntimeError, ValueError):
            return False

    def sample(self, batch_size: int | None = None) -> dict | None:
        if self.size == 0:
            return None

        batch_size = batch_size or config.optimizer.batch_size
        effective_batch_size = min(batch_size, self.size)

        try:
            indices = torch.randint(
                0, self.size, (effective_batch_size,), device=self.device
            )

            return {
                "board_tensors": self.boards[indices],
                "target_values": self.values[indices],
                "target_policies": self.policies[indices],
                "legal_move_masks": self.legal_masks[indices],
            }
        except RuntimeError:
            return None


class Trainer:
    def __init__(
        self,
        model: ChessModel,
        device: str,
        move_encoder: MoveEncoder,
        learning_rate: float | None = None,
    ) -> None:
        self.model: ChessModel = model
        self.device: str = device
        self.move_encoder: MoveEncoder = move_encoder

        lr = learning_rate or config.optimizer.learning_rate
        self.optimizer: optim.Adam = optim.Adam(model.parameters(), lr=lr)
        self.scheduler: CosineAnnealingLR = CosineAnnealingLR(
            self.optimizer,
            T_max=config.optimizer.t_max,
            eta_min=lr * config.optimizer.eta_min_mult,
        )
        self.value_loss_fn: nn.MSELoss = nn.MSELoss()
        self.games_played: int = 0
        self.training_steps: int = 0
        self.buffer: Buffer = Buffer(
            max_size=config.training.buffer_size,
            device=device,
            move_encoder=move_encoder,
            model=model,
        )
        self.accumulation_step: int = 0

    def train_on_batch(self, batch: dict | None) -> dict[str, float] | None:
        if batch is None:
            return None

        try:
            board_tensors = batch["board_tensors"]
            target_values = batch["target_values"].unsqueeze(1)
            target_policies = batch["target_policies"]
            legal_move_masks = batch.get("legal_move_masks", None)

            self.model.train()

            if self.accumulation_step == 0:
                self.optimizer.zero_grad()

            outputs = self.model(board_tensors)
            predicted_values = outputs.value
            predicted_policies = outputs.policy

            value_loss = self.value_loss_fn(predicted_values, target_values)

            if legal_move_masks is not None:
                masked_predictions = predicted_policies * legal_move_masks
                masked_targets = target_policies * legal_move_masks
                masked_predictions = masked_predictions / (
                    masked_predictions.sum(dim=1, keepdim=True)
                    + config.system.policy_epsilon
                )

                policy_loss = f.kl_div(
                    torch.log(masked_predictions + config.system.policy_epsilon),
                    masked_targets,
                    reduction="batchmean",
                )
            else:
                policy_loss = f.kl_div(
                    torch.log(predicted_policies + config.system.policy_epsilon),
                    target_policies,
                    reduction="batchmean",
                )

            total_loss = (
                value_loss + policy_loss
            ) / config.optimizer.gradient_accumulation_steps

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                return None

            total_loss.backward()

            self.accumulation_step += 1

            if self.accumulation_step >= config.optimizer.gradient_accumulation_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=config.optimizer.clip_norm
                )
                self.optimizer.step()
                self.accumulation_step = 0
            else:
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm**0.5

            self.training_steps += 1

            grad_norm_val = (
                float(grad_norm.item())
                if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm)
            )

            actual_total_loss = (
                total_loss.item() * config.optimizer.gradient_accumulation_steps
            )

            loss_dict = {
                "total_loss": float(actual_total_loss),
                "value_loss": float(value_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "grad_norm": grad_norm_val,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "accumulation_step": self.accumulation_step,
            }

            return loss_dict

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            return None
        except ValueError:
            return None

    def _should_resign(
        self, board: chess.Board, resignation_disabled: bool = False
    ) -> bool:
        if resignation_disabled or board.is_game_over():
            return False

        try:
            with torch.no_grad():
                board_tensor = self.model.encode_board(board).unsqueeze(0)
                output = self.model(board_tensor)
                position_value = float(output.value.item())

                if not board.turn:
                    position_value = -position_value

                return position_value < config.game.resignation_threshold

        except (RuntimeError, ValueError):
            return False

    def _process_move(
        self, game: Game, game_idx: int, move_probs: dict, move_counts: np.ndarray
    ) -> bool:
        if not move_probs:
            legal_moves = list(game.board.legal_moves)
            move_probs = uniform_probs(legal_moves)
            if not move_probs:
                return False

        try:
            board_tensor = self.model.encode_board(game.board)
            game.add_position(board_tensor, move_probs, None)

            rng = np.random.default_rng()
            if rng.random() < config.game.exploration_epsilon:
                legal_moves = list(game.board.legal_moves)
                move = rng.choice(len(legal_moves))
                move = legal_moves[move]
            else:
                temperature = get_temp(move_counts[game_idx])
                move = sample_move(move_probs, temperature)

            if game.history_size > 0:
                game.history[game.history_size - 1]["move_played"] = move

            if game.make_move(move):
                move_counts[game_idx] += 1
                return True
            else:
                return False
        except (RuntimeError, ValueError):
            return False

    def _collect_game_stats(
        self, games: list[Game], total_moves: int, mcts_searches: int
    ) -> dict:
        completed_games = len([g for g in games if g.board.is_game_over()])
        timeout_games = len([g for g in games if not g.board.is_game_over()])
        avg_game_length = total_moves / len(games) if games else 0
        zero_move_games = len([g for g in games if g.history_size == 0])

        game_results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        for game in games:
            if game.board.is_game_over():
                result = game.board.result()
                game_results[result] = game_results.get(result, 0) + 1

        completion_rate = (completed_games / len(games) * 100) if games else 0
        win_rate = (game_results.get("1-0", 0) / len(games) * 100) if games else 0

        return {
            "completed": completed_games,
            "timeout": timeout_games,
            "total": len(games),
            "avg_length": avg_game_length,
            "completion_rate": completion_rate,
            "wins": game_results.get("1-0", 0),
            "losses": game_results.get("0-1", 0),
            "draws": game_results.get("1/2-1/2", 0),
            "win_rate": win_rate,
            "mcts_searches": mcts_searches,
            "zero_move_games": zero_move_games,
        }

    def play_games(self, num_games: int | None = None) -> tuple[list[Game], dict]:
        num_games = num_games or config.game.games_per_iteration

        try:
            mcts = MCTS(self.model, self.move_encoder, self.device)
            games: list[Game] = [Game() for _ in range(num_games)]

            move_counts = np.zeros(num_games, dtype=np.int32)
            completed = np.zeros(num_games, dtype=bool)
            resignation_disabled = (
                np.random.random(num_games) < config.game.resignation_disabled_rate
            )

            total_moves = 0
            mcts_searches = 0

            while not completed.all():
                active_mask = ~completed
                active_indices = np.where(active_mask)[0]

                if len(active_indices) == 0:
                    break

                for i in active_indices:
                    game_over = games[i].board.is_game_over()
                    max_moves_reached = move_counts[i] >= config.game.max_moves_per_game
                    should_resign = (
                        not resignation_disabled[i]
                        and not game_over
                        and self._should_resign(games[i].board)
                    )

                    if game_over or max_moves_reached or should_resign:
                        completed[i] = True

                active_mask = ~completed
                active_indices = np.where(active_mask)[0]

                if len(active_indices) == 0:
                    break

                active_boards = [games[i].board for i in active_indices]

                try:
                    move_probs_batch = mcts.search_batch(active_boards)
                    mcts_searches += 1
                except RuntimeError:
                    break

                for idx, game_idx in enumerate(active_indices):
                    if self._process_move(
                        games[game_idx], game_idx, move_probs_batch[idx], move_counts
                    ):
                        total_moves += 1
                    else:
                        completed[game_idx] = True

                if mcts_searches % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            game_stats = self._collect_game_stats(games, total_moves, mcts_searches)
            self.games_played += len(games)
            return games, game_stats

        except RuntimeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            empty_stats = {
                "completed": 0,
                "timeout": 0,
                "total": 0,
                "avg_length": 0,
                "completion_rate": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "win_rate": 0,
                "mcts_searches": 0,
                "zero_move_games": 0,
            }
            return [], empty_stats

    def train_step(
        self, num_games: int | None = None, iteration: int | None = None
    ) -> IterationMetrics | None:
        iteration_start = time.perf_counter()
        games, game_stats = self.play_games(num_games)

        if not games:
            return None

        self.buffer.add_batch(games)
        batch = self.buffer.sample()
        if batch is None:
            return None

        training_result = self.train_on_batch(batch)
        if (
            training_result is not None
            and training_result.get("accumulation_step", 0) == 0
        ):
            self.scheduler.step()

        metrics = IterationMetrics(
            training_loss=training_result["total_loss"] if training_result else 0.0,
            value_loss=training_result["value_loss"] if training_result else 0.0,
            policy_loss=training_result["policy_loss"] if training_result else 0.0,
            grad_norm=training_result["grad_norm"] if training_result else 0.0,
            learning_rate=training_result["learning_rate"] if training_result else 0.0,
            games_completed=game_stats["completed"],
            games_timeout=game_stats["timeout"],
            games_total=game_stats["total"],
            avg_game_length=game_stats["avg_length"],
            wins=game_stats["wins"],
            losses=game_stats["losses"],
            draws=game_stats["draws"],
            buffer_size=self.buffer.size,
            games_played_total=self.games_played,
            training_steps_total=self.training_steps,
            iteration_time=time.perf_counter() - iteration_start,
            gpu_memory_gb=torch.cuda.memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0,
            mcts_searches=game_stats["mcts_searches"],
            zero_move_games=game_stats["zero_move_games"],
        )

        return metrics

    def save_model(self, filepath: str) -> None:
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "games_played": self.games_played,
                "training_steps": self.training_steps,
                "move_encoder_num_moves": self.move_encoder.num_moves,
            }
            torch.save(checkpoint, filepath)
        except OSError:
            pass


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path("./checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer: Trainer | None = None
    try:
        move_encoder = MoveEncoder()
        model = ChessModel(device=device)
        trainer = Trainer(model, device, move_encoder)

        print("Training Started")

        start_time = time.perf_counter()
        for iteration in range(1, config.training.iterations + 1):
            metrics = trainer.train_step(iteration=iteration)

            if metrics:
                metrics.log_summary(iteration)

            if iteration % config.training.save_every == 0:
                save_path = save_dir / f"model_iteration_{iteration}.pt"
                trainer.save_model(str(save_path))

        total_training_time = time.perf_counter() - start_time
        final_path = save_dir / "model_final.pt"
        trainer.save_model(str(final_path))

        print("\nTraining Completed Successfully!")
        print(
            f"Total time: {total_training_time / 3600:.1f} hours | Games played: {trainer.games_played:,}"
        )

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
