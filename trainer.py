import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from config import config
from game import Game, get_temp, sample_move, uniform_probs
from mcts import MCTS
from model import ChessModel, MoveEncoder
from torch import nn, optim
from torch.nn import functional
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import performance_monitor

if TYPE_CHECKING:
    from utils import ConsoleMetricsLogger


class Buffer:
    def __init__(
        self,
        max_size: int = config.training.buffer_size,
        device: str = "cuda",
        move_encoder: MoveEncoder | None = None,
        model: "ChessModel | None" = None,
    ):
        self.logger: logging.Logger = logging.getLogger("chess_ai.training.buffer")
        self.max_size: int = max_size
        self.device: str = device
        self.move_encoder: MoveEncoder | None = move_encoder
        self.model: ChessModel | None = model
        self.size: int = 0
        self.pos: int = 0

        self.boards: torch.Tensor = torch.zeros(
            (max_size, config.board_encoding_size), dtype=torch.float32, device=device
        )
        self.values: torch.Tensor = torch.zeros(
            max_size, dtype=torch.float32, device=device
        )
        self.policies: torch.Tensor = torch.zeros(
            (max_size, config.model.move_space_size), dtype=torch.float32, device=device
        )
        self.legal_masks: torch.Tensor = torch.zeros(
            (max_size, config.model.move_space_size), dtype=torch.float32, device=device
        )

    @performance_monitor
    def add_batch(self, games: list[Game]) -> int:
        if not games:
            self.logger.warning("add_batch called with empty games list")
            return 0

        positions_added = 0
        invalid_moves = 0

        for game in games:
            training_data = game.training_data(self.model)
            for position in training_data:
                board_tensor = position["board_tensor"]
                value = position["value"]
                move_probs = position["move_probs"]

                if str(board_tensor.device) != self.device:
                    board_tensor = board_tensor.to(self.device)

                try:
                    self.boards[self.pos] = board_tensor.flatten()
                    self.values[self.pos] = value

                    self.policies[self.pos].zero_()
                    self.legal_masks[self.pos].zero_()

                    for move, prob in move_probs.items():
                        try:
                            if self.move_encoder is None:
                                continue
                            idx = self.move_encoder.encode_move(move)
                            if 0 <= idx < self.policies.shape[1]:
                                self.legal_masks[self.pos, idx] = 1.0
                                self.policies[self.pos, idx] = prob
                            else:
                                invalid_moves += 1
                        except ValueError:
                            invalid_moves += 1

                    self.pos = (self.pos + 1) % self.max_size
                    self.size = min(self.size + 1, self.max_size)
                    positions_added += 1

                except (RuntimeError, ValueError):
                    continue

        if positions_added > 0:
            self.logger.info(
                f"Buffer update: +{positions_added} positions, size={self.size}/{self.max_size}"
            )

        return positions_added

    def sample(self, batch_size: int | None = None) -> dict | None:
        if batch_size is None:
            batch_size = config.optimizer.batch_size
        if self.size == 0:
            self.logger.warning("Attempted to sample from empty buffer")
            return None

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
        except (RuntimeError, ValueError) as e:
            self.logger.error(f"Error sampling from buffer: {e}")
            return None


class Trainer:
    def __init__(
        self,
        model: ChessModel,
        device: str,
        move_encoder: MoveEncoder,
        training_logger: "ConsoleMetricsLogger | None" = None,
        learning_rate: float | None = None,
    ) -> None:
        self.logger: logging.Logger = logging.getLogger("chess_ai.training")
        self.model: ChessModel = model
        self.device: str = device
        self.move_encoder: MoveEncoder = move_encoder
        self.training_logger: ConsoleMetricsLogger | None = training_logger

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

        self.losses: list[float] = []
        self.val_losses: list[float] = []
        self.pol_losses: list[float] = []

        self.logger.info(
            f"Trainer initialized: lr={lr}, device={device}, "
            f"model_params={sum(p.numel() for p in model.parameters()):,}"
        )

    def analyze_predictions(
        self,
        predicted_policies: torch.Tensor,
        predicted_values: torch.Tensor,
        target_values: torch.Tensor,
        legal_move_masks: torch.Tensor | None = None,
    ) -> dict[str, float]:
        with torch.no_grad():
            policies_np = predicted_policies.detach().cpu().numpy()
            value_predictions = predicted_values.detach().cpu().numpy().flatten()
            value_targets = target_values.detach().cpu().numpy().flatten()

            policy_sums = (
                np.sum(policies_np, axis=1, keepdims=True)
                + config.system.policy_epsilon
            )
            policies_normalized = policies_np / policy_sums
            policy_entropies = -np.sum(
                policies_normalized * np.log(policies_normalized + 1e-10), axis=1
            )

            policies_sorted = np.sort(policies_np, axis=1)[:, ::-1]
            top_k_probabilities = np.sum(policies_sorted[:, :3], axis=1)

            if len(value_predictions) > 1:
                try:
                    corr_matrix = np.corrcoef(value_predictions, value_targets)
                    value_correlation = (
                        float(corr_matrix[0, 1])
                        if not np.isnan(corr_matrix).any()
                        else 0.0
                    )
                except (ValueError, np.linalg.LinAlgError):
                    value_correlation = 0.0
            else:
                value_correlation = 0.0

            return {
                "policy_entropy_avg": float(np.mean(policy_entropies)),
                "top3_probability_mass_avg": float(np.mean(top_k_probabilities)),
                "value_correlation": float(value_correlation),
            }

    @performance_monitor
    def train_on_batch(self, batch: dict | None) -> dict[str, float] | None:
        if batch is None:
            self.logger.warning("train_on_batch called with None batch")
            return None

        try:
            board_tensors = batch["board_tensors"]
            target_values = batch["target_values"].unsqueeze(1)
            target_policies = batch["target_policies"]
            legal_move_masks = batch.get("legal_move_masks", None)

            self.model.train()
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

                if torch.any(torch.isnan(masked_predictions)) or torch.any(
                    torch.isnan(masked_targets)
                ):
                    self.logger.error("NaN detected in predictions")
                    return None

                policy_loss = functional.kl_div(
                    torch.log(masked_predictions + config.system.policy_epsilon),
                    masked_targets,
                    reduction="batchmean",
                )
            else:
                policy_loss = functional.kl_div(
                    torch.log(predicted_policies + config.system.policy_epsilon),
                    target_policies,
                    reduction="batchmean",
                )

            total_loss = value_loss + policy_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.logger.error(f"Invalid loss detected: {total_loss.item()}")
                return None

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=config.optimizer.clip_norm
            )
            self.optimizer.step()

            self.training_steps += 1

            prediction_analysis = self.analyze_predictions(
                predicted_policies, predicted_values, target_values, legal_move_masks
            )

            loss_dict = {
                "total_loss": float(total_loss.item()),
                "value_loss": float(value_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "grad_norm": float(grad_norm.item())
                if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }

            loss_dict.update(prediction_analysis)

            self.losses.append(loss_dict["total_loss"])
            self.val_losses.append(loss_dict["value_loss"])
            self.pol_losses.append(loss_dict["policy_loss"])

            return loss_dict

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.error(f"GPU out of memory during training: {e}")
                torch.cuda.empty_cache()
            else:
                self.logger.error(f"Runtime error during training: {e}")
            return None
        except ValueError as e:
            self.logger.error(f"Unexpected error during training: {e}", exc_info=True)
            return None

    def _get_active(
        self, games: list[Game], move_counts: list[int], completed: set[int]
    ) -> tuple[list[int], list]:
        active_indices = []
        active_boards = []
        for i, game in enumerate(games):
            if i not in completed:
                if (
                    game.board.is_game_over()
                    or move_counts[i] >= config.game.max_moves_per_game
                ):
                    completed.add(i)
                    if move_counts[i] >= config.game.max_moves_per_game:
                        self.logger.debug(
                            f"Game {i} reached max moves ({config.game.max_moves_per_game})"
                        )
                else:
                    active_indices.append(i)
                    active_boards.append(game.board)
        return active_indices, active_boards

    def _process_move(
        self, game: Game, game_idx: int, move_probs: dict, move_counts: list[int]
    ) -> bool:
        if not move_probs:
            self.logger.debug(
                f"No move probabilities for game {game_idx}, using uniform"
            )
            legal_moves = list(game.board.legal_moves)
            move_probs = uniform_probs(legal_moves)
            if not move_probs:
                self.logger.warning(f"Game {game_idx} has no legal moves")
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

            if game.history:
                game.history[-1]["move_played"] = move

            if game.make_move(move):
                move_counts[game_idx] += 1
                return True
            else:
                self.logger.debug(f"Invalid move {move} in game {game_idx}")
                return False
        except (RuntimeError, ValueError) as e:
            self.logger.error(f"Error processing game {game_idx}: {e}")
            return False

    def _collect_stats(
        self, games: list[Game], total_moves: int, mcts_searches: int
    ) -> None:
        completed_games = len([g for g in games if g.board.is_game_over()])
        timeout_games = len([g for g in games if not g.board.is_game_over()])
        avg_game_length = total_moves / len(games) if games else 0
        game_results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

        for game in games:
            if game.board.is_game_over():
                result = game.board.result()
                game_results[result] = game_results.get(result, 0) + 1

        self.logger.info(
            f"Generated {len(games)} games: {completed_games} finished, {timeout_games} timeouts, "
            f"avg_length={avg_game_length:.1f}, {mcts_searches} MCTS searches, "
            f"results={game_results}"
        )

    @performance_monitor
    def play_games(self, num_games: int | None = None) -> list[Game]:
        num_games = num_games or config.game.games_per_iteration
        self.logger.info(f"Generating {num_games} games")

        try:
            mcts = MCTS(
                self.model, self.move_encoder, self.device, self.training_logger
            )
            games: list[Game] = [Game() for _ in range(num_games)]
            move_counts = [0] * num_games
            completed = set()
            total_moves = 0
            mcts_searches = 0

            while len(completed) < num_games:
                active_indices, active_boards = self._get_active(
                    games, move_counts, completed
                )

                if not active_indices:
                    break

                try:
                    move_probs_batch = mcts.search_batch(active_boards)
                    mcts_searches += 1
                except RuntimeError as e:
                    self.logger.error(f"MCTS search failed: {e}")
                    break

                for idx, game_idx in enumerate(active_indices):
                    if self._process_move(
                        games[game_idx], game_idx, move_probs_batch[idx], move_counts
                    ):
                        total_moves += 1
                    else:
                        completed.add(game_idx)

            self._collect_stats(games, total_moves, mcts_searches)
            self.games_played += len(games)
            return games

        except RuntimeError as e:
            self.logger.error(f"Critical error in game generation: {e}", exc_info=True)
            return []

    @performance_monitor
    def train_step(
        self, num_games: int | None = None, iteration: int | None = None
    ) -> dict[str, float] | None:
        iteration_start = time.perf_counter()

        games = self.play_games(num_games)
        if not games:
            self.logger.error("No games generated, skipping iteration")
            return None

        completed_games = len([g for g in games if g.board.is_game_over()])
        timeout_games = len([g for g in games if not g.board.is_game_over()])
        total_moves = sum(len(g.history) for g in games)
        avg_game_length = total_moves / len(games) if games else 0
        game_results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

        for game in games:
            if game.board.is_game_over():
                result = game.board.result()
                game_results[result] = game_results.get(result, 0) + 1

        if self.training_logger and iteration is not None:
            self.training_logger.log_games(
                completed_games, timeout_games, avg_game_length, game_results, iteration
            )

        positions_added = self.buffer.add_batch(games)
        if positions_added == 0:
            self.logger.warning("No positions added to buffer")

        batch = self.buffer.sample()
        if batch is None:
            self.logger.warning("Could not sample batch for training")
            return None

        result = self.train_on_batch(batch)

        if result is not None:
            self.scheduler.step()

            if len(self.losses) >= config.system.loss_trend_window:
                recent_losses = self.losses[-config.system.loss_trend_window :]
                loss_trend = (
                    (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
                    if len(recent_losses) > 0
                    else 0.0
                )
                result["loss_trend"] = loss_trend

            result["buffer_size"] = self.buffer.size
            result["games_played"] = self.games_played
            result["training_steps"] = self.training_steps

            result["games_completed"] = completed_games
            result["games_total"] = len(games)
            result["games_timeout"] = timeout_games
            result["avg_game_length"] = avg_game_length

            iteration_time = time.perf_counter() - iteration_start
            result["iteration_time"] = iteration_time

        return result

    def save_model(self, filepath: Path | str) -> None:
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "games_played": self.games_played,
                "training_steps": self.training_steps,
                "loss_history": self.losses[-config.system.loss_history_size :],
                "value_loss_history": self.val_losses[
                    -config.system.loss_history_size :
                ],
                "policy_loss_history": self.pol_losses[
                    -config.system.loss_history_size :
                ],
            }
            torch.save(checkpoint, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except OSError as e:
            self.logger.error(f"Failed to save model to {filepath}: {e}")
