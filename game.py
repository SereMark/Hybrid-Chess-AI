import logging
from typing import Any, Union

import chess
import numpy as np
import torch
from config import get_config
from move_encoder import MoveEncoder
from move_utils import (
    evaluate_game_result,
)

logger = logging.getLogger(__name__)


class Game:
    def __init__(self, fen: str | None = None) -> None:
        self.board: chess.Board = chess.Board(fen) if fen else chess.Board()
        self.history: list[dict[str, Any]] = []
        self.moves_played: list[chess.Move] = []
        self.game_over: bool = False
        self.result: str | None = None

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def get_result(self) -> float | None:
        if not self.board.is_game_over():
            return None
        return evaluate_game_result(self.board, self.board.turn == chess.WHITE)

    def add_position(
        self,
        board_tensor: torch.Tensor,
        move_probs: Union[dict[chess.Move, float], torch.Tensor],
        move_played: chess.Move | None,
    ) -> None:
        self.history.append(
            {
                "board_tensor": board_tensor.clone(),
                "move_probs": (
                    move_probs.copy() if isinstance(move_probs, dict) else move_probs
                ),
                "move_played": move_played,
                "player": self.board.turn,
            }
        )

    def make_move(self, move: chess.Move) -> bool:
        if move in self.board.legal_moves:
            self.moves_played.append(move)
            self.board.push(move)
            return True
        else:
            return False

    def get_training_data(self) -> list[dict[str, Any]]:
        if not self.board.is_game_over():
            timeout_values_config = get_config("game", "timeout_game_values")
            timeout_values = (
                timeout_values_config
                if isinstance(timeout_values_config, list)
                else [-0.1, 0.0, 0.1]
            )
            final_result = np.random.default_rng().choice(timeout_values)
        else:
            final_result = self.get_result()
        training_data = []
        for _i, position in enumerate(self.history):
            if position["player"] == chess.WHITE:
                value = final_result if final_result is not None else 0.0
            else:
                value = -final_result if final_result is not None else 0.0
            training_data.append(
                {
                    "board_tensor": position["board_tensor"],
                    "move_probs": position["move_probs"],
                    "value": value,
                    "move_played": position["move_played"],
                }
            )
        return training_data

    def get_fen(self) -> str:
        return self.board.fen()

    def copy(self) -> "Game":
        new_game = Game(self.board.fen())
        new_game.history = [pos.copy() for pos in self.history]
        new_game.moves_played = self.moves_played.copy()
        new_game.game_over = self.game_over
        new_game.result = self.result
        return new_game


def create_training_batch(
    games: list[Game], device: str
) -> dict[str, torch.Tensor] | None:
    all_data = []
    failed_extractions = 0
    for _i, game in enumerate(games):
        try:
            training_data = game.get_training_data()
            all_data.extend(training_data)
        except Exception:
            failed_extractions += 1
            continue
    if not all_data:
        logger.error("Batch creation failed: no valid positions")
        return None
    try:
        board_tensors = []
        target_values = []
        target_policies = []
        move_encoder = MoveEncoder()
        policy_nonzero_count = 0
        for _i, data in enumerate(all_data):
            if (
                "board_tensor" not in data
                or "value" not in data
                or "move_probs" not in data
            ):
                continue

            board_tensor = data["board_tensor"]
            if not isinstance(board_tensor, torch.Tensor):
                continue

            board_tensors.append(board_tensor)
            value = data["value"]
            if not isinstance(value, int | float) or not (-2.0 <= value <= 2.0):
                value = max(-1.0, min(1.0, float(value)))
            target_values.append(value)

            move_space_size_config = get_config("model", "move_space_size")
            move_space_size = (
                int(move_space_size_config)
                if isinstance(move_space_size_config, int | float | str)
                else 4096
            )
            policy_vector = torch.zeros(move_space_size, dtype=torch.float32)
            if isinstance(data["move_probs"], dict):
                valid_moves = 0
                for move, prob in data["move_probs"].items():
                    if not isinstance(prob, int | float) or prob < 0:
                        continue
                    idx = move_encoder.encode_move(move)
                    if 0 <= idx < move_space_size:
                        policy_vector[idx] = min(1.0, max(0.0, float(prob)))
                        valid_moves += 1
                if valid_moves > 0:
                    policy_nonzero_count += 1
            target_policies.append(policy_vector)
        if len(board_tensors) == 0:
            logger.error("No valid training data after filtering")
            return None

        batch = {
            "board_tensors": torch.stack(board_tensors).to(device),
            "target_values": torch.tensor(target_values, dtype=torch.float32).to(
                device
            ),
            "target_policies": torch.stack(target_policies).to(device),
        }

        if torch.any(torch.isnan(batch["board_tensors"])) or torch.any(
            torch.isinf(batch["board_tensors"])
        ):
            logger.error("Invalid board tensors detected")
            return None
        if torch.any(torch.isnan(batch["target_values"])) or torch.any(
            torch.isinf(batch["target_values"])
        ):
            logger.error("Invalid target values detected")
            return None
        if torch.any(torch.isnan(batch["target_policies"])) or torch.any(
            torch.isinf(batch["target_policies"])
        ):
            logger.error("Invalid target policies detected")
            return None

        return batch
    except Exception as e:
        logger.error(
            f"Training batch creation failed with {len(all_data)} positions: {e}"
        )
        raise RuntimeError(f"Training batch creation failed: {e}") from e
