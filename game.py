import chess
import numpy as np
import torch
from config import config


def get_temp(move_count: int) -> float:
    if move_count < config.EXPLORATION_TEMPERATURE_MOVES:
        progress = move_count / config.EXPLORATION_TEMPERATURE_MOVES
        return (
            config.HIGH_TEMPERATURE * (1 - progress) + config.LOW_TEMPERATURE * progress
        )
    return config.LOW_TEMPERATURE


def sample_move(move_probs: dict[chess.Move, float], temperature: float = 1.0):
    if not move_probs:
        raise ValueError("No moves available for selection")

    moves = list(move_probs.keys())
    probs = np.array(list(move_probs.values()), dtype=np.float64)

    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)

    probs_sum = np.sum(probs)
    if probs_sum > 0:
        probs = probs / probs_sum
    else:
        probs = np.ones(len(moves)) / len(moves)

    return moves[np.random.default_rng().choice(len(moves), p=probs)]


class Game:
    def __init__(self, fen: str | None = None):
        self.board = chess.Board(fen) if fen else chess.Board()
        self.history: list[dict] = []

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def get_result(self) -> float:
        if not self.board.is_game_over():
            return 0.0

        result = self.board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        return 0.0

    def add_position(
        self,
        board_tensor: torch.Tensor,
        move_probs: dict[chess.Move, float],
        move_played: chess.Move | None,
    ):
        self.history.append(
            {
                "board_tensor": board_tensor.clone(),
                "move_probs": move_probs.copy(),
                "move_played": move_played,
                "player": self.board.turn,
            }
        )

    def make_move(self, move: chess.Move) -> bool:
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    def get_training_data(self):
        if not self.board.is_game_over():
            timeout_values = [-0.1, 0.0, 0.1]
            final_result = float(
                np.random.default_rng().choice(timeout_values)
            )
        else:
            final_result = self.get_result()

        training_data = []
        for position in self.history:
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

