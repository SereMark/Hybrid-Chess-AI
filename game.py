from typing import TYPE_CHECKING, TypedDict

import chess
import numpy as np
import torch
from config import config

if TYPE_CHECKING:
    from model import ChessModel


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


def get_temp(move_count: int) -> float:
    if move_count < config.game.temp_moves:
        progress = move_count / config.game.temp_moves
        return (
            config.game.high_temperature * (1 - progress)
            + config.game.low_temperature * progress
        )
    return config.game.low_temperature


def uniform_probs(legal_moves: list[chess.Move]) -> dict[chess.Move, float]:
    if not legal_moves:
        return {}
    prob = 1.0 / len(legal_moves)
    return dict.fromkeys(legal_moves, prob)


def sample_move(
    move_probs: dict[chess.Move, float], temperature: float = 1.0
) -> chess.Move:
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

    rng = np.random.default_rng()
    return moves[rng.choice(len(moves), p=probs)]


class Game:
    def __init__(self, fen: str | None = None) -> None:
        self.board: chess.Board = chess.Board(fen) if fen else chess.Board()
        self.history: list[State] = []

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
    ) -> None:
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

    def training_data(self, model: "ChessModel | None" = None) -> list[Position]:
        if not self.board.is_game_over():
            rng = np.random.default_rng()
            outcome_prob = rng.random()
            if outcome_prob < 0.4:
                final_result = 1.0
            elif outcome_prob < 0.8:
                final_result = -1.0
            else:
                final_result = 0.0
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
