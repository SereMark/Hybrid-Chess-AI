import chess
import numpy as np
from config import get_config


def get_temperature_for_move(move_count: int) -> float:
    high_temp = float(get_config("game", "high_temperature") or 1.0)
    low_temp = float(get_config("game", "low_temperature") or 0.1)
    exploration_moves = int(get_config("game", "exploration_temperature_moves") or 10)

    if move_count < exploration_moves:
        progress = move_count / exploration_moves
        return high_temp * (1 - progress) + low_temp * progress
    else:
        return low_temp


def select_move_with_temperature(
    move_probs: dict[chess.Move, float],
    temperature: float = 1.0,
    board: chess.Board | None = None,
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

    selected_move = moves[np.random.default_rng().choice(len(moves), p=probs)]

    if board is not None and selected_move not in board.legal_moves:
        legal_moves = list(board.legal_moves)
        if legal_moves:
            selected_move = np.random.default_rng().choice(legal_moves)
        else:
            raise ValueError("No legal moves available")

    return selected_move


def evaluate_game_result(
    board: chess.Board, from_white_perspective: bool = True
) -> float:
    if not board.is_game_over():
        return 0.0

    result = board.result()

    if result == "1-0":
        return 1.0 if from_white_perspective else -1.0
    elif result == "0-1":
        return -1.0 if from_white_perspective else 1.0
    else:
        return 0.0
