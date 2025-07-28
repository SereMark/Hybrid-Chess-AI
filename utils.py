import chess
import numpy as np
import torch
from config import RESIGN_THRESHOLD


def should_resign_position(board: chess.Board, model) -> bool:
    if board.is_game_over() or model is None:
        return False

    with torch.no_grad():
        tensor = model.encode_board(board).unsqueeze(0)
        value = model(tensor).value.squeeze().item()
        return (
            (value < RESIGN_THRESHOLD)
            if board.turn
            else (value > -RESIGN_THRESHOLD)
        )


def sample_move_from_probabilities(probs: dict[chess.Move, float], temperature: float) -> chess.Move | None:
    moves = list(probs.keys())
    if not moves:
        return None

    values = list(probs.values())

    if temperature != 1.0:
        values = [max(v, 1e-10) ** (1.0 / temperature) for v in values]

    values_sum = sum(values)
    if values_sum == 0 or np.isnan(values_sum):
        values = [1.0 / len(values)] * len(values)
    else:
        values = [v / values_sum for v in values]

    r = np.random.random()
    cumsum = 0.0
    for i, p in enumerate(values):
        cumsum += p
        if r <= cumsum:
            return moves[i]
    return moves[-1]
