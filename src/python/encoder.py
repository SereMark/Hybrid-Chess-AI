from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

BOARD_SIZE = 8
NSQUARES = BOARD_SIZE * BOARD_SIZE
PLANES_PER_POSITION = 14
HISTORY_LENGTH = 8
INPUT_PLANES = HISTORY_LENGTH * PLANES_PER_POSITION + 7


PIECE_NAMES = ("pawn", "knight", "bishop", "rook", "queen", "king")


@dataclass(slots=True)
class PositionState:
    boards: Dict[str, Tuple[np.ndarray, np.ndarray]]
    turn: int
    castling: int
    ep_square: int | None
    halfmove: int
    fullmove: int


def encode_position_into(pos: PositionState, out: np.ndarray) -> None:
    out[...] = 0.0
    area = NSQUARES
    for piece_idx, name in enumerate(PIECE_NAMES):
        white_bb, black_bb = pos.boards.get(name, (np.zeros(NSQUARES, dtype=bool), np.zeros(NSQUARES, dtype=bool)))
        plane_w = piece_idx * 2
        plane_b = plane_w + 1
        out[plane_w, white_bb] = 1.0
        out[plane_b, black_bb] = 1.0

    reps = pos.halfmove // 50
    if reps >= 2:
        out[12, :] = 1.0
    if reps >= 3:
        out[13, :] = 1.0

    base_history = HISTORY_LENGTH * PLANES_PER_POSITION
    out[base_history, :] = 1.0 if pos.turn == 0 else 0.0
    out[base_history + 1, :] = min(1.0, pos.fullmove / 100.0)

    castling_planes = base_history + 2
    for idx in range(4):
        mask = 1 << idx
        out[castling_planes + idx, :] = 1.0 if (pos.castling & mask) else 0.0

    out[castling_planes + 4, :] = min(1.0, pos.halfmove / 100.0)


def encode_position_with_history(history: Sequence[PositionState], out: np.ndarray) -> None:
    out[...] = 0.0
    avail = min(HISTORY_LENGTH, len(history))
    for t in range(avail):
        pos = history[-1 - t]
        base = t * PLANES_PER_POSITION
        for piece_idx, name in enumerate(PIECE_NAMES):
            white_bb, black_bb = pos.boards.get(name, (np.zeros(NSQUARES, dtype=bool), np.zeros(NSQUARES, dtype=bool)))
            plane_w = base + piece_idx * 2
            plane_b = plane_w + 1
            out[plane_w, white_bb] = 1.0
            out[plane_b, black_bb] = 1.0

    if history:
        cur = history[-1]
        base_history = HISTORY_LENGTH * PLANES_PER_POSITION
        out[base_history, :] = 1.0 if cur.turn == 0 else 0.0
        out[base_history + 1, :] = min(1.0, cur.fullmove / 100.0)
        castling_planes = base_history + 2
        for idx in range(4):
            mask = 1 << idx
            out[castling_planes + idx, :] = 1.0 if (cur.castling & mask) else 0.0
        out[castling_planes + 4, :] = min(1.0, cur.halfmove / 100.0)


__all__ = [
    "BOARD_SIZE",
    "NSQUARES",
    "PLANES_PER_POSITION",
    "HISTORY_LENGTH",
    "INPUT_PLANES",
    "encode_position_into",
    "encode_position_with_history",
    "PositionState",
]