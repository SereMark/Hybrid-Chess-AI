"""Lightweight numpy-based position encoding utilities."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
import chesscore as ccore

BOARD_SIZE = 8
NSQUARES = BOARD_SIZE * BOARD_SIZE
PLANES_PER_POSITION = 14
HISTORY_LENGTH = 8
INPUT_PLANES = HISTORY_LENGTH * PLANES_PER_POSITION + 7
POLICY_PLANES = 73
POLICY_SIZE = POLICY_PLANES * NSQUARES

PIECE_NAMES = ("pawn", "knight", "bishop", "rook", "queen", "king")

__all__ = [
    "PositionState",
    "BOARD_SIZE",
    "NSQUARES",
    "PLANES_PER_POSITION",
    "HISTORY_LENGTH",
    "INPUT_PLANES",
    "encode_position",
    "encode_batch",
    "POLICY_SIZE",
    "encode_move_index",
    "encode_move_indices_batch",
]


@dataclass(slots=True)
class PositionState:
    boards: Dict[str, Tuple[np.ndarray, np.ndarray]]
    turn: int
    castling: int
    ep_square: int | None
    halfmove: int
    fullmove: int


_EMPTY_MASK = np.zeros(NSQUARES, dtype=bool)
_DIR8: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
_KNIGHT_OFFSETS: tuple[tuple[int, int], ...] = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)
_PROMO_PIECE_ORDER = {
    int(ccore.Piece.KNIGHT): 0,
    int(ccore.Piece.ROOK): 1,
    int(ccore.Piece.BISHOP): 2,
}
_PROMO_FILE_OFFSETS = {0: 0, -1: 1, 1: 2}


def encode_position(position: Any, history: Sequence[Any] | None = None) -> np.ndarray:
    """Encode a position (optionally with history) into network input planes."""
    states: list[PositionState] = []
    if history:
        states.extend(_ensure_state(item) for item in history)
    states.append(_ensure_state(position))

    encoded = np.zeros((INPUT_PLANES, NSQUARES), dtype=np.float32)
    avail = min(HISTORY_LENGTH, len(states))

    for offset in range(avail):
        state = states[-1 - offset]
        base = offset * PLANES_PER_POSITION
        _write_piece_planes(state, base, encoded)
        if offset == 0:
            _write_auxiliary_planes(state, encoded)

    return encoded.reshape(INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)


def encode_batch(positions: Sequence[Any]) -> np.ndarray:
    """Vectorised helper encoding a batch of positions."""
    if not positions:
        return np.zeros((0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    encoded = [encode_position(pos) for pos in positions]
    return np.stack(encoded, axis=0)


def _ensure_state(position: Any) -> PositionState:
    return position if isinstance(position, PositionState) else _state_from_position(position)


def _write_piece_planes(state: PositionState, base: int, out_flat: np.ndarray) -> None:
    for piece_idx, name in enumerate(PIECE_NAMES):
        white_mask, black_mask = state.boards.get(name, (_EMPTY_MASK, _EMPTY_MASK))
        plane_white = base + piece_idx * 2
        plane_black = plane_white + 1
        if white_mask.any():
            out_flat[plane_white, white_mask] = 1.0
        if black_mask.any():
            out_flat[plane_black, black_mask] = 1.0


def _write_auxiliary_planes(state: PositionState, out_flat: np.ndarray) -> None:
    base_history = HISTORY_LENGTH * PLANES_PER_POSITION
    out_flat[base_history, :] = 1.0 if state.turn == 0 else 0.0
    out_flat[base_history + 1, :] = min(1.0, state.fullmove / 100.0)
    castling_planes = base_history + 2
    for idx in range(4):
        mask = 1 << idx
        out_flat[castling_planes + idx, :] = 1.0 if (state.castling & mask) else 0.0
    out_flat[castling_planes + 4, :] = min(1.0, state.halfmove / 100.0)


def _coerce_int(value: Any, default: int = 0) -> int:
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, np.generic):
        try:
            return int(value.item())
        except Exception:
            return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bitboard_to_mask(bitboard: Any) -> np.ndarray:
    bb = _coerce_int(bitboard, 0)
    packed = np.array([bb], dtype=np.uint64)
    bits = np.unpackbits(packed.view(np.uint8), bitorder="little")
    return bits[:NSQUARES].astype(bool, copy=False)


def _state_from_position(position: Any) -> PositionState:
    pieces_obj = getattr(position, "pieces", None)
    if pieces_obj is None:
        raise TypeError("position object does not expose 'pieces' attribute")
    try:
        pieces_seq = list(pieces_obj)
    except TypeError:
        pieces_seq = list(tuple(pieces_obj))

    boards: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for idx, name in enumerate(PIECE_NAMES):
        try:
            entry = pieces_seq[idx]
        except IndexError:
            entry = (0, 0)
        try:
            white_raw, black_raw = entry
        except Exception:
            white_raw, black_raw = 0, 0
        boards[name] = (_bitboard_to_mask(white_raw), _bitboard_to_mask(black_raw))

    turn_raw = _coerce_int(getattr(position, "turn", 0), 0)
    turn = int(turn_raw & 1)
    castling = _coerce_int(getattr(position, "castling", 0), 0)
    ep_attr = getattr(position, "ep_square", None)
    ep_square = None
    if ep_attr is not None:
        ep_val = _coerce_int(ep_attr, -1)
        if ep_val >= 0:
            ep_square = ep_val
    halfmove = _coerce_int(getattr(position, "halfmove", 0), 0)
    fullmove = _coerce_int(getattr(position, "fullmove", 1), 1)
    if fullmove <= 0:
        fullmove = 1

    return PositionState(
        boards=boards,
        turn=turn,
        castling=castling,
        ep_square=ep_square,
        halfmove=halfmove,
        fullmove=fullmove,
    )


def encode_move_index(move: Any) -> int:
    """Encode a move into the 73x64 policy index space."""
    from_sq = int(getattr(move, "from_square"))
    to_sq = int(getattr(move, "to_square"))
    fr, fc = divmod(from_sq, BOARD_SIZE)
    tr, tc = divmod(to_sq, BOARD_SIZE)
    dr = tr - fr
    dc = tc - fc
    promo = int(getattr(move, "promotion", 0))

    if promo in _PROMO_PIECE_ORDER:
        file_offset = _PROMO_FILE_OFFSETS.get(dc)
        if file_offset is None:
            return -1
        plane = 64 + _PROMO_PIECE_ORDER[promo] * 3 + file_offset
        return plane * NSQUARES + from_sq

    for idx, (ndr, ndc) in enumerate(_KNIGHT_OFFSETS):
        if dr == ndr and dc == ndc:
            plane = 56 + idx
            return plane * NSQUARES + from_sq

    if dr or dc:
        dist = max(abs(dr), abs(dc))
        if 1 <= dist <= 7:
            for dir_idx, (d_row, d_col) in enumerate(_DIR8):
                if dr == d_row * dist and dc == d_col * dist:
                    plane = dir_idx * 7 + (dist - 1)
                    return plane * NSQUARES + from_sq
    return -1


def encode_move_indices_batch(
    moves_lists: Sequence[Iterable[Any]],
) -> list[np.ndarray]:
    """Encode a batch of legal move lists into policy indices."""
    encoded: list[np.ndarray] = []
    for moves in moves_lists:
        indices = [encode_move_index(move) for move in moves]
        encoded.append(np.asarray(indices, dtype=np.int32))
    return encoded
