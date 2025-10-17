"""Position encoders and policy indexing helpers matched to chesscore bindings."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import chesscore as ccore
import numpy as np

BOARD_SIZE = int(getattr(ccore, "BOARD_SIZE", 8))
NSQUARES = BOARD_SIZE * BOARD_SIZE
PLANES_PER_POSITION = 14
HISTORY_LENGTH = 8
INPUT_PLANES = HISTORY_LENGTH * PLANES_PER_POSITION + 7
POLICY_SIZE = int(getattr(ccore, "POLICY_SIZE", 73 * NSQUARES))
POLICY_PLANES = POLICY_SIZE // NSQUARES

PIECE_NAMES = ("pawn", "knight", "bishop", "rook", "queen", "king")

PositionLike = Any

__all__ = [
    "PositionState",
    "encode_position",
    "encode_batch",
    "encode_move_index",
    "encode_move_indices_batch",
    "BOARD_SIZE",
    "INPUT_PLANES",
    "POLICY_SIZE",
]


@dataclass(slots=True)
class PositionState:
    """Lightweight snapshot of the information required for encoding."""

    boards: Dict[str, Tuple[np.ndarray, np.ndarray]]
    turn: int
    castling: int
    ep_square: int | None
    halfmove: int
    fullmove: int


_EMPTY_MASK = np.zeros(NSQUARES, dtype=bool)
_DIR8: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_KNIGHT: tuple[tuple[int, int], ...] = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
_PROMO_ORDER = {int(ccore.Piece.KNIGHT): 0, int(ccore.Piece.ROOK): 1, int(ccore.Piece.BISHOP): 2}
_PROMO_FILE = {0: 0, -1: 1, 1: 2}

_HAS_NATIVE_MOVE_ENCODE = hasattr(ccore, "encode_move_index")
_HAS_NATIVE_MOVE_BATCH = hasattr(ccore, "encode_move_indices_batch")


# ---------------------------------------------------------------------------#
# Position encoding
# ---------------------------------------------------------------------------#


def encode_position(position: PositionLike, history: Sequence[PositionLike] | None = None) -> np.ndarray:
    """Encode a position (optionally with history) to `(INPUT_PLANES, 8, 8)` floats."""
    states: list[PositionState] = []
    if history:
        states.extend(_ensure_state(entry) for entry in history)
    states.append(_ensure_state(position))

    out = np.zeros((INPUT_PLANES, NSQUARES), dtype=np.float32)
    avail = min(HISTORY_LENGTH, len(states))
    for offset in range(avail):
        snapshot = states[-1 - offset]
        base = offset * PLANES_PER_POSITION
        _write_piece_planes(snapshot, base, out)
        if offset == 0:
            _write_aux_planes(snapshot, out)
    return out.reshape(INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)


def encode_batch(positions: Sequence[Any]) -> np.ndarray:
    """Vectorised wrapper around `encode_position` for sequences of positions."""
    if not positions:
        return np.zeros((0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    return np.stack([encode_position(position) for position in positions], axis=0)


def _ensure_state(position: Any) -> PositionState:
    """Return a `PositionState` wrapper, converting from bindings where necessary."""
    return position if isinstance(position, PositionState) else _state_from_position(position)


def _write_piece_planes(st: PositionState, base: int, out_flat: np.ndarray) -> None:
    """Populate piece occupancy planes for both colours."""
    for i, name in enumerate(PIECE_NAMES):
        w, b = st.boards.get(name, (_EMPTY_MASK, _EMPTY_MASK))
        pw = base + i * 2
        pb = pw + 1
        if w.any():
            out_flat[pw, w] = 1.0
        if b.any():
            out_flat[pb, b] = 1.0


def _write_aux_planes(st: PositionState, out_flat: np.ndarray) -> None:
    """Write side-to-move, move counters, and castling rights planes."""
    base = HISTORY_LENGTH * PLANES_PER_POSITION
    out_flat[base, :] = 1.0 if st.turn == 0 else 0.0
    out_flat[base + 1, :] = min(1.0, st.fullmove / 100.0)
    for i in range(4):
        out_flat[base + 2 + i, :] = 1.0 if (st.castling & (1 << i)) else 0.0
    out_flat[base + 6, :] = min(1.0, st.halfmove / 100.0)


def _coerce_int(v: Any, default: int = 0) -> int:
    """Convert a value to `int`, returning `default` on failure."""
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, np.generic):
        with np.errstate(all="ignore"):
            try:
                return int(v.item())
            except Exception:
                return default
    try:
        return int(v)
    except Exception:
        return default


def _bitboard_to_mask(bb: Any) -> np.ndarray:
    """Convert a bitboard into a boolean mask aligned with plane layout."""
    x = int(np.uint64(_coerce_int(bb, 0)))
    nbits = NSQUARES
    nbytes = (nbits + 7) // 8
    packed = np.array([x], dtype=np.uint64).view(np.uint8)
    if packed.size < nbytes:
        packed = np.pad(packed, (0, nbytes - packed.size), mode="constant")
    bits = np.unpackbits(packed[:nbytes], bitorder="little")[:nbits]
    return bits.astype(bool, copy=False)


def _state_from_position(position: Any) -> PositionState:
    """Extract minimal encoding state from a `ccore.Position` or equivalent."""
    pieces_obj = getattr(position, "pieces", None)
    if pieces_obj is None:
        raise TypeError("position lacks 'pieces'")
    try:
        seq = list(pieces_obj)
    except TypeError:
        seq = list(tuple(pieces_obj))

    boards: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i, name in enumerate(PIECE_NAMES):
        entry = seq[i] if i < len(seq) else (0, 0)
        try:
            w_raw, b_raw = entry
        except Exception:
            w_raw, b_raw = 0, 0
        boards[name] = (_bitboard_to_mask(w_raw), _bitboard_to_mask(b_raw))

    turn = _coerce_int(getattr(position, "turn", 0), 0) & 1
    castling = _coerce_int(getattr(position, "castling", 0), 0)
    ep_attr = getattr(position, "ep_square", None)
    ep_sq = None
    if ep_attr is not None:
        e = _coerce_int(ep_attr, -1)
        if e >= 0:
            ep_sq = e
    halfmove = _coerce_int(getattr(position, "halfmove", 0), 0)
    fullmove = max(1, _coerce_int(getattr(position, "fullmove", 1), 1))
    return PositionState(
        boards=boards, turn=turn, castling=castling, ep_square=ep_sq, halfmove=halfmove, fullmove=fullmove
    )


# ---------------------------------------------------------------------------#
# Move indexing
# ---------------------------------------------------------------------------#


def _encode_move_index_python(move: Any) -> int:
    """Pure Python fallback mirroring the chesscore move encoding."""
    try:
        f = int(getattr(move, "from_square"))
        t = int(getattr(move, "to_square"))
    except Exception:
        return -1
    fr, fc = divmod(f, BOARD_SIZE)
    tr, tc = divmod(t, BOARD_SIZE)
    dr, dc = tr - fr, tc - fc
    promo = int(getattr(move, "promotion", 0))

    if promo in _PROMO_ORDER:
        off = _PROMO_FILE.get(dc)
        if off is None:
            return -1
        plane = 64 + _PROMO_ORDER[promo] * 3 + off
        return plane * NSQUARES + f

    for k, (kr, kc) in enumerate(_KNIGHT):
        if dr == kr and dc == kc:
            plane = 56 + k
            return plane * NSQUARES + f

    if dr or dc:
        dist = max(abs(dr), abs(dc))
        if 1 <= dist <= 7:
            for d, (vr, vc) in enumerate(_DIR8):
                if dr == vr * dist and dc == vc * dist:
                    plane = d * 7 + (dist - 1)
                    return plane * NSQUARES + f
    return -1


def encode_move_index(move: Any) -> int:
    """Return the canonical 73x64 policy index for a move, or -1 if unsupported."""
    if _HAS_NATIVE_MOVE_ENCODE:
        try:
            return int(ccore.encode_move_index(move))
        except Exception:
            return _encode_move_index_python(move)
    return _encode_move_index_python(move)


def encode_move_indices_batch(moves_lists: Sequence[Iterable[Any]]) -> list[np.ndarray]:
    """Vectorised encoding for batches of move lists supporting the Python fallback."""
    materialised = [list(moves) for moves in moves_lists]
    if _HAS_NATIVE_MOVE_BATCH:
        try:
            native = ccore.encode_move_indices_batch(materialised)
            return [np.asarray(arr, dtype=np.int32) for arr in native]
        except Exception:
            pass
    return [np.asarray([encode_move_index(move) for move in moves], dtype=np.int32) for moves in materialised]
