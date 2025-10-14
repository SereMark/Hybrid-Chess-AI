from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import chesscore as ccore
import numpy as np

BOARD_SIZE = 8
NSQUARES = BOARD_SIZE * BOARD_SIZE
PLANES_PER_POSITION = 14
HISTORY_LENGTH = 8
INPUT_PLANES = HISTORY_LENGTH * PLANES_PER_POSITION + 7
POLICY_PLANES = 73
POLICY_SIZE = POLICY_PLANES * NSQUARES

PIECE_NAMES = ("pawn", "knight", "bishop", "rook", "queen", "king")

@dataclass(slots=True)
class PositionState:
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


def encode_position(position: Any, history: Sequence[Any] | None = None) -> np.ndarray:
    """Encode position + limited history to (INPUT_PLANES, 8, 8) float32."""
    states: list[PositionState] = []
    if history:
        states.extend(_ensure_state(h) for h in history)
    states.append(_ensure_state(position))

    out = np.zeros((INPUT_PLANES, NSQUARES), dtype=np.float32)
    avail = min(HISTORY_LENGTH, len(states))
    for off in range(avail):
        st = states[-1 - off]
        base = off * PLANES_PER_POSITION
        _write_piece_planes(st, base, out)
        if off == 0:
            _write_aux_planes(st, out)
    return out.reshape(INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)


def encode_batch(positions: Sequence[Any]) -> np.ndarray:
    if not positions:
        return np.zeros((0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    return np.stack([encode_position(p) for p in positions], axis=0)


def _ensure_state(position: Any) -> PositionState:
    return position if isinstance(position, PositionState) else _state_from_position(position)


def _write_piece_planes(st: PositionState, base: int, out_flat: np.ndarray) -> None:
    for i, name in enumerate(PIECE_NAMES):
        w, b = st.boards.get(name, (_EMPTY_MASK, _EMPTY_MASK))
        pw = base + i * 2
        pb = pw + 1
        if w.any():
            out_flat[pw, w] = 1.0
        if b.any():
            out_flat[pb, b] = 1.0


def _write_aux_planes(st: PositionState, out_flat: np.ndarray) -> None:
    base = HISTORY_LENGTH * PLANES_PER_POSITION
    out_flat[base, :] = 1.0 if st.turn == 0 else 0.0
    out_flat[base + 1, :] = min(1.0, st.fullmove / 100.0)
    for i in range(4):
        out_flat[base + 2 + i, :] = 1.0 if (st.castling & (1 << i)) else 0.0
    out_flat[base + 6, :] = min(1.0, st.halfmove / 100.0)


def _coerce_int(v: Any, default: int = 0) -> int:
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
    x = np.uint64(_coerce_int(bb, 0))
    packed = np.array([x], dtype=np.uint64)
    bits = np.unpackbits(packed.view(np.uint8), bitorder="little")
    return bits[:NSQUARES].astype(bool, copy=False)


def _state_from_position(position: Any) -> PositionState:
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
    return PositionState(boards=boards, turn=turn, castling=castling, ep_square=ep_sq, halfmove=halfmove, fullmove=fullmove)


def encode_move_index(move: Any) -> int:
    """Return policy index in 73×64 space or -1 if unsupported."""
    f = int(getattr(move, "from_square"))
    t = int(getattr(move, "to_square"))
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


def encode_move_indices_batch(moves_lists: Sequence[Iterable[Any]]) -> list[np.ndarray]:
    return [np.asarray([encode_move_index(m) for m in moves], dtype=np.int32) for moves in moves_lists]