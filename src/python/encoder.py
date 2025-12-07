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
    "encode_canonical_move",
    "encode_canonical_move_indices_batch",
    "BOARD_SIZE",
    "INPUT_PLANES",
    "POLICY_SIZE",
]


@dataclass(slots=True)
class PositionState:
    boards: Dict[str, Tuple[np.ndarray, np.ndarray]]
    turn: int
    castling: int
    ep_square: int | None
    halfmove: int
    fullmove: int
    repetition: int


_EMPTY_MASK = np.zeros(NSQUARES, dtype=bool)
_DIR8: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_KNIGHT: tuple[tuple[int, int], ...] = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
_PROMO_ORDER = {int(ccore.Piece.KNIGHT): 0, int(ccore.Piece.ROOK): 1, int(ccore.Piece.BISHOP): 2}
_PROMO_FILE = {0: 0, -1: 1, 1: 2}

_HAS_NATIVE_MOVE_ENCODE = hasattr(ccore, "encode_move_index")
_HAS_NATIVE_MOVE_BATCH = hasattr(ccore, "encode_move_indices_batch")


def _flip_state(st: PositionState) -> PositionState:
    new_boards = {}
    for name, (w, b) in st.boards.items():
        w_flipped = w.reshape(8, 8)[::-1, :].flatten()
        b_flipped = b.reshape(8, 8)[::-1, :].flatten()
        new_boards[name] = (b_flipped, w_flipped)

    new_ep = (st.ep_square ^ 56) if st.ep_square is not None else None

    c = st.castling
    new_c = 0
    if c & 1:
        new_c |= 4
    if c & 2:
        new_c |= 8
    if c & 4:
        new_c |= 1
    if c & 8:
        new_c |= 2

    return PositionState(
        boards=new_boards,
        turn=0,
        castling=new_c,
        ep_square=new_ep,
        halfmove=st.halfmove,
        fullmove=st.fullmove,
        repetition=st.repetition,
    )


def encode_position(position: PositionLike, history: Sequence[PositionLike] | None = None) -> np.ndarray:
    current_st = _ensure_state(position)
    must_flip = current_st.turn == 1

    states: list[PositionState] = []
    if history:
        for entry in history:
            st = _ensure_state(entry)
            states.append(_flip_state(st) if must_flip else st)

    states.append(_flip_state(current_st) if must_flip else current_st)

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
    if not positions:
        return np.zeros((0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    return np.stack([encode_position(position) for position in positions], axis=0)


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

    if st.ep_square is not None and 0 <= st.ep_square < NSQUARES:
        out_flat[base + 12, st.ep_square] = 1.0

    if st.repetition >= 2:
        out_flat[base + 13, :] = 1.0


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
    x = int(np.uint64(_coerce_int(bb, 0)))
    nbits = NSQUARES
    nbytes = (nbits + 7) // 8
    packed = np.array([x], dtype=np.uint64).view(np.uint8)
    if packed.size < nbytes:
        packed = np.pad(packed, (0, nbytes - packed.size), mode="constant")
    bits = np.unpackbits(packed[:nbytes], bitorder="little")[:nbits]
    return bits.astype(bool, copy=False)


def _state_from_position(position: Any) -> PositionState:
    pieces_obj = getattr(position, "pieces", None)
    if pieces_obj is None:
        raise TypeError("a pozíció objektumnak nincs 'pieces' attribútuma")
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

    repetition = 0
    count_fn = getattr(position, "count_repetitions", None)
    if callable(count_fn):
        repetition = _coerce_int(count_fn(), 0)
    elif hasattr(position, "repetition_count"):
        val = getattr(position, "repetition_count")
        repetition = _coerce_int(val() if callable(val) else val, 0)

    return PositionState(
        boards=boards,
        turn=turn,
        castling=castling,
        ep_square=ep_sq,
        halfmove=halfmove,
        fullmove=fullmove,
        repetition=repetition,
    )


class _FakeMove:
    __slots__ = ("from_square", "to_square", "promotion")

    def __init__(self, f: int, t: int, p: int) -> None:
        self.from_square = f
        self.to_square = t
        self.promotion = p


def _encode_move_index_python(move: Any) -> int:
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
    if _HAS_NATIVE_MOVE_ENCODE:
        try:
            return int(ccore.encode_move_index(move))
        except Exception:
            return _encode_move_index_python(move)
    return _encode_move_index_python(move)


def encode_canonical_move(move: Any, turn: int) -> int:
    flip = turn == 1
    if _HAS_NATIVE_MOVE_ENCODE:
        try:
            return int(ccore.encode_move_index(move, flip))
        except Exception:
            pass

    if flip:
        try:
            f = int(getattr(move, "from_square"))
            t = int(getattr(move, "to_square"))
            p = int(getattr(move, "promotion", 0))
            return _encode_move_index_python(_FakeMove(f ^ 56, t ^ 56, p))
        except Exception:
            return -1
    return _encode_move_index_python(move)


def encode_canonical_move_indices_batch(moves_lists: Sequence[Iterable[Any]], turns: Sequence[int]) -> list[np.ndarray]:
    materialised = [list(moves) for moves in moves_lists]
    if _HAS_NATIVE_MOVE_BATCH:
        try:
            native = ccore.encode_move_indices_batch(materialised, turns)
            return [np.asarray(arr, dtype=np.int32) for arr in native]
        except Exception:
            pass

    return [
        np.asarray([encode_canonical_move(move, t) for move in moves], dtype=np.int32)
        for moves, t in zip(materialised, turns)
    ]
