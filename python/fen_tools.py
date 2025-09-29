from __future__ import annotations

from collections.abc import Iterable

_CASTLING_ORDER = "KQkq"
_FILES = "abcdefgh"


def _flip_square(square: str) -> str:
    if not square or square == "-":
        return "-"
    sq = square.strip()
    if len(sq) != 2:
        return "-"
    file_ch, rank_ch = sq[0], sq[1]
    if file_ch not in _FILES or not rank_ch.isdigit():
        return "-"
    rank_val = int(rank_ch)
    if not 1 <= rank_val <= 8:
        return "-"
    flipped_file = _FILES[::-1][_FILES.index(file_ch)]
    flipped_rank = str(9 - rank_val)
    return f"{flipped_file}{flipped_rank}"


def _swap_castling_rights(rights: str) -> str:
    if not rights or rights == "-":
        return "-"
    swapped: list[str] = []
    for ch in rights:
        if ch in _CASTLING_ORDER:
            swapped.append(ch.lower() if ch.isupper() else ch.upper())
    swapped = list(dict.fromkeys(swapped))
    ordered = [c for c in _CASTLING_ORDER if c in swapped]
    return "".join(ordered) if ordered else "-"


def sanitize_fen(fen: str) -> str:
    parts = fen.strip().split()
    if len(parts) < 6:
        defaults = ["w", "-", "-", "0", "1"]
        parts += defaults[len(parts) - 1 :]
    if len(parts) > 6:
        head = parts[:6]
        tail = parts[6:]
    else:
        head = parts
        tail: list[str] = []
    if len(head) < 6:
        head += ["-"] * (6 - len(head))
        head[1] = "w"
        head[4] = "0"
        head[5] = "1"
    sanitized = " ".join(head)
    if tail:
        sanitized = " ".join([sanitized, *tail])
    return sanitized


def flip_fen_perspective(fen: str) -> str:
    board, side, castling, ep, halfmove, fullmove, *rest = sanitize_fen(fen).split()
    rows = board.split("/")
    flipped_rows: list[str] = []
    for row in reversed(rows):
        new_row_chars: list[str] = []
        for ch in row:
            if ch.isdigit():
                new_row_chars.append(ch)
            elif ch.isalpha():
                new_row_chars.append(ch.swapcase())
            else:
                new_row_chars.append(ch)
        flipped_rows.append("".join(new_row_chars))
    flipped_board = "/".join(flipped_rows)
    flipped_side = "b" if side.lower() == "w" else "w"
    flipped_castling = _swap_castling_rights(castling)
    flipped_ep = _flip_square(ep)
    flipped_fullmove = fullmove
    flipped_halfmove = halfmove
    components: list[str] = [
        flipped_board,
        flipped_side,
        flipped_castling,
        flipped_ep,
        flipped_halfmove,
        flipped_fullmove,
    ]
    if rest:
        components.extend(rest)
    return " ".join(components)


def iter_with_perspectives(fens: Iterable[str]) -> list[str]:
    out: list[str] = []
    for fen in fens:
        cleaned = sanitize_fen(fen)
        out.append(cleaned)
        out.append(flip_fen_perspective(cleaned))
    return out
