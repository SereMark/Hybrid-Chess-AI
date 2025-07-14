from typing import Dict, List, Optional
import random
import chess

OPENING_BOOK: Dict[str, List[str]] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [
        "e2e4",
        "d2d4",
        "g1f3",
        "c2c4",
    ],
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": [
        "e7e5",
        "c7c5",
        "e7e6",
        "c7c6",
        "d7d5",
    ],
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": [
        "d7d5",
        "g8f6",
        "f7f5",
        "e7e6",
    ],
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": [
        "g1f3",
        "b1c3",
        "f1c4",
        "f2f4",
    ],
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": [
        "g1f3",
        "b1c3",
        "c2c3",
        "f2f4",
    ],
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": [
        "b8c6",
        "g8f6",
        "f7f5",
        "d7d6",
    ],
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": [
        "f1b5",
        "f1c4",
        "d2d4",
        "b1c3",
    ],
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2": [
        "c2c4",
        "g1f3",
        "b1c3",
        "c1g5",
    ],
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2": [
        "c2c4",
        "g1f3",
        "b1c3",
        "c1f4",
    ],
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2": [
        "e7e6",
        "d5c4",
        "c7c6",
        "e7e5",
    ],
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1": [
        "d7d5",
        "g8f6",
        "c7c5",
        "g7g6",
    ],
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": [
        "f8c5",
        "f7f5",
        "g8f6",
        "f8e7",
    ],
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": [
        "a7a6",
        "g8f6",
        "f7f5",
        "f8c5",
    ],
}


def get_opening_move(fen: str) -> Optional[str]:
    if fen in OPENING_BOOK:
        moves = OPENING_BOOK[fen]
        return random.choice(moves) if moves else None
    return None


def is_in_opening_book(fen: str) -> bool:
    return fen in OPENING_BOOK


def get_opening_move_chess(board: chess.Board) -> Optional[chess.Move]:
    fen = board.fen()
    move_uci = get_opening_move(fen)
    if move_uci:
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                return move
        except (ValueError, chess.InvalidMoveError):
            pass
    return None


def extend_opening_book(fen: str, moves: List[str]) -> None:
    if fen not in OPENING_BOOK:
        OPENING_BOOK[fen] = moves
    else:
        for move in moves:
            if move not in OPENING_BOOK[fen]:
                OPENING_BOOK[fen].append(move)
