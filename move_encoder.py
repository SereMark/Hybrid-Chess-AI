import chess
from config import (
    BLACK_PROMOTION_END,
    BLACK_PROMOTION_START,
    BOARD_DIM,
    PROMOTION_OFFSETS,
    SQUARES_COUNT,
    WHITE_PROMOTION_END,
    WHITE_PROMOTION_START,
)


class MoveEncoder:
    def __init__(self):
        self.move_to_idx = {}
        idx = 0

        for from_sq in range(SQUARES_COUNT):
            for to_sq in range(SQUARES_COUNT):
                if from_sq != to_sq and self._is_pseudo_legal(from_sq, to_sq):
                    self.move_to_idx[chess.Move(from_sq, to_sq)] = idx
                    idx += 1

        piece_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        for from_sq in range(WHITE_PROMOTION_START, WHITE_PROMOTION_END):
            for offset in PROMOTION_OFFSETS:
                to_sq = from_sq + offset
                if (
                    WHITE_PROMOTION_END <= to_sq < SQUARES_COUNT
                    and 0 <= (from_sq % BOARD_DIM + offset - BOARD_DIM) < BOARD_DIM
                ):
                    for piece in piece_types:
                        self.move_to_idx[chess.Move(from_sq, to_sq, piece)] = idx
                        idx += 1

        for from_sq in range(BLACK_PROMOTION_START, BLACK_PROMOTION_END):
            for offset in PROMOTION_OFFSETS:
                to_sq = from_sq - offset
                if (
                    0 <= to_sq < BLACK_PROMOTION_START
                    and 0 <= (from_sq % BOARD_DIM - offset + BOARD_DIM) < BOARD_DIM
                ):
                    for piece in piece_types:
                        self.move_to_idx[chess.Move(from_sq, to_sq, piece)] = idx
                        idx += 1

    def _is_pseudo_legal(self, from_sq: int, to_sq: int) -> bool:
        from_file, from_rank = from_sq % BOARD_DIM, from_sq // BOARD_DIM
        to_file, to_rank = to_sq % BOARD_DIM, to_sq // BOARD_DIM
        file_diff = abs(to_file - from_file)
        rank_diff = abs(to_rank - from_rank)

        if rank_diff in {0, file_diff} or file_diff == 0:
            return True
        if (file_diff == 2 and rank_diff == 1) or (file_diff == 1 and rank_diff == 2):
            return True
        return file_diff <= 1 and rank_diff <= 1

    def encode_move(self, move: chess.Move) -> int:
        return self.move_to_idx.get(move, -1)
