import chess
from config import get_config


class MoveEncoder:
    def __init__(self) -> None:
        self.move_to_idx: dict[chess.Move, int] = {}
        self.idx_to_move: dict[int, chess.Move] = {}
        self._build_move_map()

    def _build_move_map(self) -> None:
        idx = 0
        total_squares = get_config("game", "total_squares")
        for from_sq in range(total_squares):
            for to_sq in range(total_squares):
                if from_sq != to_sq:
                    move = chess.Move(from_sq, to_sq)
                    self.move_to_idx[move] = idx
                    self.idx_to_move[idx] = move
                    idx += 1
        promotion_pieces = get_config("game", "promotion_pieces")
        pawn_promotion_ranks = get_config("game", "pawn_promotion_ranks")
        for color_name in ["white", "black"]:
            from_start, from_end, to_start, to_end = pawn_promotion_ranks[color_name]
            for from_sq in range(from_start, from_end):
                for to_sq in range(to_start, to_end):
                    for promo in promotion_pieces:
                        move = chess.Move(from_sq, to_sq, promotion=promo)
                        self.move_to_idx[move] = idx
                        self.idx_to_move[idx] = move
                        idx += 1

    def encode_move(self, move: chess.Move) -> int:
        return self.move_to_idx.get(move, 0)
