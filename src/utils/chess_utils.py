import chess
import numpy as np
from itertools import product

class MoveMapping:
    def __init__(self):
        moves = []
        for from_sq, to_sq in product(chess.SQUARES, repeat=2):
            if from_sq == to_sq:
                continue

            moves.append(chess.Move(from_sq, to_sq))

            if chess.square_rank(to_sq) in (0, 7):
                for promo in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                    moves.append(chess.Move(from_sq, to_sq, promotion=promo))

        self.MOVE_MAPPING = dict(enumerate(moves))
        self.INDEX_MAPPING = {move: idx for idx, move in self.MOVE_MAPPING.items()}
        self.TOTAL_MOVES = len(moves)

    def get_move_by_index(self, index):
        return self.MOVE_MAPPING.get(index)

    def get_index_by_move(self, move):
        return self.INDEX_MAPPING.get(move)

move_mapping = MoveMapping()

def get_total_moves():
    return move_mapping.TOTAL_MOVES

def get_move_mapping():
    return move_mapping

def flip_board(board: chess.Board) -> chess.Board:
    return board.mirror()

def flip_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion
    )

def mirror_rank(board: chess.Board) -> chess.Board:
    fen_parts = board.fen().split(" ")
    board_rows = fen_parts[0].split("/")
    reversed_rows = "/".join(reversed(board_rows))
    reversed_fen = reversed_rows + " " + " ".join(fen_parts[1:])
    return chess.Board(reversed_fen)

def mirror_move_rank(move: chess.Move) -> chess.Move:
    from_sq_rank = 7 - chess.square_rank(move.from_square)
    from_sq_file = chess.square_file(move.from_square)
    to_sq_rank = 7 - chess.square_rank(move.to_square)
    to_sq_file = chess.square_file(move.to_square)

    from_sq_mirrored = chess.square(from_sq_file, from_sq_rank)
    to_sq_mirrored = chess.square(to_sq_file, to_sq_rank)

    return chess.Move(from_sq_mirrored, to_sq_mirrored, promotion=move.promotion)

def convert_board_to_tensor(board: chess.Board) -> np.ndarray:
    planes = np.zeros((25, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()

    piece_type_indices = {
        (chess.PAWN,   chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK,   chess.WHITE): 3,
        (chess.QUEEN,  chess.WHITE): 4,
        (chess.KING,   chess.WHITE): 5,
        (chess.PAWN,   chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK,   chess.BLACK): 9,
        (chess.QUEEN,  chess.BLACK): 10,
        (chess.KING,   chess.BLACK): 11,
    }

    for sq, piece in piece_map.items():
        idx = piece_type_indices.get((piece.piece_type, piece.color))
        if idx is not None:
            row, col = divmod(sq, 8)
            planes[idx, row, col] = 1.0

    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]
    for i, right in enumerate(castling_rights):
        planes[12 + i, 0, 0] = float(right)

    if board.ep_square is not None:
        ep_row, ep_col = divmod(board.ep_square, 8)
        planes[16, ep_row, ep_col] = 1.0

    planes[17, 0, 0] = board.halfmove_clock / 100.0
    planes[18, 0, 0] = board.fullmove_number / 100.0

    planes[19, 0, 0] = 1.0 if board.turn else 0.0

    planes[20, 0, 0] = 1.0 if board.is_repetition(3) else 0.0

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            attacks = board.attacks(sq)
            if piece.color == chess.WHITE:
                for attack_sq in attacks:
                    r, c = divmod(attack_sq, 8)
                    planes[21, r, c] = 1.0
            else:
                for attack_sq in attacks:
                    r, c = divmod(attack_sq, 8)
                    planes[22, r, c] = 1.0

    for sq, piece in piece_map.items():
        if piece.piece_type == chess.PAWN and is_passed_pawn(board, sq):
            row, col = divmod(sq, 8)
            if piece.color == chess.WHITE:
                planes[23, row, col] = 1.0
            else:
                planes[24, row, col] = 1.0

    return planes

def is_passed_pawn(board: chess.Board, square: int) -> bool:
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    color = board.color_at(square)
    if color is None:
        return False

    enemy_pawns = list(board.pieces(chess.PAWN, not color))

    valid_files = {file}
    if file > 0:
        valid_files.add(file - 1)
    if file < 7:
        valid_files.add(file + 1)

    if color == chess.WHITE:
        return not any(
            chess.square_file(sq) in valid_files
            and chess.square_rank(sq) > rank
            for sq in enemy_pawns
        )
    else:
        return not any(
            chess.square_file(sq) in valid_files
            and chess.square_rank(sq) < rank
            for sq in enemy_pawns
        )

def apply_augmentations(board: chess.Board, move: chess.Move, method: str = "flip"):
    if method == "flip":
        return flip_board(board), flip_move(move)
    elif method == "mirror_rank":
        return mirror_rank(board), mirror_move_rank(move)
    else:
        return board, move