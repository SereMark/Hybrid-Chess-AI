import chess
import numpy as np
from itertools import product

class MoveMapping:
    def __init__(self):
        moves = []
        for from_sq, to_sq in product(chess.SQUARES, repeat=2):
            if from_sq == to_sq:
                continue

            # Normal move
            moves.append(chess.Move(from_sq, to_sq))

            # Promotion moves (if the target square is on the 1st or 8th rank)
            if chess.square_rank(to_sq) in (0, 7):
                for promo in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                    moves.append(chess.Move(from_sq, to_sq, promotion=promo))

        # Build index <-> move mappings
        self.MOVE_MAPPING = dict(enumerate(moves))
        self.INDEX_MAPPING = {move: idx for idx, move in self.MOVE_MAPPING.items()}
        self.TOTAL_MOVES = len(moves)

    def get_move_by_index(self, index):
        return self.MOVE_MAPPING.get(index)

    def get_index_by_move(self, move):
        return self.INDEX_MAPPING.get(move)

# Create a global instance of the MoveMapping
move_mapping = MoveMapping()


def get_total_moves():
    return move_mapping.TOTAL_MOVES

def get_move_mapping():
    return move_mapping

def flip_board(board):
    return board.mirror()

def flip_move(move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion
    )

def convert_board_to_tensor(board):
    """
    Converts the given chess.Board into a 25 x 8 x 8 NumPy tensor (float32).
    
    Plane breakdown (indices in the 0th dimension):
    0-11 : One-hot planes for piece type & color
           [ White Pawn, White Knight, White Bishop, White Rook, White Queen, White King,
             Black Pawn, Black Knight, Black Bishop, Black Rook, Black Queen, Black King ]
    12-15: Castling rights
           [ White king-side, White queen-side, Black king-side, Black queen-side ]
    16   : En passant square (if any)
    17   : Normalized halfmove clock (divided by 100)
    18   : Normalized fullmove number (divided by 100)
    19   : Whose turn plane (1 if white, 0 if black)
    20   : Threefold repetition indicator (1 if the position has repeated at least 3 times)
    21   : Squares attacked by white
    22   : Squares attacked by black
    23   : White passed pawns
    24   : Black passed pawns
    """
    planes = np.zeros((25, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()

    # Map (piece_type, color) to plane index
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

    # 1) Encode piece positions
    for sq, piece in piece_map.items():
        idx = piece_type_indices.get((piece.piece_type, piece.color))
        if idx is not None:
            row, col = divmod(sq, 8)
            planes[idx, row, col] = 1.0

    # 2) Encode castling rights
    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]
    for i, c_right in enumerate(castling_rights):
        planes[12 + i, 0, 0] = float(c_right)

    # 3) Encode en passant square
    if board.ep_square is not None:
        ep_row, ep_col = divmod(board.ep_square, 8)
        planes[16, ep_row, ep_col] = 1.0

    # 4) Normalize halfmove clock and fullmove number
    planes[17, 0, 0] = board.halfmove_clock / 100.0
    planes[18, 0, 0] = board.fullmove_number / 100.0

    # 5) Encode turn (white=1, black=0)
    planes[19, 0, 0] = 1.0 if board.turn == chess.WHITE else 0.0

    # 6) Encode repetition count (3-fold)
    planes[20, 0, 0] = 1.0 if board.is_repetition(3) else 0.0

    # 7) Encode squares attacked by white (plane 21) and black (plane 22)
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

    # 8) Encode passed pawns (plane 23 for white, 24 for black)
    for sq, piece in piece_map.items():
        if piece.piece_type == chess.PAWN and is_passed_pawn(board, sq):
            row, col = divmod(sq, 8)
            if piece.color == chess.WHITE:
                planes[23, row, col] = 1.0
            else:
                planes[24, row, col] = 1.0

    return planes

def is_passed_pawn(board, square):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    color = board.color_at(square)
    if not color:
        return False  # No piece at this square

    # Get all enemy pawns
    enemy_pawns = list(board.pieces(chess.PAWN, not color))

    # The set of files to check for enemy pawns
    valid_files = {file}
    if file > 0:
        valid_files.add(file - 1)
    if file < 7:
        valid_files.add(file + 1)

    # For white, any black pawns with rank > this pawn's rank block it from being passed
    # For black, any white pawns with rank < this pawn's rank block it
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