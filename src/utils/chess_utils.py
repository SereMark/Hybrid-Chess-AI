import chess
import numpy as np

class MoveMapping:
    def __init__(self):
        self.MOVE_MAPPING = {}
        self.INDEX_MAPPING = {}
        self.TOTAL_MOVES = 0
        idx = 0
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq == to_sq:
                    continue
                move = chess.Move(from_sq, to_sq)
                if move != chess.Move.null():
                    self.MOVE_MAPPING[idx] = move
                    self.INDEX_MAPPING[move] = idx
                    idx += 1
                    if chess.square_rank(to_sq) in [0, 7]:
                        for promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                            promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                            self.MOVE_MAPPING[idx] = promo_move
                            self.INDEX_MAPPING[promo_move] = idx
                            idx += 1
        self.TOTAL_MOVES = idx

    def get_move_by_index(self, index):
        return self.MOVE_MAPPING.get(index)

    def get_index_by_move(self, move):
        return self.INDEX_MAPPING.get(move)

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
    planes = np.zeros((25, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    piece_type_indices = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    # Encode piece positions
    for sq, piece in piece_map.items():
        idx = piece_type_indices.get((piece.piece_type, piece.color))
        if idx is not None:
            row, col = divmod(sq, 8)
            planes[idx, row, col] = 1

    # Encode castling rights
    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]
    for i, c_right in enumerate(castling_rights):
        planes[12 + i, 0, 0] = float(c_right)

    # Encode en passant square
    if board.ep_square is not None:
        ep_row, ep_col = divmod(board.ep_square, 8)
        planes[16, ep_row, ep_col] = 1

    # Normalize halfmove clock and fullmove number
    planes[17, 0, 0] = board.halfmove_clock / 100.0
    planes[18, 0, 0] = board.fullmove_number / 100.0

    # Encode turn
    planes[19, 0, 0] = 1.0 if board.turn == chess.WHITE else 0.0

    # Encode repetition count
    planes[20, 0, 0] = 1.0 if board.is_repetition(3) else 0.0

    # Encode attacks
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            attacks = board.attacks(sq)
            if piece.color == chess.WHITE:
                for attack_sq in attacks:
                    planes[21, *divmod(attack_sq, 8)] = 1
            else:
                for attack_sq in attacks:
                    planes[22, *divmod(attack_sq, 8)] = 1

    # Encode passed pawns
    for sq, piece in piece_map.items():
        if piece.piece_type == chess.PAWN and is_passed_pawn(board, sq):
            row, col = divmod(sq, 8)
            if piece.color == chess.WHITE:
                planes[23, row, col] = 1
            else:
                planes[24, row, col] = 1

    return planes

def is_passed_pawn(board, square):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    color = board.color_at(square)
    enemy_pawns = [sq for sq in board.pieces(chess.PAWN, not color)]

    valid_files = {file}
    if file > 0:
        valid_files.add(file - 1)
    if file < 7:
        valid_files.add(file + 1)

    if color == chess.WHITE:
        return not any(
            chess.square_file(sq) in valid_files and chess.square_rank(sq) > rank
            for sq in enemy_pawns
        )
    else:
        return not any(
            chess.square_file(sq) in valid_files and chess.square_rank(sq) < rank
            for sq in enemy_pawns
        )