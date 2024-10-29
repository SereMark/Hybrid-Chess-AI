import chess
import numpy as np

MOVE_MAPPING = {}
INDEX_MAPPING = {}
TOTAL_MOVES = 0


def initialize_move_mappings():
    global MOVE_MAPPING, INDEX_MAPPING, TOTAL_MOVES
    index = 0
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            move = chess.Move(from_sq, to_sq)
            MOVE_MAPPING[index] = move
            INDEX_MAPPING[move] = index
            index += 1
            if chess.square_rank(to_sq) in [0, 7]:
                for promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    move = chess.Move(from_sq, to_sq, promotion=promo)
                    MOVE_MAPPING[index] = move
                    INDEX_MAPPING[move] = index
                    index += 1
    TOTAL_MOVES = index


def flip_board(board):
    flipped_board = board.mirror()
    return flipped_board


def flip_move(move):
    flipped_move = chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion
    )
    return flipped_move


def convert_board_to_tensor(board):
    planes = np.zeros((20, 8, 8), dtype=np.float32)
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
    for square, piece in board.piece_map().items():
        plane_idx = piece_type_indices[(piece.piece_type, piece.color)]
        row = square // 8
        col = square % 8
        planes[plane_idx, row, col] = 1

    castling = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]
    for i, right in enumerate(castling):
        if right:
            planes[12 + i, :, :] = 1

    if board.ep_square is not None:
        row = board.ep_square // 8
        col = board.ep_square % 8
        planes[16, row, col] = 1

    planes[17, :, :] = board.halfmove_clock / 100.0
    planes[18, :, :] = board.fullmove_number / 100.0
    planes[19, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return planes