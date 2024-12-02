import chess, numpy as np

class MoveMapping:
    def __init__(self):
        self.MOVE_MAPPING = {}
        self.INDEX_MAPPING = {}
        self.TOTAL_MOVES = 0
        self._initialize_move_mappings()

    def _initialize_move_mappings(self):
        index = 0
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq == to_sq:
                    continue
                move = chess.Move(from_sq, to_sq)
                if chess.Move.null() != move:
                    self.MOVE_MAPPING[index] = move
                    self.INDEX_MAPPING[move] = index
                    index += 1
                    if chess.square_rank(to_sq) in [0, 7]:
                        for promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                            promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                            self.MOVE_MAPPING[index] = promo_move
                            self.INDEX_MAPPING[promo_move] = index
                            index += 1
        self.TOTAL_MOVES = index

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
        plane_idx = piece_type_indices.get((piece.piece_type, piece.color))
        if plane_idx is not None:
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

def get_game_result(board):
    result = board.result()
    if result == '1-0':
        return 1.0
    elif result == '0-1':
        return -1.0
    else:
        return 0.0