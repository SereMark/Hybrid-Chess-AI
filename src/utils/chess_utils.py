import h5py
import torch
import chess
import numpy as np
from itertools import product
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, path, indices):
        self.path = path
        self.indices = indices
        self.file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, x):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        return tuple(
            torch.tensor(self.file[k][self.indices[x]], dtype=t)
            for k, t in zip(['inputs', 'policy_targets', 'value_targets'],
                            [torch.float32, torch.long, torch.float32])
        )

    def __del__(self):
        if self.file:
            self.file.close()

class MoveMapping:
    def __init__(self):
        moves = []
        for f, t in product(chess.SQUARES, repeat=2):
            if f != t:
                moves.append(chess.Move(f, t))
                if chess.square_rank(t) in (0, 7):
                    for promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                        moves.append(chess.Move(f, t, promotion=promotion))
        self.MOVE_MAPPING = dict(enumerate(moves))
        self.INDEX_MAPPING = {v: k for k, v in self.MOVE_MAPPING.items()}
        self.TOTAL_MOVES = len(moves)

    def get_move_by_index(self, i):
        return self.MOVE_MAPPING.get(i)

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

def mirror_rank(board):
    fen_parts = board.fen().split()
    ranks = fen_parts[0].split("/")
    reversed_ranks = "/".join(reversed(ranks)) + " " + " ".join(fen_parts[1:])
    return chess.Board(reversed_ranks)

def mirror_move_rank(move):
    from_rank = 7 - chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)
    to_rank = 7 - chess.square_rank(move.to_square)
    to_file = chess.square_file(move.to_square)
    return chess.Move(
        chess.square(to_file, to_rank),
        chess.square(from_file, from_rank),
        promotion=move.promotion
    )

def convert_single_board(board):
    x = np.zeros((64, 18), np.float32)
    piece_map = {
        (chess.PAWN, True): 0, (chess.KNIGHT, True): 1, (chess.BISHOP, True): 2,
        (chess.ROOK, True): 3, (chess.QUEEN, True): 4, (chess.KING, True): 5,
        (chess.PAWN, False): 6, (chess.KNIGHT, False): 7, (chess.BISHOP, False): 8,
        (chess.ROOK, False): 9, (chess.QUEEN, False): 10, (chess.KING, False): 11
    }
    for square, piece in board.piece_map().items():
        idx = piece_map.get((piece.piece_type, piece.color))
        if idx is not None:
            x[square, idx] = 1
    if board.turn:
        x[:, 12] = 1
    ep = board.ep_square
    if ep is not None:
        x[ep, 13] = 1
    if board.has_kingside_castling_rights(chess.WHITE):
        x[:, 14] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        x[:, 15] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        x[:, 16] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        x[:, 17] = 1
    return x

def convert_board_to_transformer_input(board):
    temp_board = board.copy()
    boards = [temp_board.copy()]
    history = list(temp_board.move_stack)
    for _ in range(7):
        if not history:
            break
        history.pop()
        temp_board.pop()
        boards.insert(0, temp_board.copy())
    while len(boards) < 8:
        boards.insert(0, boards[0].copy())
    return np.concatenate([convert_single_board(b) for b in boards], axis=1)