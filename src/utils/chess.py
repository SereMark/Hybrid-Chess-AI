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
    
    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        i = self.indices[idx]
        return (
            torch.tensor(self.file["inputs"][i], dtype=torch.float32),
            torch.tensor(self.file["policy_targets"][i], dtype=torch.long),
            torch.tensor(self.file["value_targets"][i], dtype=torch.float32)
        )
    
    def __del__(self):
        if self.file:
            self.file.close()

class MoveMap:
    def __init__(self):
        moves = []
        for f, t in product(chess.SQUARES, repeat=2):
            if f != t:
                moves.append(chess.Move(f, t))
                if chess.square_rank(t) in (0, 7):
                    for pm in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                        moves.append(chess.Move(f, t, promotion=pm))
        self.moves = dict(enumerate(moves))
        self.indices = {v: k for k, v in self.moves.items()}
        self.count = len(moves)
    
    def move_by_idx(self, i):
        return self.moves.get(i)
    
    def idx_by_move(self, m):
        return self.indices.get(m)

_move_map = MoveMap()

def get_move_count():
    return _move_map.count

def get_move_map():
    return _move_map

def board_to_input(board):
    x = np.zeros((64, 18), np.float32)
    pieces = {
        (chess.PAWN, True): 0, (chess.KNIGHT, True): 1, (chess.BISHOP, True): 2, 
        (chess.ROOK, True): 3, (chess.QUEEN, True): 4, (chess.KING, True): 5,
        (chess.PAWN, False): 6, (chess.KNIGHT, False): 7, (chess.BISHOP, False): 8, 
        (chess.ROOK, False): 9, (chess.QUEEN, False): 10, (chess.KING, False): 11
    }
    
    for sq, piece in board.piece_map().items():
        idx = pieces.get((piece.piece_type, piece.color))
        if idx is not None:
            x[sq, idx] = 1
    
    x[:, 12] = 1 if board.turn else 0
    if board.ep_square is not None:
        x[board.ep_square, 13] = 1
    x[:, 14] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    x[:, 15] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    x[:, 16] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    x[:, 17] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    
    x = x.transpose(1, 0).reshape(18, 8, 8)
    pad = np.zeros((7, 8, 8), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)