import h5py
import torch
import chess
import numpy as np
from itertools import product
from collections import deque
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

def calculate_material_score(board):
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    
    white_material = sum(values[piece.piece_type] for piece in board.pieces(color=chess.WHITE))
    black_material = sum(values[piece.piece_type] for piece in board.pieces(color=chess.BLACK))
    
    return white_material - black_material

def calculate_phase(board):
    piece_count = len(board.piece_map())
    if piece_count >= 28:
        return 0
    elif 28 > piece_count >= 14:
        return 1
    else:
        return 2

def encode_single_board(board):
    x = np.zeros((64, 23), np.float32)
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
    
    x[:, 18] = 1 if board.is_check() else 0
    
    x[:, 19] = board.halfmove_clock / 100.0
    x[:, 20] = board.fullmove_number / 100.0
    
    material_score = calculate_material_score(board)
    x[:, 21] = np.tanh(material_score / 15.0)
    
    phase = calculate_phase(board)
    x[:, 22] = phase / 2.0
    
    return x.transpose(1, 0).reshape(23, 8, 8)

class BoardHistory:
    def __init__(self, max_history=7):
        self.history = deque(maxlen=max_history + 1)
    
    def add_board(self, board):
        self.history.append(board.copy(stack=False))
        
    def clear(self):
        self.history.clear()
        
    def get_history(self, include_current=True):
        if not self.history:
            return []
        return list(self.history) if include_current else list(self.history)[:-1]
    
    def __len__(self):
        return len(self.history)

def board_to_input(board, history=None):
    current_board_planes = encode_single_board(board)
    
    if history is not None and len(history) > 1:
        past_boards = history.get_history(include_current=False)
        history_planes = []
        
        for past_board in past_boards[-7:]:
            history_planes.append(encode_single_board(past_board))
            
        while len(history_planes) < 7:
            history_planes.append(np.zeros_like(current_board_planes))
            
        return np.concatenate([current_board_planes] + history_planes, axis=0)
    else:
        padding = np.zeros((7 * 23, 8, 8), dtype=np.float32)
        return np.concatenate([current_board_planes, padding], axis=0)