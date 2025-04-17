import h5py, torch, chess, numpy as np
from itertools import product
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self,path,indices): self.path,self.indices,self.file=path,indices,None
    def __len__(self): return len(self.indices)
    def __getitem__(self,x):
        if self.file is None: self.file=h5py.File(self.path,"r")
        return tuple(torch.tensor(self.file[k][self.indices[x]],dtype=t) for k,t in zip(["inputs","policy_targets","value_targets"],[torch.float32,torch.long,torch.float32]))
    def __del__(self):
        if self.file: self.file.close()

class MoveMapping:
    def __init__(self):
        moves=[]
        for f,t in product(chess.SQUARES,repeat=2):
            if f!=t:
                moves.append(chess.Move(f,t))
                if chess.square_rank(t)in(0,7):
                    for pm in(chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN):
                        moves.append(chess.Move(f,t,promotion=pm))
        self.MOVE_MAPPING=dict(enumerate(moves))
        self.INDEX_MAPPING={v:k for k,v in self.MOVE_MAPPING.items()}
        self.TOTAL_MOVES=len(moves)
    def get_move_by_index(self,i):
        return self.MOVE_MAPPING.get(i)
    def get_index_by_move(self,m):
        return self.INDEX_MAPPING.get(m)

move_mapping=MoveMapping()

def get_total_moves(): return move_mapping.TOTAL_MOVES
def get_move_mapping(): return move_mapping

def convert_board_to_input(b):
    x = np.zeros((64, 18), np.float32)
    pm = {
        (chess.PAWN, True): 0, (chess.KNIGHT, True): 1, (chess.BISHOP, True): 2, 
        (chess.ROOK, True): 3, (chess.QUEEN, True): 4, (chess.KING, True): 5,
        (chess.PAWN, False): 6, (chess.KNIGHT, False): 7, (chess.BISHOP, False): 8, 
        (chess.ROOK, False): 9, (chess.QUEEN, False): 10, (chess.KING, False): 11
    }
    for sq, piece in b.piece_map().items():
        i = pm.get((piece.piece_type, piece.color))
        if i is not None: x[sq, i] = 1
    if b.turn: x[:, 12] = 1
    ep = b.ep_square
    if ep is not None: x[ep, 13] = 1
    if b.has_kingside_castling_rights(chess.WHITE): x[:, 14] = 1
    if b.has_queenside_castling_rights(chess.WHITE): x[:, 15] = 1
    if b.has_kingside_castling_rights(chess.BLACK): x[:, 16] = 1
    if b.has_queenside_castling_rights(chess.BLACK): x[:, 17] = 1
    x = x.transpose(1, 0).reshape(18, 8, 8)
    pad = np.zeros((7, 8, 8), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)