
import numpy as np
import chessai
from config import Config


class PositionEncoder:
    
    def __init__(self):
        self.planes = Config.INPUT_PLANES
        self.history_length = Config.HISTORY_LENGTH
        self.planes_per_position = Config.PLANES_PER_POSITION
        
    def encode_single(self, position, history=None):
        tensor = np.zeros((Config.INPUT_PLANES, 8, 8), dtype=np.float32)
        
        if history is not None:
            positions = [position] + history[:7]
        else:
            positions = [position]
            
        while len(positions) < self.history_length:
            positions.append(positions[-1] if positions else position)
            
        for t in range(self.history_length):
            pos = positions[t] if t < len(positions) else positions[-1]
            base_plane = t * self.planes_per_position
            
            for piece in range(6):
                for color in range(2):
                    plane_idx = base_plane + piece * 2 + color
                    bb = pos.pieces[piece][color]
                    
                    while bb:
                        square = chessai.lsb(bb)
                        bb &= bb - 1
                        row, col = square // 8, square % 8
                        tensor[plane_idx, row, col] = 1.0
            
            if t < len(positions):
                repetitions = self._count_repetitions(pos, positions[:t+1])
                if repetitions >= 1:
                    tensor[base_plane + 12] = 1.0
                if repetitions >= 2:
                    tensor[base_plane + 13] = 1.0
        
        if position.turn == chessai.WHITE:
            tensor[112] = 1.0
        
        tensor[113] = min(position.fullmove / 100.0, 1.0)
        
        castling_rights = [position.castling & 1, position.castling & 2, 
                          position.castling & 4, position.castling & 8]
        for i, has_right in enumerate(castling_rights):
            if has_right:
                tensor[114 + i] = 1.0
        
        tensor[118] = min(position.halfmove / 100.0, 1.0)
        
        return tensor
    
    def _count_repetitions(self, target_pos, history):
        target_hash = target_pos.hash
        count = 0
        for pos in history:
            if pos.hash == target_hash:
                count += 1
        return count - 1
    
    def encode_batch(self, positions, histories=None):
        if not positions:
            return np.zeros((0, self.planes, 8, 8), dtype=np.float32)
        
        batch_size = len(positions)
        batch_tensor = np.zeros((batch_size, self.planes, 8, 8), dtype=np.float32)
        
        for i, position in enumerate(positions):
            history = histories[i] if histories else None
            batch_tensor[i] = self.encode_single(position, history)
            
        return batch_tensor