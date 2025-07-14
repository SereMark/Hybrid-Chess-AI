from typing import Dict, List, Optional, Any
import logging
import numpy as np
import torch
from collections import deque
from main import get_config
from model import MoveEncoder

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    
    def __init__(self, max_size: int = 50000, device: str = 'cuda') -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if max_size > 1000000:
            logger.warning(f"Very large buffer max_size={max_size} will use significant memory")
            
        self.max_size = max_size
        self.device = device
        
        board_size = get_config('model', 'board_size')
        piece_types = get_config('model', 'piece_types')
        colors = get_config('model', 'colors')
        board_encoding_size = board_size * board_size * piece_types * colors
        move_space_size = get_config('model', 'move_space_size')
        
        logger.info(f"Initializing experience buffer with capacity {max_size}")
        self.boards = np.zeros((max_size, board_encoding_size), dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.policies = np.zeros((max_size, move_space_size), dtype=np.float32)
        
        self.size = 0
        self.pos = 0
        
        self._move_encoder: Optional[MoveEncoder] = None
        
        memory_mb = (self.boards.nbytes + self.values.nbytes + self.policies.nbytes) / (1024 ** 2)
        logger.info(f"Experience buffer memory usage: {memory_mb:.1f}MB")
        
    def add_batch(self, games: List[Any]) -> int:
        if not games:
            return 0
            
        positions_added = 0
        
        for game in games:
            try:
                training_data = game.get_training_data()
                
                for position in training_data:
                    board_tensor = position['board_tensor']
                    value = position['value']
                    move_probs = position['move_probs']
                    
                    if isinstance(board_tensor, torch.Tensor):
                        board_flat = board_tensor.cpu().numpy().flatten()
                    else:
                        board_flat = board_tensor.flatten()
                        
                    self.boards[self.pos] = board_flat
                    self.values[self.pos] = value
                    
                    if isinstance(move_probs, dict):
                        policy_vector = np.zeros(self.policies.shape[1], dtype=np.float32)
                        
                        if self._move_encoder is None:
                            self._move_encoder = MoveEncoder()
                            
                        for move, prob in move_probs.items():
                            idx = self._move_encoder.encode_move(move)
                            if 0 <= idx < len(policy_vector):
                                policy_vector[idx] = prob
                                
                        self.policies[self.pos] = policy_vector
                    else:
                        if isinstance(move_probs, torch.Tensor):
                            self.policies[self.pos] = move_probs.cpu().numpy()
                        else:
                            self.policies[self.pos] = move_probs
                    
                    self.pos = (self.pos + 1) % self.max_size
                    self.size = min(self.size + 1, self.max_size)
                    positions_added += 1
                    
            except Exception as e:
                logger.warning(f"Failed to add game to buffer: {e}")
                continue
                
        if positions_added > 0:
            logger.debug(f"Added {positions_added} positions to buffer (total: {self.size})")
            
        return positions_added
    
    def sample(self, batch_size: int = 256) -> Optional[Dict[str, torch.Tensor]]:
        if self.size == 0:
            logger.warning("Cannot sample from empty buffer")
            return None
            
        if batch_size > self.size:
            logger.debug(f"Batch size {batch_size} > buffer size {self.size}, using full buffer")
            batch_size = self.size
            
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {
            'board_tensors': torch.from_numpy(self.boards[indices]).to(self.device),
            'target_values': torch.from_numpy(self.values[indices]).to(self.device),
            'target_policies': torch.from_numpy(self.policies[indices]).to(self.device)
        }
        
        return batch
    
    def sample_recent(self, batch_size: int = 256, recent_fraction: float = 0.5) -> Optional[Dict[str, torch.Tensor]]:
        if self.size == 0:
            return None
            
        if batch_size > self.size:
            batch_size = self.size
            
        recent_samples = int(batch_size * recent_fraction)
        random_samples = batch_size - recent_samples
        
        indices = []
        
        if recent_samples > 0:
            recent_start = max(0, self.size - self.size // 5)
            recent_indices = np.random.choice(
                range(recent_start, self.size), 
                min(recent_samples, self.size - recent_start), 
                replace=False
            )
            indices.extend(recent_indices)
            
        if random_samples > 0:
            random_indices = np.random.choice(self.size, random_samples, replace=False)
            indices.extend(random_indices)
            
        indices = np.array(indices[:batch_size])
        
        batch = {
            'board_tensors': torch.from_numpy(self.boards[indices]).to(self.device),
            'target_values': torch.from_numpy(self.values[indices]).to(self.device),
            'target_policies': torch.from_numpy(self.policies[indices]).to(self.device)
        }
        
        return batch
    
    def clear(self) -> None:
        self.size = 0
        self.pos = 0
        logger.info("Experience buffer cleared")
        
    def get_stats(self) -> Dict[str, Any]:
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.max_size,
                'utilization': 0.0,
                'avg_value': 0.0,
                'value_std': 0.0
            }
            
        values_subset = self.values[:self.size]
        
        return {
            'size': self.size,
            'capacity': self.max_size,
            'utilization': self.size / self.max_size,
            'avg_value': float(np.mean(values_subset)),
            'value_std': float(np.std(values_subset)),
            'position': self.pos
        }