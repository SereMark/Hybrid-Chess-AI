from typing import Dict, List, Optional, Tuple, Any, Set
import logging
import math
import time
import chess
import numpy as np
import torch
from collections import deque
from mcts import Node
from main import get_config

logger = logging.getLogger(__name__)


class BatchNode(Node):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.virtual_loss = 0
        self.pending_value = None
        
    def apply_virtual_loss(self, virtual_loss_value: float = 1.0) -> None:
        self.virtual_loss += virtual_loss_value
        self.visits += 1
        self.value_sum -= virtual_loss_value
        
    def revert_virtual_loss(self, virtual_loss_value: float = 1.0) -> None:
        self.virtual_loss -= virtual_loss_value
        self.visits -= 1
        self.value_sum += virtual_loss_value
        
    def backup_with_virtual_loss(self, value: float) -> None:
        if self.virtual_loss > 0:
            self.revert_virtual_loss()
        self.backup(value)


class BatchMCTS:
    
    def __init__(self, 
                 model: Any, 
                 move_encoder: Any, 
                 device: str,
                 batch_size: int = 16,
                 num_simulations: int = None,
                 c_puct: float = None,
                 virtual_loss_value: float = 1.0) -> None:
        num_simulations = num_simulations or get_config('mcts', 'simulations')
        c_puct = c_puct or get_config('mcts', 'c_puct')
        
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > 256:
            logger.warning(f"Large batch_size={batch_size} may cause memory issues")
        if num_simulations <= 0:
            raise ValueError(f"num_simulations must be positive, got {num_simulations}")
        if c_puct <= 0:
            raise ValueError(f"c_puct must be positive, got {c_puct}")
        if virtual_loss_value <= 0:
            raise ValueError(f"virtual_loss_value must be positive, got {virtual_loss_value}")
        
        self.model = model
        self.move_encoder = move_encoder
        self.device = device
        self.batch_size = batch_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.virtual_loss_value = virtual_loss_value
        
        self._init_tensor_pools()
        
        logger.info(f"BatchMCTS initialized: batch_size={batch_size}, simulations={num_simulations}")
        
    def _init_tensor_pools(self) -> None:
        board_size = get_config('model', 'board_size')
        piece_types = get_config('model', 'piece_types')
        colors = get_config('model', 'colors')
        self.board_encoding_size = board_size * board_size * piece_types * colors
        
        self.batch_tensor_pool = {}
        for size in [4, 8, 16, 32]:
            self.batch_tensor_pool[size] = torch.zeros(
                size, self.board_encoding_size, 
                dtype=torch.float32, device=self.device
            )
            
    def search_batch(self, boards: List[chess.Board]) -> List[Dict[chess.Move, float]]:
        if not boards:
            return []
            
        logger.debug(f"Batch search starting for {len(boards)} boards")
        
        roots = [BatchNode(board) for board in boards]
        
        active_roots = list(range(len(roots)))
        
        eval_queue = deque()
        
        for sim in range(self.num_simulations):
            if logger.isEnabledFor(logging.DEBUG) and (sim + 1) % max(1, self.num_simulations // 4) == 0:
                logger.debug(f"Batch MCTS progress: {sim + 1}/{self.num_simulations}")
                
            for root_idx in active_roots:
                root = roots[root_idx]
                if not root.is_terminal():
                    leaf_node = self._select_leaf(root)
                    if leaf_node and not leaf_node.is_terminal():
                        eval_queue.append((root_idx, leaf_node))
                        leaf_node.apply_virtual_loss(self.virtual_loss_value)
            
            if len(eval_queue) >= self.batch_size or (sim == self.num_simulations - 1 and eval_queue):
                self._batch_evaluate(eval_queue)
                eval_queue.clear()
        
        results = []
        for root in roots:
            visit_counts = root.get_visit_counts()
            if visit_counts:
                total_visits = sum(visit_counts.values())
                move_probs = {move: count / total_visits 
                             for move, count in visit_counts.items()}
            else:
                move_probs = {}
            results.append(move_probs)
            
        return results
    
    def search(self, board: chess.Board) -> Dict[chess.Move, float]:
        results = self.search_batch([board])
        return results[0] if results else {}
    
    def _select_leaf(self, node: BatchNode) -> Optional[BatchNode]:
        path = []
        current = node
        
        while current.is_expanded and not current.is_terminal():
            child = self._select_best_child(current)
            if child is None:
                break
            current = child
            path.append(current)
            
        return current
    
    def _select_best_child(self, node: BatchNode) -> Optional[BatchNode]:
        if not node.children:
            return None
            
        best_child = None
        best_score = float('-inf')
        
        for move, child in node.children.items():
            if child.visits + child.virtual_loss == 0:
                score = float('inf')
            else:
                q_value = (child.value_sum - child.virtual_loss * self.virtual_loss_value) / (child.visits + child.virtual_loss)
                exploration = (self.c_puct * child.prior * 
                             math.sqrt(node.visits) / (1 + child.visits + child.virtual_loss))
                score = q_value + exploration
                
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def _batch_evaluate(self, eval_queue: deque) -> None:
        if not eval_queue:
            return
            
        batch_size = len(eval_queue)
        logger.debug(f"Batch evaluating {batch_size} positions")
        
        if batch_size in self.batch_tensor_pool:
            batch_tensor = self.batch_tensor_pool[batch_size][:batch_size]
        else:
            batch_tensor = torch.zeros(
                batch_size, self.board_encoding_size,
                dtype=torch.float32, device=self.device
            )
        
        nodes_to_expand = []
        for i, (root_idx, node) in enumerate(eval_queue):
            if node.is_terminal():
                value = self._evaluate_terminal_node(node)
                node.pending_value = value
            else:
                board_tensor = self.model.encode_board_vectorized(node.board)
                batch_tensor[i] = board_tensor
                nodes_to_expand.append((root_idx, node, i))
        
        if nodes_to_expand:
            with torch.no_grad():
                if logger.isEnabledFor(logging.DEBUG):
                    eval_start = time.time()
                    
                outputs = self.model(batch_tensor[:len(nodes_to_expand)])
                policies = outputs['policy']
                values = outputs['value']
                
                if logger.isEnabledFor(logging.DEBUG):
                    eval_time = time.time() - eval_start
                    logger.debug(f"Batch NN evaluation: {eval_time * 1000:.2f}ms for {len(nodes_to_expand)} positions")
            
            for (root_idx, node, batch_idx) in nodes_to_expand:
                policy = policies[batch_idx]
                value = values[batch_idx].item()
                
                self._expand_node(node, policy, value)
                
        for root_idx, node in eval_queue:
            if hasattr(node, 'pending_value') and node.pending_value is not None:
                value = node.pending_value
                node.pending_value = None
            else:
                value = node.value_sum / node.visits if node.visits > 0 else 0.0
                
            if node.virtual_loss > 0:
                node.revert_virtual_loss(self.virtual_loss_value)
            node.backup(value)
    
    def _expand_node(self, node: BatchNode, policy: torch.Tensor, value: float) -> None:
        if node.is_expanded or node.is_terminal():
            return
            
        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            return
            
        priors = []
        for move in legal_moves:
            move_idx = self.move_encoder.encode_move(move)
            prior = (policy[move_idx].item() 
                    if move_idx < len(policy) 
                    else get_config('mcts', 'move_prior'))
            priors.append(prior)
            
        prior_sum = sum(priors)
        if prior_sum > 0:
            priors = [p / prior_sum for p in priors]
        else:
            priors = [1.0 / len(legal_moves)] * len(legal_moves)
            
        for move, prior in zip(legal_moves, priors):
            child_board = node.board.copy()
            child_board.push(move)
            child = BatchNode(child_board, parent=node, move=move, prior=prior)
            node.children[move] = child
            
        node.is_expanded = True
        node.pending_value = value
        
    def _evaluate_terminal_node(self, node: BatchNode) -> float:
        result = node.board.result()
        if result == get_config('game', 'chess_white_win'):
            value = get_config('game', 'win_value')
        elif result == get_config('game', 'chess_black_win'):
            value = get_config('game', 'loss_value')
        else:
            value = get_config('game', 'draw_value')
            
        if not node.board.turn:
            value = -value
            
        return value
    
    def get_best_move(self, 
                     board: chess.Board, 
                     temperature: float = 1.0) -> Optional[chess.Move]:
        move_probs = self.search(board)
        if not move_probs:
            return None
            
        if temperature == 0:
            return max(move_probs, key=move_probs.get)
        else:
            moves = list(move_probs.keys())
            probs = np.array(list(move_probs.values()), dtype=np.float64)
            
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)
                
            return np.random.choice(moves, p=probs)