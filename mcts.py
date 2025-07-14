from typing import Dict, List, Optional, Any
import logging
import math
import time
import chess
import numpy as np
import torch
from main import get_config

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, 
                 board: chess.Board, 
                 parent: Optional['Node'] = None, 
                 move: Optional[chess.Move] = None, 
                 prior: float = None) -> None:
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior or get_config('mcts', 'move_prior')
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[chess.Move, 'Node'] = {}
        self.is_expanded: bool = False

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def get_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def get_ucb_score(self, c_puct: float = None) -> float:
        c_puct = c_puct or get_config('mcts', 'c_puct')
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            raise ValueError("Root node should not be scored with UCB")
        exploration = (c_puct * self.prior * 
                      math.sqrt(self.parent.visits) / (1 + self.visits))
        ucb_score = self.get_value() + exploration
        if logger.isEnabledFor(logging.DEBUG) and self.move:
            logger.debug(f"UCB score for {self.move}: {ucb_score:.4f} "
                        f"(value: {self.get_value():.4f}, exploration: {exploration:.4f}, "
                        f"visits: {self.visits}, prior: {self.prior:.4f})")
        return ucb_score

    def select_child(self, c_puct: float = None) -> Optional['Node']:
        c_puct = c_puct or get_config('mcts', 'c_puct')
        if not self.children:
            return None
        best_move = None
        best_score = float('-inf')
        for move, child in self.children.items():
            try:
                score = child.get_ucb_score(c_puct)
                if score > best_score:
                    best_score = score
                    best_move = move
            except ValueError as e:
                logger.warning(f"Error calculating UCB for move {move}: {e}")
                continue
        return self.children[best_move] if best_move is not None else None

    def expand(self, model: Any, move_encoder: Any, device: str) -> Optional[float]:
        if self.is_expanded or self.is_terminal():
            logger.debug(f"Node already expanded or terminal (expanded: {self.is_expanded}, terminal: {self.is_terminal()})")
            return None
        try:
            if logger.isEnabledFor(logging.DEBUG):
                expand_start = time.time()
                logger.debug(f"Expanding node at position: {self.board.fen()[:20]}...")
            board_tensor = model.encode_board(self.board).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(board_tensor)
            policy = output['policy'][0]
            value = output['value'][0].item()
            legal_moves = list(self.board.legal_moves)
            logger.debug(f"Found {len(legal_moves)} legal moves for expansion")
            priors = []
            for move in legal_moves:
                move_idx = move_encoder.encode_move(move)
                prior = (policy[move_idx].item() 
                        if move_idx < len(policy) 
                        else get_config('mcts', 'move_prior'))
                priors.append(prior)
                child_board = self.board.copy()
                child_board.push(move)
                child = Node(child_board, parent=self, move=move, prior=prior)
                self.children[move] = child
            self.is_expanded = True
            if logger.isEnabledFor(logging.DEBUG):
                expand_time = time.time() - expand_start
                avg_prior = sum(priors) / len(priors) if priors else 0
                max_prior = max(priors) if priors else 0
                logger.debug(f"Node expansion completed in {expand_time * 1000:.2f}ms")
                logger.debug(f"Children: {len(self.children)}, Value: {value:.4f}")
                logger.debug(f"Prior stats - avg: {avg_prior:.4f}, max: {max_prior:.4f}")
            return value
        except Exception as e:
            logger.error(f"Error expanding node: {e}")
            logger.debug(f"Board state during error: {self.board.fen()}")
            return None

    def backup(self, value: float) -> None:
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)

    def get_visit_counts(self) -> Dict[chess.Move, int]:
        return {move: child.visits for move, child in self.children.items()}


class MCTS:
    def __init__(self, 
                 model: Any, 
                 move_encoder: Any, 
                 device: str, 
                 num_simulations: int = None, 
                 c_puct: float = None) -> None:
        num_simulations = num_simulations or get_config('mcts', 'simulations')
        c_puct = c_puct or get_config('mcts', 'c_puct')
        if num_simulations <= 0:
            raise ValueError(f"num_simulations must be positive, got {num_simulations}")
        self.model = model
        self.move_encoder = move_encoder
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, board: chess.Board) -> Dict[chess.Move, float]:
        if board.is_game_over():
            logger.warning("MCTS search on terminal position")
            logger.debug(f"Terminal position: {board.result()}")
            return {}
        if logger.isEnabledFor(logging.DEBUG):
            search_start = time.time()
            logger.debug(f"Starting MCTS search: {self.num_simulations} simulations")
            logger.debug(f"Position: {board.fen()[:30]}...")
        root = Node(board)
        completed = 0
        failed = 0
        for i in range(self.num_simulations):
            try:
                if logger.isEnabledFor(logging.DEBUG) and (i + 1) % max(1, self.num_simulations // 4) == 0:
                    logger.debug(f"MCTS progress: {i + 1}/{self.num_simulations} simulations")
                self._simulate(root)
                completed += 1
            except Exception as e:
                failed += 1
                if completed == 0 and failed <= 3:  
                    logger.warning(f"MCTS simulation {i + 1} failed: {e}")
                elif failed == 1:
                    logger.debug(f"First simulation failure at step {i + 1}: {e}")
                continue
        if completed == 0:
            logger.error("All MCTS simulations failed")
            return {}
        if failed > 0:
            failure_rate = failed / (completed + failed) * 100
            if failure_rate > 20:
                logger.warning(f"High MCTS failure rate: {failure_rate:.1f}% ({failed}/{failed + completed})")
            else:
                logger.debug(f"MCTS failures: {failed}/{failed + completed} ({failure_rate:.1f}%)")
        visit_counts = root.get_visit_counts()
        if not visit_counts:
            logger.warning("No move visits recorded")
            return {}
        total_visits = sum(visit_counts.values())
        move_probs = {move: count / total_visits 
                     for move, count in visit_counts.items()}
        if completed < self.num_simulations:
            completion_rate = completed / self.num_simulations * 100
            if completion_rate < 50:
                logger.warning(f"Low MCTS completion: {completed}/{self.num_simulations} ({completion_rate:.1f}%)")
            else:
                logger.debug(f"MCTS completion: {completed}/{self.num_simulations} ({completion_rate:.1f}%)")
        if logger.isEnabledFor(logging.DEBUG):
            search_time = time.time() - search_start
            sims_per_sec = completed / search_time if search_time > 0 else 0
            logger.debug(f"MCTS search completed in {search_time * 1000:.1f}ms ({sims_per_sec:.1f} sims/sec)")
            sorted_moves = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
            top_moves = sorted_moves[:3]
            move_info = [f"{move} ({visits} visits, {move_probs[move]:.3f})" for move, visits in top_moves]
            logger.debug(f"Top moves: {', '.join(move_info)}")
        return move_probs

    def _simulate(self, root: Node) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            sim_start = time.time()
        node = root
        path = [node]
        while node.is_expanded and not node.is_terminal():
            child = node.select_child(self.c_puct)
            if child is None:
                logger.warning("No child selected in MCTS (possible error in UCB calculation)")
                logger.debug(f"Node children count: {len(node.children)}, expanded: {node.is_expanded}")
                break
            node = child
            path.append(node)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"MCTS path length: {len(path)} nodes")
        if node.is_terminal():
            value = self._evaluate_terminal_node(node)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Terminal node reached: value={value}, result={node.board.result()}")
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Expanding leaf node...")
            value = node.expand(self.model, self.move_encoder, self.device)
            if value is None:
                logger.warning("Node expansion failed, using 0.0")
                value = 0.0
        if logger.isEnabledFor(logging.DEBUG):
            backup_values = []
        for i, path_node in enumerate(reversed(path)):
            path_node.backup(value)
            if logger.isEnabledFor(logging.DEBUG):
                backup_values.append((path_node.move, value))
            value = -value
        if logger.isEnabledFor(logging.DEBUG):
            sim_time = time.time() - sim_start
            if sim_time > 0.1:  
                logger.debug(f"Slow MCTS simulation: {sim_time * 1000:.1f}ms")
                logger.debug(f"Backup values: {backup_values[:3]}...")

    def _evaluate_terminal_node(self, node: Node) -> float:
        result = node.board.result()
        if result == get_config('game', 'chess_white_win'):
            value = get_config('game', 'win_value')
        elif result == get_config('game', 'chess_black_win'):
            value = get_config('game', 'loss_value')
        else:
            value = get_config('game', 'draw_value')
        if not node.board.turn:
            value = -value
        if logger.isEnabledFor(logging.DEBUG):
            turn_str = "White" if node.board.turn else "Black"
            logger.debug(f"Terminal evaluation: {result} from {turn_str} perspective = {value}")
        return value

    def get_best_move(self, 
                     board: chess.Board, 
                     temperature: float = 1.0) -> Optional[chess.Move]:
        if temperature < 0:
            raise ValueError(f"Temperature must be non-negative, got {temperature}")
        logger.debug(f"Getting best move with temperature {temperature}")
        move_probs = self.search(board)
        if not move_probs:
            logger.warning("No moves available from MCTS search")
            return None
        if temperature == 0:
            best_move = max(move_probs, key=move_probs.get)
            best_prob = move_probs[best_move]
            logger.debug(f"Deterministic selection: {best_move} (prob: {best_prob:.3f})")
            return best_move
        else:
            moves = list(move_probs.keys())
            probs = np.array(list(move_probs.values()), dtype=np.float64)
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)
            selected_move = np.random.choice(moves, p=probs)
            if logger.isEnabledFor(logging.DEBUG):
                sorted_indices = np.argsort(probs)[::-1][:3]
                candidates = [f"{moves[i]} ({probs[i]:.3f})" for i in sorted_indices]
                logger.debug(f"Move candidates: {', '.join(candidates)}")
                logger.debug(f"Selected: {selected_move}")
            return selected_move
