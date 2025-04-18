import math
import chess
import torch
import numpy as np
from src.utils.chess import board_to_input, get_move_map

class Node:
    def __init__(self, parent, prior_p, board, move):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.Q = 0.0
        self.u = 0.0
        self.P = prior_p
        self.board = board
        self.move = move

    def expand(self, probs):
        for mv, prob in probs.items():
            if prob > 0.0 and mv not in self.children:
                next_board = self.board.copy()
                next_board.push(mv)
                self.children[mv] = Node(self, prob, next_board, mv)

    def select(self, c_puct):
        best_move = None
        best_node = None
        best_val = -float('inf')
        for mv, node in self.children.items():
            node.u = c_puct * node.P * math.sqrt(self.visits) / (1 + node.visits)
            val = node.Q + node.u
            if val > best_val:
                best_val = val
                best_move = mv
                best_node = node
        return best_move, best_node

    def update(self, leaf_val):
        if self.parent:
            self.parent.update(-leaf_val)
        self.visits += 1
        self.Q += (leaf_val - self.Q) / self.visits

class MCTS:
    def __init__(self, model, device, c_puct=1.4, n_sims=800):
        self.root = None
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_sims = n_sims
        self.move_map = get_move_map()

    def policy_value(self, board):
        board_tensor = torch.from_numpy(board_to_input(board)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.model(board_tensor)
        probs = torch.softmax(policy[0], dim=0).cpu().numpy()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, float(value.item())
        action_probs = {}
        total_prob = 0.0
        for mv in legal_moves:
            idx = self.move_map.idx_by_move(mv)
            prob = max(probs[idx], 1e-8) if idx is not None and idx < len(probs) else 1e-8
            action_probs[mv] = prob
            total_prob += prob
        if total_prob > 1e-12:
            for mv in action_probs:
                action_probs[mv] /= total_prob
        else:
            for mv in action_probs:
                action_probs[mv] = 1.0 / len(legal_moves)
        return action_probs, float(value.item())

    def set_root(self, board):
        self.root = Node(None, 1.0, board.copy(), None)
        action_probs, _ = self.policy_value(board)
        self.root.expand(action_probs)

    def get_move_probs(self, temp=1e-3):
        for _ in range(self.n_sims):
            node = self.root
            while node.children:
                _, node = node.select(self.c_puct)
            action_probs, leaf_val = self.policy_value(node.board)
            if not node.board.is_game_over():
                node.expand(action_probs)
            else:
                result_map = {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0}
                leaf_val = result_map.get(node.board.result(), 0.0)
            node.update(-leaf_val)
        if not self.root.children:
            return {}
        move_visits = {mv: child.visits for mv, child in self.root.children.items()}
        moves, visits = zip(*move_visits.items())
        visits = np.array(visits, dtype=np.float32)
        if temp <= 1e-3:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            exps = np.exp((visits - np.max(visits)) / temp)
            probs = exps / np.sum(exps)
        return dict(zip(moves, probs))

    def update_with_move(self, move):
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            next_board = self.root.board.copy()
            next_board.push(move)
            self.set_root(next_board)