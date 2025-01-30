import math
import chess
import torch
import numpy as np

from src.utils.chess_utils import convert_board_to_tensor, get_move_mapping

class TreeNode:
    def __init__(self, parent, prior_p, board, move):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0.0
        self.u = 0.0
        self.P = prior_p
        self.board = board
        self.move = move

    def expand(self, action_priors):
        for mv, prob in action_priors.items():
            if prob > 0.0 and mv not in self.children:
                next_board = self.board.copy()
                next_board.push(mv)
                self.children[mv] = TreeNode(self, prob, next_board, mv)

    def select(self, c_puct):
        best_move = None
        best_node = None
        best_value = float('-inf')
        for mv, node in self.children.items():
            if node.parent:
                node.u = (c_puct * node.P * math.sqrt(node.parent.n_visits) / (1 + node.n_visits))
                value = node.Q + node.u
            else:
                value = node.Q
            if value > best_value:
                best_value = value
                best_move = mv
                best_node = node
        return best_move, best_node

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

class MCTS:
    def __init__(self, model, device, c_puct=1.4, n_simulations=800):
        self.root = None
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.move_mapping = get_move_mapping()

    def _policy_value_fn(self, board: chess.Board):
        board_tensor = torch.from_numpy(convert_board_to_tensor(board)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value_out = self.model(board_tensor)
        policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, float(value_out.item())

        action_probs = {}
        total_prob = 0.0
        for mv in legal_moves:
            idx = self.move_mapping.get_index_by_move(mv)
            if idx is not None and idx < len(policy):
                prob = max(policy[idx], 1e-8)
            else:
                prob = 1e-8
            action_probs[mv] = prob
            total_prob += prob

        if total_prob > 1e-12:
            for mv in action_probs:
                action_probs[mv] /= total_prob
        else:
            for mv in action_probs:
                action_probs[mv] = 1.0 / len(legal_moves)

        return action_probs, float(value_out.item())

    def set_root_node(self, board: chess.Board):
        self.root = TreeNode(parent=None, prior_p=1.0, board=board.copy(), move=None)
        action_probs, _ = self._policy_value_fn(board)
        self.root.expand(action_probs)

    def get_move_probs(self, temperature=1e-3):
        for _ in range(self.n_simulations):
            node = self.root
            while node.children:
                _, node = node.select(self.c_puct)
            action_probs, leaf_value = self._policy_value_fn(node.board)
            if not node.board.is_game_over():
                node.expand(action_probs)
            else:
                result_map = {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0}
                leaf_value = result_map.get(node.board.result(), 0.0)
            node.update_recursive(-leaf_value)

        if not self.root.children:
            return {}
        move_visits = {mv: child.n_visits for mv, child in self.root.children.items()}
        moves, visits = zip(*move_visits.items())
        visits = np.array(visits, dtype=np.float32)

        if temperature <= 1e-3:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            exps = np.exp((visits - np.max(visits)) / temperature)
            probs = exps / np.sum(exps)

        return dict(zip(moves, probs))

    def update_with_move(self, last_move: chess.Move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            new_board = self.root.board.copy()
            new_board.push(last_move)
            self.set_root_node(new_board)