import math
import numpy as np
import chess
import torch
from src.utils.common_utils import get_game_result
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
            if mv not in self.children and prob > 0.0:
                next_board = self.board.copy()
                next_board.push(mv)
                self.children[mv] = TreeNode(self, prob, next_board, mv)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda item: item[1].get_value(c_puct))

    def update(self, leaf_value):
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return len(self.children) == 0

    def get_value(self, c_puct):
        if self.parent:
            parent_visits = self.parent.n_visits
            self.u = c_puct * self.P * math.sqrt(parent_visits) / (1 + self.n_visits)
        return self.Q + self.u

class MCTS:
    def __init__(self, model, device, c_puct=1.4, n_simulations=800):
        self.root = None
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def _policy_value_fn(self, board: chess.Board):
        board_tensor = convert_board_to_tensor(board)
        board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value_out = self.model(board_tensor)
            policy_logits = policy_logits[0]
            value_float = value_out.item()

        policy = torch.softmax(policy_logits, dim=0).cpu().numpy()
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return {}, value_float

        action_probs = {}
        total_prob = 0.0
        move_mapping = get_move_mapping()

        for mv in legal_moves:
            idx = move_mapping.get_index_by_move(mv)
            if idx is not None and idx < len(policy):
                prob = max(policy[idx], 1e-8)
                action_probs[mv] = prob
                total_prob += prob
            else:
                action_probs[mv] = 1e-8
                total_prob += 1e-8

        if total_prob > 0:
            for mv in action_probs:
                action_probs[mv] /= total_prob
        else:
            uniform = 1.0 / len(legal_moves)
            for mv in action_probs:
                action_probs[mv] = uniform

        return action_probs, value_float

    def set_root_node(self, board: chess.Board):
        self.root = TreeNode(None, 1.0, board.copy(), None)
        action_probs, _ = self._policy_value_fn(board)
        self.root.expand(action_probs)

    def simulate(self):
        node = self.root

        # Selection
        while not node.is_leaf():
            _, node = node.select(self.c_puct)

        # Expansion & Evaluation
        action_probs, leaf_value = self._policy_value_fn(node.board)
        if not node.board.is_game_over():
            node.expand(action_probs)
        else:
            leaf_value = get_game_result(node.board)

        # Backpropagation
        node.update_recursive(-leaf_value)

    def get_move_probs(self, temperature=1e-3):
        for _ in range(self.n_simulations):
            self.simulate()

        if not self.root.children:
            return {}

        move_visits = [(mv, child.n_visits) for mv, child in self.root.children.items()]
        moves, visits = zip(*move_visits)
        visits = np.array(visits, dtype=np.float32)

        if temperature <= 1e-3:
            # Deterministic: pick the move with highest visit count
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # Softmax over visit counts
            visits_exp = np.exp((visits - np.max(visits)) / temperature)
            probs = visits_exp / visits_exp.sum()

        return dict(zip(moves, probs))

    def update_with_move(self, last_move: chess.Move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            new_board = self.root.board.copy()
            new_board.push(last_move)
            self.set_root_node(new_board)