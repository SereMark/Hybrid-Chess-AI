import math
import numpy as np
import threading
from src.utils.common_utils import get_game_result

class TreeNode:
    def __init__(self, parent, prior_p, board, move):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0  # Mean value of the node
        self.u = 0  # Upper confidence bound
        self.P = prior_p  # Prior probability of selecting this node
        self.board = board
        self.move = move

    def expand(self, action_priors):
        for move, prob in action_priors.items():
            if move not in self.children and prob > 0:
                next_board = self.board.copy()
                next_board.push(move)
                self.children[move] = TreeNode(self, prob, next_board, move)

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
            self.u = c_puct * self.P * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.u

class MCTS:
    def __init__(self, policy_value_fn, c_puct=1.4, n_simulations=800):
        self.root = None
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.tree_lock = threading.Lock()

    def set_root_node(self, board):
        self.root = TreeNode(None, 1.0, board.copy(), None)
        action_probs, _ = self.policy_value_fn(board)
        self.root.expand(action_probs)

    def simulate(self):
        node = self.root
        while not node.is_leaf():
            _, node = node.select(self.c_puct)

        action_probs, leaf_value = self.policy_value_fn(node.board)
        if not node.board.is_game_over():
            node.expand(action_probs)
        else:
            leaf_value = get_game_result(node.board)

        node.update_recursive(-leaf_value)

    def get_move_probs(self, temperature=1e-3):
        for _ in range(self.n_simulations):
            self.simulate()

        move_visits = [(move, child.n_visits) for move, child in self.root.children.items()]
        if not move_visits:
            return {}

        moves, visits = zip(*move_visits)
        visits = np.array(visits, dtype=np.float32)

        if temperature <= 1e-3:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            visits_exp = np.exp((visits - np.max(visits)) / temperature)
            probs = visits_exp / np.sum(visits_exp)

        return dict(zip(moves, probs))

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            new_board = self.root.board.copy()
            new_board.push(last_move)
            self.set_root_node(new_board)

    def get_tree_data(self, max_depth=3):
        nodes, edges = [], []
        visited = set()

        def recurse(node, depth=0):
            if depth > max_depth or id(node) in visited:
                return
            visited.add(id(node))
            nodes.append({
                'id': id(node),
                'Q': node.Q,
                'n_visits': node.n_visits,
                'move': str(node.move),
                'parent_id': id(node.parent) if node.parent else None,
            })
            for child in node.children.values():
                edges.append({'from': id(node), 'to': id(child)})
                recurse(child, depth + 1)

        with self.tree_lock:
            if self.root:
                recurse(self.root)
        return nodes, edges