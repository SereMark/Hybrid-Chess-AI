import math, numpy as np, threading
from src.utils.chess_utils import get_game_result

class TreeNode:
    def __init__(self, parent, prior_p, board, move):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.P = prior_p
        self.board = board
        self.move = move

    def expand(self, action_priors):
        for mv, prob in action_priors.items():
            if mv not in self.children and prob > 0:
                nxt_board = self.board.copy()
                nxt_board.push(mv)
                self.children[mv] = TreeNode(self, prob, nxt_board, mv)

    def select(self, c_puct):
        max_value = float('-inf')
        best_move = None
        best_child = None
        for mv, child in self.children.items():
            val = child.get_value(c_puct)
            if val > max_value:
                max_value = val
                best_move = mv
                best_child = child
        return best_move, best_child

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
        while True:
            if node.is_leaf():
                break
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
        move_visits = [(mv, child.n_visits) for mv, child in self.root.children.items()]
        if not move_visits:
            return {}

        moves, visits = zip(*move_visits)
        visits = np.array(visits, dtype=np.float32)
        if temperature <= 1e-3:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            exponent = visits / temperature
            exponent -= np.max(exponent)
            visits_exp = np.exp(exponent)
            total_sum = np.sum(visits_exp)
            if total_sum > 0:
                probs = visits_exp / total_sum
            else:
                probs = np.ones_like(visits) / len(visits)
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
        nodes = []
        edges = []
        visited = set()

        def recurse(nd, depth=0):
            if depth > max_depth:
                return
            nid = id(nd)
            if nid in visited:
                return
            visited.add(nid)
            nodes.append((nid, {
                'Q': nd.Q,
                'n_visits': nd.n_visits,
                'move': str(nd.move),
                'parent': id(nd.parent) if nd.parent else None
            }))
            for c in nd.children.values():
                edges.append((nid, id(c)))
                recurse(c, depth + 1)

        with self.tree_lock:
            if self.root:
                recurse(self.root)
        return nodes, edges