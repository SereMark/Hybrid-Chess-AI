import math, chess, numpy as np
from src.utils.chess_utils import initialize_move_mappings

initialize_move_mappings()


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
        for move, prob in action_priors.items():
            if move not in self.children and prob > 0:
                next_board = self.board.copy()
                next_board.push(move)
                self.children[move] = TreeNode(self, prob, next_board, move)

    def select(self, c_puct):
        if not self.children:
            raise ValueError("Cannot select from a node with no children")

        max_value = float('-inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            try:
                value = child.get_value(c_puct)
                if value > max_value:
                    max_value = value
                    best_move = move
                    best_child = child
            except ValueError:
                continue

        if best_move is None:
            raise ValueError("No valid moves available")

        return best_move, best_child

    def update(self, leaf_value):
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        current = self
        while current is not None:
            current.update(leaf_value)
            leaf_value = -leaf_value
            current = current.parent

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_value(self, c_puct):
        self.u = c_puct * self.P * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.u


class MCTS:
    def __init__(self, policy_value_fn, c_puct=1.4, n_simulations=800):
        self.root = None
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.log_update = None

    def set_logger(self, log_update):
        self.log_update = log_update

    def log(self, message):
        if self.log_update:
            self.log_update.emit(str(message))

    def set_root_node(self, board: chess.Board):
        try:
            if board is None:
                raise ValueError("Board cannot be None")

            self.root = TreeNode(None, 1.0, board.copy(), None)
            self.root.n_visits = 1

            legal_moves = list(board.legal_moves)
            if legal_moves:
                action_probs, leaf_value = self.policy_value_fn(board)
                valid_probs = {}
                total_prob = 0

                for move in legal_moves:
                    prob = action_probs.get(move, 0)
                    if prob > 0:
                        valid_probs[move] = prob
                        total_prob += prob

                if total_prob <= 0:
                    prob = 1.0 / len(legal_moves)
                    valid_probs = {move: prob for move in legal_moves}
                else:
                    valid_probs = {k: v / total_prob for k, v in valid_probs.items()}

                self.root.expand(valid_probs)
                self.root.Q = -leaf_value

        except Exception as e:
            if board is not None:
                self.root = TreeNode(None, 1.0, board.copy(), None)
                self.root.n_visits = 1

    def simulate(self):
        try:
            node = self.root
            if node is None:
                raise ValueError("Root node not initialized")

            depth = 0

            while not node.is_leaf():
                if node.board.is_game_over():
                    break

                if not node.children:
                    break

                try:
                    move, node = node.select(self.c_puct)
                    depth += 1
                    if depth > 50:
                        break
                except ValueError:
                    break

            if node.board.is_game_over():
                leaf_value = self.get_game_result(node.board)
                node.update_recursive(-leaf_value)
                return

            legal_moves = list(node.board.legal_moves)
            if not legal_moves:
                if node.board.is_checkmate():
                    leaf_value = self.get_game_result(node.board)
                    node.update_recursive(-leaf_value)
                return

            action_probs, leaf_value = self.policy_value_fn(node.board)

            if not node.board.is_game_over():
                valid_probs = {}
                total_prob = 0
                for move in legal_moves:
                    prob = action_probs.get(move, 0)
                    if prob > 0:
                        valid_probs[move] = prob
                        total_prob += prob

                if total_prob <= 0:
                    prob = 1.0 / len(legal_moves)
                    valid_probs = {move: prob for move in legal_moves}
                else:
                    valid_probs = {k: v / total_prob for k, v in valid_probs.items()}

                node.expand(valid_probs)
            else:
                leaf_value = self.get_game_result(node.board)

            node.update_recursive(-leaf_value)

        except Exception as e:
            if self.log_update:
                self.log_update.emit(f"MCTS simulation error: {str(e)}")
            pass

    def get_move_probs(self, temperature=1e-3):
        try:
            legal_moves = list(self.root.board.legal_moves)
            if not legal_moves:
                return {}

            position_complexity = len(legal_moves)
            piece_count = len(self.root.board.piece_map())

            if piece_count > 24:
                n_sims = min(self.n_simulations // 2, 400)
            elif piece_count < 10:
                n_sims = self.n_simulations
            else:
                if position_complexity <= 3:
                    n_sims = min(self.n_simulations // 8, 100)
                elif position_complexity <= 10:
                    n_sims = min(self.n_simulations // 4, 200)
                elif position_complexity <= 20:
                    n_sims = min(self.n_simulations // 2, 400)
                else:
                    n_sims = min(self.n_simulations, 800)

            for _ in range(n_sims):
                self.simulate()

            moves = []
            visits = []
            for move, child in self.root.children.items():
                if move in legal_moves:
                    moves.append(move)
                    visits.append(max(child.n_visits, 1))

            if not moves:
                return {move: 1.0 / len(legal_moves) for move in legal_moves}

            visits = np.array(visits, dtype=np.float32)

            if temperature == 0 or len(legal_moves) <= 2:
                probs = np.zeros_like(visits)
                best_idx = np.argmax(visits)
                probs[best_idx] = 1.0
            else:
                visits = np.maximum(visits, 1)
                if temperature < 0.01:
                    visits = visits ** 20
                else:
                    visits = visits ** (1.0 / temperature)
                probs = visits / np.sum(visits)

            move_probs = {move: float(prob) for move, prob in zip(moves, probs)}

            for move in legal_moves:
                if move not in move_probs:
                    move_probs[move] = 1e-8

            total_prob = sum(move_probs.values())
            move_probs = {k: v / total_prob for k, v in move_probs.items()}

            return move_probs

        except Exception as e:
            if self.log_update:
                self.log_update.emit(f"MCTS get_move_probs error: {str(e)}")
            return {move: 1.0 / len(legal_moves) for move in legal_moves}

    def get_game_result(self, board):
        if board.is_checkmate():
            return 1.0 if not board.turn else -1.0
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        elif board.is_fifty_moves():
            return 0.0
        elif board.is_repetition(3):
            return 0.0
        else:
            return 0.0

    def update_with_move(self, last_move):
        try:
            if self.root and last_move in self.root.children:
                old_root = self.root
                self.root = self.root.children[last_move]
                self.root.parent = None
                self.root.n_visits = max(self.root.n_visits, 1)

                old_root.children.clear()
                del old_root
            else:
                if not self.root:
                    board = chess.Board()
                else:
                    board = self.root.board.copy()

                board.push(last_move)
                self.root = TreeNode(None, 1.0, board, None)
                self.root.n_visits = 1

            legal_moves = list(self.root.board.legal_moves)
            if legal_moves:
                action_probs, leaf_value = self.policy_value_fn(self.root.board)
                valid_probs = {}
                total_prob = 0

                for move in legal_moves:
                    prob = action_probs.get(move, 0)
                    if prob > 0:
                        valid_probs[move] = prob
                        total_prob += prob

                if total_prob <= 0:
                    prob = 1.0 / len(legal_moves)
                    valid_probs = {move: prob for move in legal_moves}
                else:
                    valid_probs = {k: v / total_prob for k, v in valid_probs.items()}

                self.root.expand(valid_probs)
                self.root.Q = -leaf_value

        except Exception as e:
            if self.log_update:
                self.log_update.emit(f"MCTS update_with_move error: {str(e)}")
            board = chess.Board()
            self.root = TreeNode(None, 1.0, board, None)
            self.root.n_visits = 1