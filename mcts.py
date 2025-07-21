import math

import chess
import torch
from config import config


class Node:
    def __init__(
        self,
        board: chess.Board,
        parent: "Node | None" = None,
        move: chess.Move | None = None,
        prior: float = 0.001,
    ):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[chess.Move, Node] = {}
        self.is_expanded = False

    def get_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def get_ucb_score(self) -> float:
        if self.visits == 0:
            return float("inf")
        exploration = (
            config.C_PUCT
            * self.prior
            * math.sqrt(self.parent.visits)
            / (1 + self.visits)
        )
        return self.get_value() + exploration

    def select_child(self) -> "Node | None":
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.get_ucb_score())

    def backup(self, value: float) -> None:
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)

    def get_visit_counts(self) -> dict[chess.Move, int]:
        return {move: child.visits for move, child in self.children.items()}


class MCTS:
    def __init__(self, model, move_encoder, device: str):
        self.model = model
        self.move_encoder = move_encoder
        self.device = device

    def search_batch(self, boards):
        if not boards:
            return []

        roots = [Node(board) for board in boards]

        for _ in range(config.MCTS_SIMULATIONS):
            nodes_to_evaluate = []
            for root in roots:
                if not root.board.is_game_over():
                    leaf = self._select_leaf(root)
                    if leaf:
                        nodes_to_evaluate.append(leaf)
            if nodes_to_evaluate:
                self._evaluate_batch(nodes_to_evaluate)

        results = []
        for root in roots:
            visit_counts = root.get_visit_counts()
            total_visits = sum(visit_counts.values())
            if total_visits > 0:
                move_probs = {
                    move: count / total_visits for move, count in visit_counts.items()
                }
            else:
                legal_moves = list(root.board.legal_moves)
                move_probs = (
                    {move: 1.0 / len(legal_moves) for move in legal_moves}
                    if legal_moves
                    else {}
                )
            results.append(move_probs)

        return results

    def _select_leaf(self, node):
        current = node
        while current.is_expanded and not current.board.is_game_over():
            child = current.select_child()
            if child is None:
                break
            current = child
        return None if current.board.is_game_over() else current

    def _evaluate_batch(self, nodes):
        for i in range(0, len(nodes), config.MCTS_BATCH_SIZE):
            batch_nodes = nodes[i : i + config.MCTS_BATCH_SIZE]
            self._process_batch(batch_nodes)

    def _process_batch(self, nodes):
        for node in nodes:
            if node.board.is_game_over():
                value = self._get_terminal_value(node)
                node.backup(value)
            else:
                board_tensor = (
                    self.model.encode_board(node.board)
                    .unsqueeze(0)
                    .to(self.device)
                )
                with torch.no_grad():
                    outputs = self.model(board_tensor)
                    policy = outputs["policy"][0]
                    value = outputs["value"][0].item()
                self._expand_node(node, policy)
                node.backup(value)

    def _expand_node(self, node, policy):
        if node.is_expanded:
            return
        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            return
        priors = []
        for move in legal_moves:
            move_idx = self.move_encoder.encode_move(move)
            prior = (
                policy[move_idx].item() if move_idx < len(policy) else 0.001
            )
            priors.append(prior)
        prior_sum = sum(priors)
        priors = [p / prior_sum for p in priors]
        for move, prior in zip(legal_moves, priors, strict=False):
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = Node(child_board, parent=node, move=move, prior=prior)
        node.is_expanded = True

    def _get_terminal_value(self, node):
        result = node.board.result()
        return {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
