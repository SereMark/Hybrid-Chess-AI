import math
from typing import Any

import chess
import torch
from config import get_config


class Node:
    def __init__(
        self,
        board: chess.Board,
        parent: "Node | None" = None,
        move: chess.Move | None = None,
        prior: float | None = None,
    ) -> None:
        self.board: chess.Board = board.copy()
        self.parent: Node | None = parent
        self.move: chess.Move | None = move
        self.prior: float = prior or get_config("mcts", "move_prior")
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: dict[chess.Move, Node] = {}
        self.is_expanded: bool = False

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def get_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def get_ucb_score(self, c_puct: float | None = None) -> float:
        c_puct = c_puct or get_config("mcts", "c_puct")
        if c_puct is None:
            raise ValueError("c_puct must be provided or configured")
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            raise ValueError("Root node should not be scored with UCB")
        exploration = (
            c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        )
        ucb_score = self.get_value() + exploration
        return ucb_score

    def select_child(self, c_puct: float | None = None) -> "Node | None":
        c_puct = c_puct or get_config("mcts", "c_puct")
        if not self.children:
            return None
        best_move = None
        best_score = float("-inf")
        for move, child in self.children.items():
            try:
                score = child.get_ucb_score(c_puct)
                if score > best_score:
                    best_score = score
                    best_move = move
            except ValueError:
                continue
        return self.children[best_move] if best_move is not None else None

    def expand(self, model: Any, move_encoder: Any, device: str) -> float | None:
        if self.is_expanded or self.is_terminal():
            return None
        try:
            board_tensor = (
                model.encode_board_vectorized(self.board).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                output = model(board_tensor)
            policy = output["policy"][0]
            value = output["value"][0].item()
            legal_moves = list(self.board.legal_moves)
            priors = []
            for move in legal_moves:
                move_idx = move_encoder.encode_move(move)
                prior = (
                    policy[move_idx].item()
                    if move_idx < len(policy)
                    else get_config("mcts", "move_prior")
                )
                priors.append(prior)
                child_board = self.board.copy()
                child_board.push(move)
                child = Node(child_board, parent=self, move=move, prior=prior)
                self.children[move] = child
            self.is_expanded = True
            return value
        except Exception:
            return None

    def backup(self, value: float) -> None:
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)

    def get_visit_counts(self) -> dict[chess.Move, int]:
        return {move: child.visits for move, child in self.children.items()}


class BatchMCTS:
    def __init__(
        self,
        model,
        move_encoder,
        device: str,
        batch_size: int = 16,
        num_simulations=None,
        c_puct=None,
    ):
        self.model = model
        self.move_encoder = move_encoder
        self.device = device
        self.batch_size = batch_size
        self.num_simulations = (
            num_simulations or get_config("mcts", "simulations") or 25
        )
        self.c_puct = c_puct or get_config("mcts", "c_puct") or 1.0

    def search_batch(self, boards):
        if not boards:
            return []

        roots = [Node(board) for board in boards]

        for _ in range(self.num_simulations):
            nodes_to_evaluate = []

            for root in roots:
                if not root.is_terminal():
                    leaf = self._select_leaf(root)
                    if leaf and not leaf.is_terminal():
                        nodes_to_evaluate.append(leaf)

            if nodes_to_evaluate:
                self._evaluate_nodes(nodes_to_evaluate)

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
                if legal_moves:
                    move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
                else:
                    move_probs = {}

            results.append(move_probs)

        return results

    def search(self, board):
        results = self.search_batch([board])
        return results[0] if results else {}

    def _select_leaf(self, node):
        current = node

        while current.is_expanded and not current.is_terminal():
            child = self._select_best_child(current)
            if child is None:
                break
            current = child

        return current

    def _select_best_child(self, node):
        if not node.children:
            return None

        best_child = None
        best_score = float("-inf")

        for child in node.children.values():
            if child.visits == 0:
                score = float("inf")
            else:
                q_value = child.get_value()
                exploration = (
                    self.c_puct
                    * child.prior
                    * math.sqrt(node.visits)
                    / (1 + child.visits)
                )
                score = q_value + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _evaluate_nodes(self, nodes):
        for i in range(0, len(nodes), self.batch_size):
            batch_nodes = nodes[i : i + self.batch_size]
            self._evaluate_batch(batch_nodes)

    def _evaluate_batch(self, nodes):
        if not nodes:
            return

        for node in nodes:
            if node.is_terminal():
                value = self._get_terminal_value(node)
                node.backup(value)
                continue

            board_tensor = self.model.encode_board_vectorized(node.board)

            with torch.no_grad():
                batch_tensor = board_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(batch_tensor)
                policy = outputs["policy"][0]
                value = outputs["value"][0].item()

            self._expand_node(node, policy)
            node.backup(value)

    def _expand_node(self, node, policy):
        if node.is_expanded or node.is_terminal():
            return

        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            return

        priors = []
        for move in legal_moves:
            move_idx = self.move_encoder.encode_move(move)
            if move_idx < len(policy):
                prior = policy[move_idx].item()
            else:
                prior = 0.001
            priors.append(prior)

        prior_sum = sum(priors) or 1.0
        priors = [p / prior_sum for p in priors]

        for move, prior in zip(legal_moves, priors, strict=False):
            child_board = node.board.copy()
            child_board.push(move)
            child = Node(child_board, parent=node, move=move, prior=prior)
            node.children[move] = child

        node.is_expanded = True

    def _get_terminal_value(self, node):
        result = node.board.result()

        if result == "1-0":
            value = 1.0
        elif result == "0-1":
            value = -1.0
        else:
            value = 0.0

        if not node.board.turn:
            value = -value

        return value
