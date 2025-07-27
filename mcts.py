import math
from typing import Optional

import chess
import numpy as np
import torch
from config import C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, MOVE_COUNT, SIMULATIONS
from model import ChessModel
from move_encoder import MoveEncoder


class Node:
    def __init__(
        self,
        board: chess.Board,
        parent: Optional["Node"] = None,
        move: chess.Move | None = None,
        prior: float = 0.001,
    ):
        self.board: chess.Board = board.copy() if parent is None else board
        self.parent: Node | None = parent
        self.move: chess.Move | None = move
        self.prior: float = prior
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: dict[chess.Move, Node] = {}
        self.is_expanded: bool = False

    def ucb_score(self) -> float:
        if self.visits == 0:
            return float("inf")
        q = self.value_sum / self.visits
        if self.parent:
            u = C_PUCT * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
            return q + u
        return q

    def select_child(self) -> Optional["Node"]:
        return (
            max(self.children.values(), key=lambda c: c.ucb_score())
            if self.children
            else None
        )

    def backup(self, value: float) -> None:
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)


class MCTS:
    def __init__(self, model: ChessModel, move_encoder: MoveEncoder):
        self.model: ChessModel = model
        self.move_encoder: MoveEncoder = move_encoder
        self.reset_search_stats()

    def reset_search_stats(self) -> None:
        self.total_nodes_explored = 0
        self.total_nodes_expanded = 0
        self.terminal_nodes_hit = 0
        self.total_simulations = 0
        self.max_tree_depth = 0
        self.model_forward_calls = 0
        self.searches_performed = 0

    def search_batch(self, boards: list[chess.Board]) -> list[dict[chess.Move, float]]:
        roots = [Node(board) for board in boards]
        self.searches_performed += 1
        simulation_nodes_explored = 0

        for _ in range(SIMULATIONS):
            self.total_simulations += 1
            leaves = []

            for root in roots:
                node = root
                depth = 0
                while node.is_expanded and not node.board.is_game_over():
                    node = node.select_child()
                    depth += 1
                    self.total_nodes_explored += 1
                    if node is None:
                        break
                self.max_tree_depth = max(self.max_tree_depth, depth)
                if node and not node.board.is_game_over():
                    leaves.append(node)

            if not leaves:
                continue

            boards_to_eval = []
            terminal_nodes = []

            for node in leaves:
                if node.board.is_game_over():
                    self.terminal_nodes_hit += 1
                    result_str = node.board.result()
                    result_value = {"1-0": 1.0, "0-1": -1.0}.get(result_str, 0.0)
                    value = (
                        result_value
                        if node.board.turn == chess.WHITE
                        else -result_value
                    )
                    terminal_nodes.append((node, value))
                else:
                    boards_to_eval.append(node)

            if boards_to_eval:
                board_tensors = self.model.encode_board(
                    [n.board for n in boards_to_eval]
                )
                with torch.no_grad():
                    self.model_forward_calls += 1
                    outputs = self.model(board_tensors)
                    policies = outputs.policy
                    values = outputs.value.squeeze(-1)

                for i, node in enumerate(boards_to_eval):
                    if not node.is_expanded:
                        legal_moves = list(node.board.legal_moves)
                        if legal_moves:
                            move_indices = [
                                self.move_encoder.encode_move(move)
                                for move in legal_moves
                            ]
                            priors = [
                                max(
                                    policies[i][idx].item()
                                    if 0 <= idx < MOVE_COUNT
                                    else 0.001,
                                    0.001,
                                )
                                for idx in move_indices
                            ]

                            prior_sum = sum(priors)
                            priors = [p / prior_sum for p in priors]

                            if node.parent is None:
                                noise = np.random.dirichlet(
                                    [DIRICHLET_ALPHA] * len(priors)
                                )
                                priors = [
                                    (1 - DIRICHLET_EPSILON) * p + DIRICHLET_EPSILON * n
                                    for p, n in zip(priors, noise, strict=False)
                                ]

                            for move, prior in zip(legal_moves, priors, strict=False):
                                child_board = node.board.copy()
                                child_board.push(move)
                                node.children[move] = Node(
                                    child_board, node, move, prior
                                )

                            node.is_expanded = True
                            self.total_nodes_expanded += 1

                    node.backup(values[i].item())

            for node, value in terminal_nodes:
                node.backup(value)

        results = []
        for root in roots:
            if root.children:
                visits = {move: child.visits for move, child in root.children.items()}
                total = sum(visits.values())
                probs = (
                    {move: count / total for move, count in visits.items()}
                    if total > 0
                    else {}
                )
            else:
                moves = list(root.board.legal_moves)
                probs = {move: 1.0 / len(moves) for move in moves} if moves else {}
            results.append(probs)

        return results

    def get_search_stats(self) -> dict[str, float]:
        return {
            "total_simulations": self.total_simulations,
            "nodes_explored": self.total_nodes_explored,
            "nodes_expanded": self.total_nodes_expanded,
            "terminal_nodes_hit": self.terminal_nodes_hit,
            "max_tree_depth": self.max_tree_depth,
            "model_forward_calls": self.model_forward_calls,
            "searches_performed": self.searches_performed,
            "avg_nodes_per_search": self.total_nodes_explored / max(self.searches_performed, 1),
            "avg_expansions_per_search": self.total_nodes_expanded / max(self.searches_performed, 1),
            "terminal_hit_rate": self.terminal_nodes_hit / max(self.total_simulations, 1) * 100,
            "expansion_efficiency": self.total_nodes_expanded / max(self.total_nodes_explored, 1) * 100,
        }
