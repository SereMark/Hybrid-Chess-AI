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
        copy_board: bool = True,
    ):
        if copy_board:
            self.board: chess.Board = board.copy()
        else:
            self.board: chess.Board = board
        self.parent: Node | None = parent
        self.move: chess.Move | None = move
        self.prior: float = prior
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: dict[chess.Move, Node] = {}
        self.is_expanded: bool = False
        self._legal_moves: list[chess.Move] | None = None
        self._is_terminal: bool | None = None

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
    
    def get_legal_moves(self) -> list[chess.Move]:
        if self._legal_moves is None:
            self._legal_moves = list(self.board.legal_moves)
        return self._legal_moves
    
    def is_terminal(self) -> bool:
        if self._is_terminal is None:
            self._is_terminal = self.board.is_game_over()
        return self._is_terminal
    
    @staticmethod
    def create_child_node(parent: "Node", move: chess.Move, prior: float) -> "Node":
        child_board = parent.board.copy()
        child_board.push(move)
        return Node(child_board, parent, move, prior, copy_board=False)


class MCTS:
    def __init__(self, model: ChessModel, move_encoder: MoveEncoder):
        self.model: ChessModel = model
        self.move_encoder: MoveEncoder = move_encoder
        self.reset_search_stats()

    def reset_search_stats(self) -> None:
        self.total_nodes_expanded: int = 0
        self.terminal_nodes_hit: int = 0
        self.total_simulations: int = 0
        self.model_forward_calls: int = 0
        self.searches_performed: int = 0

    def search_batch(self, boards: list[chess.Board]) -> list[dict[chess.Move, float]]:
        roots = [Node(board) for board in boards]
        self.searches_performed += 1

        for _ in range(SIMULATIONS):
            self.total_simulations += 1
            leaves = []

            for root in roots:
                node = root
                while node.is_expanded and not node.is_terminal():
                    node = node.select_child()
                    if node is None:
                        break
                if node:
                    leaves.append(node)

            if not leaves:
                continue

            boards_to_eval = []
            terminal_nodes = []

            for node in leaves:
                if node.is_terminal():
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
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                    self.model_forward_calls += 1
                    outputs = self.model(board_tensors)
                    policies = outputs.policy
                    values = outputs.value.squeeze(-1)

                policy_cpu = policies.cpu()
                values_cpu = values.cpu()
                for i, node in enumerate(boards_to_eval):
                    if not node.is_expanded:
                        legal_moves = node.get_legal_moves()
                        if legal_moves:
                            move_indices = [
                                self.move_encoder.encode_move(move)
                                for move in legal_moves
                            ]
                            priors = [
                                max(
                                    policy_cpu[i][idx].item()
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
                                node.children[move] = Node.create_child_node(node, move, prior)

                            node.is_expanded = True
                            self.total_nodes_expanded += 1

                    node.backup(values_cpu[i].item())

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
                moves = root.get_legal_moves()
                probs = {move: 1.0 / len(moves) for move in moves} if moves else {}
            results.append(probs)

        return results

    def get_search_stats(self) -> dict[str, float]:
        return {
            "total_simulations": self.total_simulations,
            "nodes_expanded": self.total_nodes_expanded,
            "terminal_nodes_hit": self.terminal_nodes_hit,
            "model_forward_calls": self.model_forward_calls,
            "searches_performed": self.searches_performed,
            "avg_expansions_per_search": self.total_nodes_expanded
            / max(self.searches_performed, 1),
            "terminal_hit_rate": self.terminal_nodes_hit
            / max(self.total_simulations, 1)
            * 100,
        }
