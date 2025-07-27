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

    def search_batch(self, boards: list[chess.Board]) -> list[dict[chess.Move, float]]:
        roots = [Node(board) for board in boards]

        for _ in range(SIMULATIONS):
            leaves = []

            for root in roots:
                node = root
                while node.is_expanded and not node.board.is_game_over():
                    node = node.select_child()
                    if node is None:
                        break
                if node and not node.board.is_game_over():
                    leaves.append(node)

            if not leaves:
                continue

            boards_to_eval = []
            terminal_nodes = []

            for node in leaves:
                if node.board.is_game_over():
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
