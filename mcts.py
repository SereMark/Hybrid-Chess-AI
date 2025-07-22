import logging
import math
import time
from typing import TYPE_CHECKING

import chess
import torch
from config import config
from game import uniform_probs
from model import ChessModel, MoveEncoder

if TYPE_CHECKING:
    from utils import ConsoleMetricsLogger


class Node:
    def __init__(
        self,
        board: chess.Board,
        parent: "Node | None" = None,
        move: chess.Move | None = None,
        prior: float = config.mcts.default_prior,
    ):
        self.board: chess.Board = board.copy()
        self.parent: Node | None = parent
        self.move: chess.Move | None = move
        self.prior: float = prior
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: dict[chess.Move, Node] = {}
        self.is_expanded: bool = False

    def get_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def get_ucb_score(self) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.get_value()
        exploration = (
            config.mcts.c_puct
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


class MCTS:
    def __init__(
        self,
        model: ChessModel,
        move_encoder: MoveEncoder,
        device: str,
        training_logger: "ConsoleMetricsLogger | None" = None,
    ) -> None:
        self.logger: logging.Logger = logging.getLogger("chess_ai.search")
        self.model: ChessModel = model
        self.move_encoder: MoveEncoder = move_encoder
        self.device: str = device
        self.training_logger: ConsoleMetricsLogger | None = training_logger

        self.total_searches: int = 0
        self.nodes_evaluated: int = 0
        self.total_search_time: float = 0.0
        self.expand_fails: int = 0

        self.logger.debug(
            f"MCTS initialized with {config.mcts.simulations} simulations per search"
        )

    def search_batch(self, boards: list[chess.Board]) -> list[dict[chess.Move, float]]:
        if not boards:
            self.logger.debug("Empty boards list provided to search_batch")
            return []

        search_start = time.perf_counter()
        batch_size = len(boards)
        self.total_searches += 1

        self.logger.debug(
            f"Starting MCTS search: {batch_size} positions, {config.mcts.simulations} simulations"
        )

        try:
            roots = [Node(board) for board in boards]
            total_evaluations = 0
            simulation_stats = {
                "leaves_found": 0,
                "evaluations": 0,
                "terminal_nodes": 0,
            }

            for simulation in range(config.mcts.simulations):
                nodes_to_evaluate = []
                for root in roots:
                    if not root.board.is_game_over():
                        leaf = self._select(root)
                        if leaf:
                            nodes_to_evaluate.append(leaf)
                            simulation_stats["leaves_found"] += 1

                if nodes_to_evaluate:
                    evaluation_count = self._eval_nodes(nodes_to_evaluate)
                    simulation_stats["evaluations"] += evaluation_count
                    total_evaluations += evaluation_count

                if (
                    batch_size > config.mcts.log_batch_size
                    and simulation % config.visit_threshold == 0
                ):
                    self.logger.debug(
                        f"Simulation {simulation}/{config.mcts.simulations} completed"
                    )

            results = []
            position_stats = {
                "valid_moves": 0,
                "uniform_fallback": 0,
                "empty_searches": 0,
            }

            for i, root in enumerate(roots):
                visit_counts = {
                    move: child.visits for move, child in root.children.items()
                }
                total_visits = sum(visit_counts.values())

                if total_visits > 0:
                    move_probs = {
                        move: count / total_visits
                        for move, count in visit_counts.items()
                    }
                    position_stats["valid_moves"] += 1

                    if total_visits < config.visit_threshold:
                        self.logger.debug(
                            f"Position {i} has low visits: {total_visits}"
                        )
                else:
                    legal_moves = list(root.board.legal_moves)
                    move_probs = uniform_probs(legal_moves)
                    if move_probs:
                        position_stats["uniform_fallback"] += 1
                    else:
                        position_stats["empty_searches"] += 1
                        self.logger.warning(f"Position {i} has no legal moves")

                results.append(move_probs)

            self.nodes_evaluated += total_evaluations
            search_time = time.perf_counter() - search_start
            self.total_search_time += search_time

            avg_time_per_position = search_time / batch_size
            evaluations_per_second = (
                total_evaluations / search_time if search_time > 0 else 0
            )

            if (
                batch_size >= config.mcts.min_log_size
                or search_time > config.mcts.slow_threshold
            ):
                avg_legal_moves = sum(
                    len(list(board.legal_moves)) for board in boards
                ) / len(boards)
                avg_pieces = sum(len(board.piece_map()) for board in boards) / len(
                    boards
                )

                self.logger.info(
                    f"MCTS Search Complete: {batch_size} positions in {search_time:.2f}s "
                    f"({avg_time_per_position:.3f}s/pos), {total_evaluations} evaluations "
                    f"({evaluations_per_second:.0f} eval/s)"
                )

                if self.training_logger:
                    self.training_logger.log_mcts(
                        search_time,
                        evaluations_per_second,
                        batch_size,
                        avg_legal_moves,
                        avg_pieces,
                    )

            if (
                position_stats["uniform_fallback"] > 0
                or position_stats["empty_searches"] > 0
            ):
                self.logger.info(
                    f"Search Quality: {position_stats['valid_moves']} normal, "
                    f"{position_stats['uniform_fallback']} uniform fallback, "
                    f"{position_stats['empty_searches']} empty"
                )

            return results

        except RuntimeError as e:
            search_time = time.perf_counter() - search_start
            self.logger.error(
                f"MCTS search failed after {search_time:.2f}s with {batch_size} positions: {e}",
                exc_info=True,
            )
            fallback_results = []
            for board in boards:
                legal_moves = list(board.legal_moves)
                fallback_results.append(uniform_probs(legal_moves))
            return fallback_results

    def _select(self, node: Node) -> Node | None:
        current = node
        while current.is_expanded and not current.board.is_game_over():
            child = current.select_child()
            if child is None:
                break
            current = child
        return None if current.board.is_game_over() else current

    def _eval_nodes(self, nodes: list[Node]) -> int:
        total_evaluations = 0

        for i in range(0, len(nodes), config.mcts.batch_size):
            batch_nodes = nodes[i : i + config.mcts.batch_size]
            evaluations = self._process(batch_nodes)
            total_evaluations += evaluations

        return total_evaluations

    def _process(self, nodes: list[Node]) -> int:
        if not nodes:
            return 0

        terminal_nodes = []
        active_nodes = []
        active_boards = []
        evaluations = 0

        for node in nodes:
            if node.board.is_game_over():
                terminal_nodes.append(node)
            else:
                active_nodes.append(node)
                active_boards.append(node.board)

        for node in terminal_nodes:
            try:
                value = self._terminal_value(node)
                node.backup(value)
                evaluations += 1
            except ValueError as e:
                self.logger.error(f"Error processing terminal node: {e}")

        if active_nodes:
            try:
                board_tensors = self.model.encode_board(active_boards)

                with torch.no_grad():
                    outputs = self.model(board_tensors)
                    policies = outputs.policy
                    values = outputs.value.squeeze(-1)

                if torch.any(torch.isnan(values)) or torch.any(torch.isnan(policies)):
                    self.logger.error("NaN detected in model outputs")
                    return evaluations

                for i, node in enumerate(active_nodes):
                    try:
                        policy = policies[i]
                        value = values[i].item()

                        if math.isnan(value) or math.isinf(value):
                            self.logger.warning(
                                f"Invalid value {value} for node {i}, using 0.0"
                            )
                            value = 0.0

                        self._expand(node, policy)
                        node.backup(value)
                        evaluations += 1

                    except (RuntimeError, ValueError) as e:
                        self.logger.error(f"Error processing active node {i}: {e}")
                        node.backup(0.0)
                        evaluations += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.error(
                        f"GPU OOM during batch processing ({len(active_nodes)} nodes)"
                    )
                    torch.cuda.empty_cache()
                else:
                    self.logger.error(f"Runtime error during batch processing: {e}")
                return evaluations
            except ValueError as e:
                self.logger.error(
                    f"Unexpected error during neural network evaluation: {e}",
                    exc_info=True,
                )
                return evaluations

        return evaluations

    def _expand(self, node: Node, policy: torch.Tensor) -> None:
        if node.is_expanded:
            return

        try:
            legal_moves = list(node.board.legal_moves)
            if not legal_moves:
                self.logger.debug("No legal moves for node expansion")
                node.is_expanded = True
                return

            valid_moves = []
            priors = []
            encoding_failures = 0

            for move in legal_moves:
                try:
                    move_idx = self.move_encoder.encode_move(move)
                    valid_moves.append(move)
                    priors.append(move_idx if move_idx < len(policy) else -1)
                except ValueError:
                    encoding_failures += 1
                    continue

            if not valid_moves:
                self.logger.warning(
                    f"No valid moves after encoding, using all {len(legal_moves)} legal moves"
                )
                valid_moves = legal_moves
                priors = [config.mcts.default_prior] * len(valid_moves)
                self.expand_fails += 1
            else:
                policy_indices = [idx for idx in priors if idx >= 0]
                if policy_indices:
                    try:
                        policy_values = policy[policy_indices].cpu().numpy()
                        policy_dict = dict(
                            zip(policy_indices, policy_values, strict=True)
                        )
                    except (RuntimeError, ValueError):
                        policy_dict = {}
                else:
                    policy_dict = {}

                priors = [
                    policy_dict.get(idx, config.mcts.default_prior) for idx in priors
                ]

            prior_sum = sum(priors)
            if prior_sum > 0:
                priors = [p / prior_sum for p in priors]
            else:
                self.logger.warning(
                    f"Zero prior sum, using uniform distribution for {len(valid_moves)} moves"
                )
                priors = [1.0 / len(valid_moves)] * len(valid_moves)

            children_created = 0
            for move, prior in zip(valid_moves, priors, strict=True):
                try:
                    child_board = node.board.copy()
                    child_board.push(move)
                    node.children[move] = Node(
                        child_board, parent=node, move=move, prior=prior
                    )
                    children_created += 1
                except ValueError:
                    continue

            node.is_expanded = True

            if encoding_failures > 0:
                self.logger.debug(
                    f"Node expansion: {children_created}/{len(legal_moves)} children created, {encoding_failures} encoding failures"
                )

        except (RuntimeError, ValueError):
            self.logger.error("Critical error in node expansion")
            node.is_expanded = True
            self.expand_fails += 1

    def _terminal_value(self, node: Node) -> float:
        try:
            result = node.board.result()
            return {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
        except ValueError as e:
            self.logger.error(f"Error getting terminal value: {e}")
            return 0.0

    def get_statistics(self) -> dict[str, float]:
        avg_search_time = (
            self.total_search_time / self.total_searches
            if self.total_searches > 0
            else 0.0
        )
        avg_evaluations = (
            self.nodes_evaluated / self.total_searches
            if self.total_searches > 0
            else 0.0
        )
        eval_rate = (
            self.nodes_evaluated / self.total_search_time
            if self.total_search_time > 0
            else 0.0
        )

        return {
            "total_searches": float(self.total_searches),
            "total_nodes_evaluated": float(self.nodes_evaluated),
            "total_search_time": self.total_search_time,
            "avg_search_time": avg_search_time,
            "avg_evaluations_per_search": avg_evaluations,
            "evaluations_per_second": eval_rate,
            "expansion_failures": float(self.expand_fails),
            "failure_rate": self.expand_fails / self.total_searches
            if self.total_searches > 0
            else 0.0,
        }
