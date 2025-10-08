#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
PYTHON_RELEASE_DIR = PYTHON_DIR / "Release"

for candidate in (PYTHON_RELEASE_DIR, PYTHON_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import chess as pychess
import chesscore as ccore


DEFAULT_POSITIONS = [200]
DEFAULT_MAX_PLIES = [80]
DEFAULT_SIMULATIONS = [256]
DEFAULT_MAX_BATCH = [32]
DEFAULT_REPETITIONS = [3]
DEFAULT_SEEDS = [2025]


@dataclass(frozen=True)
class MCTSScenario:
    name: str
    positions: int
    max_random_plies: int
    simulations: int
    max_batch: int
    repetitions: int
    seed: int
    description: str = "Random playout positions"

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "positions": self.positions,
            "max_random_plies": self.max_random_plies,
            "simulations": self.simulations,
            "max_batch": self.max_batch,
            "repetitions": self.repetitions,
            "seed": self.seed,
            "description": self.description,
        }


TEMPLATE_SCENARIOS: Dict[str, List[Dict[str, object]]] = {
    "baseline": [
        {
            "name": "mcts_small",
            "positions": 200,
            "max_random_plies": 80,
            "simulations": 256,
            "max_batch": 32,
            "repetitions": 3,
            "seed": 9001,
            "description": "Random playout positions (small)",
        },
        {
            "name": "mcts_medium",
            "positions": 400,
            "max_random_plies": 120,
            "simulations": 384,
            "max_batch": 48,
            "repetitions": 3,
            "seed": 42,
            "description": "Random playout positions (medium)",
        },
        {
            "name": "mcts_deep",
            "positions": 600,
            "max_random_plies": 200,
            "simulations": 512,
            "max_batch": 64,
            "repetitions": 3,
            "seed": 1337,
            "description": "Random playout positions (deep)",
        },
    ],
    "quick": [
        {
            "name": "mcts_quick",
            "positions": 100,
            "max_random_plies": 60,
            "simulations": 128,
            "max_batch": 16,
            "repetitions": 2,
            "seed": 2025,
            "description": "Quick diagnostic",
        }
    ],
}


def generate_random_fens(count: int, max_random_plies: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    fens: List[str] = []
    for _ in range(count):
        board = pychess.Board()
        plies = rng.randint(0, max_random_plies)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
        fens.append(board.fen())
    return fens


def uniform_evaluator_single(position: ccore.Position) -> Tuple[List[float], float]:
    moves = position.legal_moves()
    count = len(moves)
    if count == 0:
        return [], 0.0
    prior = 1.0 / count
    return [prior] * count, 0.0


def uniform_evaluator_batched(positions: Sequence[ccore.Position], moves_lists: Sequence[Sequence[ccore.Move]]) -> Tuple[List[List[float]], List[float]]:
    policies: List[List[float]] = []
    values: List[float] = []
    for moves in moves_lists:
        count = len(moves)
        if count == 0:
            policies.append([])
        else:
            prior = 1.0 / count
            policies.append([prior] * count)
        values.append(0.0)
    return policies, values


class PythonMCTS:
    def __init__(self, simulations: int = 256, c_puct: float = 1.5) -> None:
        self.simulations = simulations
        self.c_puct = c_puct
        self.nodes: Dict[int, "PythonNode"] = {}

    def reset(self) -> None:
        self.nodes.clear()

    def search_batched_legal(self, position: ccore.Position, evaluator: Callable[[ccore.Position], Tuple[List[float], float]], max_batch: int) -> List[int]:
        root_copy = ccore.Position(position)
        root_turn = root_copy.turn
        for _ in range(self.simulations):
            self._simulate(root_copy, root_turn, evaluator)
        root_node = self.nodes.get(root_copy.hash)
        if not root_node:
            return []
        return [root_node.visits[move_id] for move_id in root_node.move_ids]

    def _simulate(self, root_position: ccore.Position, root_turn: int, evaluator: Callable[[ccore.Position], Tuple[List[float], float]]) -> None:
        position = ccore.Position(root_position)
        nodes_path: List[Tuple["PythonNode", int]] = []
        current_turn = root_turn
        while True:
            node, leaf_value, created = self._get_or_create_node(position, evaluator)
            if created:
                value = leaf_value if current_turn == root_turn else -leaf_value
                break
            if not node.move_ids:
                result = position.result()
                value = terminal_value(result, root_turn)
                break
            move_id = self._select_move(node)
            move = node.moves_by_id[move_id]
            nodes_path.append((node, move_id))
            result = position.make_move(move)
            current_turn = position.turn
            if result != ccore.Result.ONGOING:
                value = terminal_value(result, root_turn)
                break
        self._backpropagate(nodes_path, value)

    def _get_or_create_node(self, position: ccore.Position, evaluator: Callable[[ccore.Position], Tuple[List[float], float]]) -> Tuple["PythonNode", float, bool]:
        key = position.hash
        node = self.nodes.get(key)
        if node is not None:
            return node, 0.0, False
        moves = position.legal_moves()
        priors, value = evaluator(position)
        move_ids = [move.to_square * 64 + move.from_square for move in moves]
        visits = {move_id: 0 for move_id in move_ids}
        values = {move_id: 0.0 for move_id in move_ids}
        priors_map = {move_id: priors[idx] if idx < len(priors) else 0.0 for idx, move_id in enumerate(move_ids)}
        node = PythonNode(move_ids=move_ids, moves=[m for m in moves], moves_by_id={mid: m for mid, m in zip(move_ids, moves)},
                          priors=priors_map, visits=visits, values=values, total_visits=0)
        self.nodes[key] = node
        return node, value, True

    def _select_move(self, node: "PythonNode") -> int:
        best_score = -float("inf")
        best_move = node.move_ids[0]
        total = max(node.total_visits, 1)
        sqrt_total = math.sqrt(total)
        for move_id in node.move_ids:
            n = node.visits[move_id]
            w = node.values[move_id]
            q = w / n if n > 0 else 0.0
            p = node.priors.get(move_id, 0.0)
            u = self.c_puct * p * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move_id
        return best_move

    def _backpropagate(self, path: List[Tuple["PythonNode", int]], value: float) -> None:
        for node, move_id in reversed(path):
            node.visits[move_id] += 1
            node.values[move_id] += value
            node.total_visits += 1
            value = -value


@dataclass
class PythonNode:
    move_ids: List[int]
    moves: List[ccore.Move]
    moves_by_id: Dict[int, ccore.Move]
    priors: Dict[int, float]
    visits: Dict[int, int]
    values: Dict[int, float]
    total_visits: int = 0


def terminal_value(result: ccore.Result, root_turn: int) -> float:
    if result == ccore.Result.DRAW:
        return 0.0
    if result == ccore.Result.WHITE_WIN:
        winner = ccore.Color.WHITE
    elif result == ccore.Result.BLACK_WIN:
        winner = ccore.Color.BLACK
    else:
        return 0.0
    return 1.0 if winner == root_turn else -1.0


def benchmark_cpp_mcts(fens: Sequence[str], scenario: MCTSScenario) -> float:
    engine = ccore.MCTS(scenario.simulations)
    start = time.perf_counter()
    for fen in fens:
        engine.reset_tree()
        pos = ccore.Position()
        pos.from_fen(fen)
        engine.search_batched_legal(pos, uniform_evaluator_batched, scenario.max_batch)
    return time.perf_counter() - start


def benchmark_python_mcts(fens: Sequence[str], scenario: MCTSScenario) -> float:
    engine = PythonMCTS(simulations=scenario.simulations)
    start = time.perf_counter()
    for fen in fens:
        engine.reset()
        pos = ccore.Position()
        pos.from_fen(fen)
        engine.search_batched_legal(pos, uniform_evaluator_single, scenario.max_batch)
    return time.perf_counter() - start


def build_markdown_report(reports: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    for report in reports:
        scenario = report["scenario"]
        stats = report["stats"]
        lines.append(f"# MCTS Benchmark Report — {scenario['name']}")
        lines.append("")
        lines.append(f"Generated: {report['generated_at']}")
        lines.append(f"Python: {report['python_version']}")
        lines.append(f"Platform: {report['platform']}")
        lines.append("")
        lines.append("## Scenario")
        lines.append("")
        lines.append(f"- Positions: {scenario['positions']}")
        lines.append(f"- Max random plies: {scenario['max_random_plies']}")
        lines.append(f"- Simulations: {scenario['simulations']}")
        lines.append(f"- Max batch: {scenario['max_batch']}")
        lines.append(f"- Repetitions: {scenario['repetitions']}")
        lines.append(f"- Seed: {scenario['seed']}")
        lines.append(f"- Description: {scenario['description']}")
        lines.append("")
        lines.append("## Timing")
        lines.append("")
        cpp = stats["cpp"]
        py = stats["python"]
        lines.append(f"- C++ MCTS time: {cpp['time_seconds']:.4f} s")
        lines.append(f"- Python MCTS time: {py['time_seconds']:.4f} s")
        lines.append(f"- Speedup (cpp/python): {py['time_seconds'] / cpp['time_seconds']:.2f}x")
        lines.append(f"- Positions per second (C++): {cpp['positions_per_second']:.0f}")
        lines.append(f"- Positions per second (Python): {py['positions_per_second']:.0f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, nargs="*", default=None,
                        help="Number of random FENs per scenario")
    parser.add_argument("--max-random-plies", type=int, nargs="*", default=None,
                        help="Upper bound on random plies per scenario")
    parser.add_argument("--simulations", type=int, nargs="*", default=None,
                        help="Simulation count per scenario")
    parser.add_argument("--max-batch", type=int, nargs="*", default=None,
                        help="Max batch size for C++ MCTS")
    parser.add_argument("--repetitions", type=int, nargs="*", default=None,
                        help="Timing repetitions per scenario")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Seeds for random position generation")
    parser.add_argument("--scenario-name", type=str, default="mcts",
                        help="Base name for ad-hoc scenarios")
    parser.add_argument("--template", choices=sorted(TEMPLATE_SCENARIOS.keys()),
                        help="Use a preset scenario suite")
    parser.add_argument("--output-json", type=Path,
                        default=Path("benchmark_reports") / "mcts" / "mcts_summary.json",
                        help="Path for JSON report")
    parser.add_argument("--output-markdown", type=Path,
                        default=Path("benchmark_reports") / "mcts" / "mcts_summary.md",
                        help="Path for Markdown report")
    args = parser.parse_args()

    positions_list = list(args.positions) if args.positions else DEFAULT_POSITIONS
    maxplies_list = list(args.max_random_plies) if args.max_random_plies else DEFAULT_MAX_PLIES
    simulations_list = list(args.simulations) if args.simulations else DEFAULT_SIMULATIONS
    max_batch_list = list(args.max_batch) if args.max_batch else DEFAULT_MAX_BATCH
    repetitions_list = list(args.repetitions) if args.repetitions else DEFAULT_REPETITIONS
    seeds_list = list(args.seeds) if args.seeds else DEFAULT_SEEDS

    if args.template:
        if args.positions or args.max_random_plies or args.simulations or args.max_batch or args.repetitions or args.seeds:
            raise SystemExit("When --template is used, omit manual scenario arguments.")
        raw = TEMPLATE_SCENARIOS[args.template]
    else:
        if not positions_list:
            raise SystemExit("Provide at least one --positions value or use --template.")
        length = max(len(positions_list), len(maxplies_list), len(simulations_list), len(max_batch_list), len(repetitions_list), len(seeds_list))
        raw = []
        for idx in range(length):
            raw.append({
                "name": f"{args.scenario_name}_{idx}" if length > 1 else args.scenario_name,
                "positions": positions_list[idx % len(positions_list)],
                "max_random_plies": maxplies_list[idx % len(maxplies_list)],
                "simulations": simulations_list[idx % len(simulations_list)],
                "max_batch": max_batch_list[idx % len(max_batch_list)],
                "repetitions": repetitions_list[idx % len(repetitions_list)],
                "seed": seeds_list[idx % len(seeds_list)],
                "description": "Random playout positions",
            })

    scenarios = [MCTSScenario(**cfg) for cfg in raw]

    for scenario in scenarios:
        if scenario.positions <= 0 or scenario.max_random_plies < 0 or scenario.simulations <= 0 or scenario.repetitions <= 0 or scenario.max_batch <= 0:
            raise SystemExit("Scenario parameters must be positive (plies can be zero or greater).")

    reports: List[Dict[str, object]] = []
    for scenario in scenarios:
        fens = generate_random_fens(scenario.positions, scenario.max_random_plies, scenario.seed)
        cpp_times: List[float] = []
        python_times: List[float] = []
        for _ in range(scenario.repetitions):
            cpp_times.append(benchmark_cpp_mcts(fens, scenario))
            python_times.append(benchmark_python_mcts(fens, scenario))
        cpp_time = statistics.fmean(cpp_times)
        python_time = statistics.fmean(python_times)
        stats = {
            "cpp": {
                "time_seconds": cpp_time,
                "positions_per_second": scenario.positions / cpp_time if cpp_time > 0 else float("inf"),
            },
            "python": {
                "time_seconds": python_time,
                "positions_per_second": scenario.positions / python_time if python_time > 0 else float("inf"),
            },
        }
        reports.append(
            {
                "scenario": scenario.to_dict(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version,
                "platform": platform.platform(),
                "stats": stats,
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(reports, indent=2))
    if args.output_markdown:
        args.output_markdown.write_text(build_markdown_report(reports))


if __name__ == "__main__":
    main()
