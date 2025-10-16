# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import csv
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
from typing import Any, Callable, Sequence, cast

from _bench_common import import_chesscore, import_python_chess, prepare_extension_import_paths

prepare_extension_import_paths()
pychess: Any = import_python_chess()

ccore: Any = import_chesscore()

# --------------------------------------------------------------------------- defaults

DEFAULT_POSITIONS = [120]
DEFAULT_MAX_PLIES = [70]
DEFAULT_SIMULATIONS = [192]
DEFAULT_MAX_BATCH = [24]
DEFAULT_REPETITIONS = [2]
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

    def to_dict(self) -> dict[str, object]:
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


TEMPLATE_SCENARIOS: dict[str, list[dict[str, object]]] = {
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
            "name": "mcts_quick_small",
            "positions": 120,
            "max_random_plies": 70,
            "simulations": 192,
            "max_batch": 24,
            "repetitions": 2,
            "seed": 2025,
            "description": "Balanced comparison (small)",
        },
        {
            "name": "mcts_quick_medium",
            "positions": 240,
            "max_random_plies": 110,
            "simulations": 224,
            "max_batch": 32,
            "repetitions": 2,
            "seed": 2026,
            "description": "Balanced comparison (medium)",
        },
        {
            "name": "mcts_quick_deep",
            "positions": 320,
            "max_random_plies": 150,
            "simulations": 288,
            "max_batch": 40,
            "repetitions": 2,
            "seed": 2027,
            "description": "Balanced comparison (deep)",
        },
    ],
}

# --------------------------------------------------------------------------- generators


def generate_random_fens(count: int, max_random_plies: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens: list[str] = []
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


def uniform_evaluator_single(position: ccore.Position) -> tuple[list[float], float]:
    moves = position.legal_moves()
    n = len(moves)
    if n == 0:
        return [], 0.0
    p = 1.0 / n
    return [p] * n, 0.0


def uniform_evaluator_batched(
    positions: Sequence[ccore.Position], moves_lists: Sequence[Sequence[ccore.Move]]
) -> tuple[list[list[float]], list[float]]:
    policies: list[list[float]] = []
    values: list[float] = []
    for moves in moves_lists:
        n = len(moves)
        policies.append([] if n == 0 else [1.0 / n] * n)
        values.append(0.0)
    return policies, values


@dataclass
class PythonNode:
    move_ids: list[int]
    moves: list[ccore.Move]
    moves_by_id: dict[int, ccore.Move]
    priors: dict[int, float]
    visits: dict[int, int]
    values: dict[int, float]
    total_visits: int = 0


class PythonMCTS:
    def __init__(self, simulations: int = 256, c_puct: float = 1.5) -> None:
        self.simulations = simulations
        self.c_puct = c_puct
        self.nodes: dict[int, PythonNode] = {}

    def reset(self) -> None:
        self.nodes.clear()

    def search_batched_legal(
        self, position: ccore.Position, evaluator: Callable[[ccore.Position], tuple[list[float], float]], max_batch: int
    ) -> list[int]:
        root_copy = ccore.Position(position)
        root_turn = root_copy.turn
        for _ in range(self.simulations):
            self._simulate(root_copy, root_turn, evaluator)
        root_node = self.nodes.get(root_copy.hash)
        if not root_node:
            return []
        return [root_node.visits[mid] for mid in root_node.move_ids]

    def _simulate(
        self,
        root_position: ccore.Position,
        root_turn: int,
        evaluator: Callable[[ccore.Position], tuple[list[float], float]],
    ) -> None:
        position = ccore.Position(root_position)
        nodes_path: list[tuple[PythonNode, int]] = []
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

    def _get_or_create_node(
        self, position: ccore.Position, evaluator: Callable[[ccore.Position], tuple[list[float], float]]
    ) -> tuple[PythonNode, float, bool]:
        key = position.hash
        node = self.nodes.get(key)
        if node is not None:
            return node, 0.0, False
        moves = position.legal_moves()
        priors, value = evaluator(position)
        move_ids = [m.to_square * 64 + m.from_square for m in moves]
        visits = {mid: 0 for mid in move_ids}
        values = {mid: 0.0 for mid in move_ids}
        priors_map = {mid: (priors[i] if i < len(priors) else 0.0) for i, mid in enumerate(move_ids)}
        node = PythonNode(
            move_ids=move_ids,
            moves=list(moves),
            moves_by_id={mid: m for mid, m in zip(move_ids, moves, strict=False)},
            priors=priors_map,
            visits=visits,
            values=values,
            total_visits=0,
        )
        self.nodes[key] = node
        return node, value, True

    def _select_move(self, node: PythonNode) -> int:
        best_score = -float("inf")
        best_move = node.move_ids[0]
        total = max(node.total_visits, 1)
        sqrt_total = math.sqrt(total)
        for mid in node.move_ids:
            n = node.visits[mid]
            w = node.values[mid]
            q = w / n if n > 0 else 0.0
            p = node.priors.get(mid, 0.0)
            u = self.c_puct * p * sqrt_total / (1 + n)
            sc = q + u
            if sc > best_score:
                best_score = sc
                best_move = mid
        return best_move

    def _backpropagate(self, path: list[tuple[PythonNode, int]], value: float) -> None:
        for node, mid in reversed(path):
            node.visits[mid] += 1
            node.values[mid] += value
            node.total_visits += 1
            value = -value


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


def _move_key(move: ccore.Move) -> str:
    try:
        return ccore.uci_of_move(move)
    except AttributeError:
        return f"{move.to_square}_{move.from_square}_{move.promotion}"


def _collect_visit_distribution_cpp(
    engine: ccore.MCTS, fen: str, scenario: MCTSScenario
) -> tuple[dict[str, float], dict[str, int]]:
    engine.reset_tree()
    position = ccore.Position()
    position.from_fen(fen)
    moves = list(position.legal_moves())
    if not moves:
        return {}, {}
    counts = engine.search_batched_legal(position, uniform_evaluator_batched, scenario.max_batch)
    if len(counts) != len(moves):
        return {}, {}
    total = float(sum(counts))
    if total <= 0.0:
        return {}, {_move_key(m): int(c) for m, c in zip(moves, counts, strict=False)}
    dist = {_move_key(m): c / total for m, c in zip(moves, counts, strict=False)}
    raw = {_move_key(m): int(c) for m, c in zip(moves, counts, strict=False)}
    return dist, raw


def _collect_visit_distribution_python(
    engine: PythonMCTS, fen: str, scenario: MCTSScenario
) -> tuple[dict[str, float], dict[str, int]]:
    engine.reset()
    position = ccore.Position()
    position.from_fen(fen)
    moves = list(position.legal_moves())
    if not moves:
        return {}, {}
    counts = engine.search_batched_legal(position, uniform_evaluator_single, scenario.max_batch)
    if len(counts) != len(moves):
        return {}, {}
    total = float(sum(counts))
    if total <= 0.0:
        return {}, {_move_key(m): int(c) for m, c in zip(moves, counts, strict=False)}
    dist = {_move_key(m): c / total for m, c in zip(moves, counts, strict=False)}
    raw = {_move_key(m): int(c) for m, c in zip(moves, counts, strict=False)}
    return dist, raw


def _top_move(distribution: dict[str, float]) -> str | None:
    if not distribution:
        return None
    return max(distribution.items(), key=lambda item: item[1])[0]


def compute_alignment_metrics(fens: Sequence[str], scenario: MCTSScenario, sample_size: int = 64) -> dict[str, object]:
    if not fens:
        return {"positions_checked": 0, "mean_l1": 0.0, "max_l1": 0.0, "top_match_pct": 100.0, "examples": []}
    cpp_engine = ccore.MCTS(scenario.simulations)
    py_engine = PythonMCTS(simulations=scenario.simulations)
    l1_distances: list[float] = []
    top_matches = 0
    evaluated = 0
    examples: list[dict[str, object]] = []
    limit = min(sample_size, len(fens))
    for fen in fens[:limit]:
        cpp_dist, _ = _collect_visit_distribution_cpp(cpp_engine, fen, scenario)
        py_dist, _ = _collect_visit_distribution_python(py_engine, fen, scenario)
        if not cpp_dist or not py_dist:
            continue
        all_moves = set(cpp_dist) | set(py_dist)
        l1 = sum(abs(cpp_dist.get(m, 0.0) - py_dist.get(m, 0.0)) for m in all_moves)
        l1_distances.append(l1)
        evaluated += 1
        if _top_move(cpp_dist) == _top_move(py_dist):
            top_matches += 1
        if len(examples) < 5 and (l1 > 0.2 or _top_move(cpp_dist) != _top_move(py_dist)):
            examples.append(
                {"fen": fen, "l1_distance": l1, "top_cpp": _top_move(cpp_dist), "top_python": _top_move(py_dist)}
            )
    if evaluated == 0:
        return {"positions_checked": 0, "mean_l1": 0.0, "max_l1": 0.0, "top_match_pct": 100.0, "examples": []}
    mean_l1 = float(statistics.fmean(l1_distances))
    max_l1 = float(max(l1_distances))
    top_match_pct = 100.0 * top_matches / evaluated
    return {
        "positions_checked": evaluated,
        "mean_l1": mean_l1,
        "max_l1": max_l1,
        "top_match_pct": top_match_pct,
        "examples": examples,
    }


def build_markdown_report(reports: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for report in reports:
        report_dict = report
        scenario = cast(dict[str, Any], report_dict["scenario"])
        stats = cast(dict[str, Any], report_dict["stats"])
        lines.append(f"# MCTS Benchmark Report - {scenario['name']}")
        lines.append("")
        lines.append(f"Generated: {report_dict['generated_at']}")
        lines.append(f"Python: {report_dict['python_version']}")
        lines.append(f"Platform: {report_dict['platform']}")
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
        cpp_stats = cast(dict[str, Any], stats["cpp"])
        py_stats = cast(dict[str, Any], stats["python"])
        lines.append(f"- C++ MCTS time: {cpp_stats['time_seconds']:.4f} s")
        lines.append(f"- Python MCTS time: {py_stats['time_seconds']:.4f} s")
        if cpp_stats["time_seconds"] > 0:
            lines.append(f"- Speedup (cpp/python): {py_stats['time_seconds'] / cpp_stats['time_seconds']:.2f}x")
        lines.append(f"- Positions per second (C++): {cpp_stats['positions_per_second']:.0f}")
        lines.append(f"- Positions per second (Python): {py_stats['positions_per_second']:.0f}")
        lines.append("")
        alignment = report_dict.get("alignment")
        if alignment:
            alignment_dict = cast(dict[str, Any], alignment)
            lines.append("## Alignment")
            lines.append("")
            lines.append(f"- Positions checked: {alignment_dict.get('positions_checked', 0)}")
            lines.append(f"- Mean L1 distance: {alignment_dict.get('mean_l1', 0.0):.4f}")
            lines.append(f"- Max L1 distance: {alignment_dict.get('max_l1', 0.0):.4f}")
            lines.append(f"- Top move agreement: {alignment_dict.get('top_match_pct', 0.0):.1f}%")
            examples = cast(list[dict[str, Any]], alignment_dict.get("examples", []))
            if examples:
                lines.append("- Example disagreements:")
                for example in examples:
                    lines.append(f"  - FEN: `{example['fen']}` (L1={example.get('l1_distance', 0.0):.3f})")
                    lines.append(f"    C++ top move: {example.get('top_cpp', 'n/a')}")
                    lines.append(f"    Python top move: {example.get('top_python', 'n/a')}")
            lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, nargs="*", default=None, help="Number of random FENs per scenario")
    parser.add_argument("--max-random-plies", type=int, nargs="*", default=None, help="Upper bound on random plies")
    parser.add_argument("--simulations", type=int, nargs="*", default=None, help="Simulation count per scenario")
    parser.add_argument("--max-batch", type=int, nargs="*", default=None, help="Max batch size for C++ MCTS")
    parser.add_argument("--repetitions", type=int, nargs="*", default=None, help="Timing repetitions per scenario")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Seeds for random position generation")
    parser.add_argument("--scenario-name", type=str, default="mcts", help="Base name for ad-hoc scenarios")
    parser.add_argument("--template", choices=sorted(TEMPLATE_SCENARIOS.keys()), help="Use a preset scenario suite")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "mcts_cpp.csv",
        help="Path for CSV timing summary",
    )
    parser.add_argument(
        "--alignment-sample", type=int, default=32, help="Positions sampled for visit distribution alignment"
    )
    args = parser.parse_args()

    positions_list = list(args.positions) if args.positions else DEFAULT_POSITIONS
    maxplies_list = list(args.max_random_plies) if args.max_random_plies else DEFAULT_MAX_PLIES
    simulations_list = list(args.simulations) if args.simulations else DEFAULT_SIMULATIONS
    max_batch_list = list(args.max_batch) if args.max_batch else DEFAULT_MAX_BATCH
    repetitions_list = list(args.repetitions) if args.repetitions else DEFAULT_REPETITIONS
    seeds_list = list(args.seeds) if args.seeds else DEFAULT_SEEDS

    raw: list[dict[str, Any]]
    if args.template:
        if any([args.positions, args.max_random_plies, args.simulations, args.max_batch, args.repetitions, args.seeds]):
            raise SystemExit("When --template is used, omit manual scenario arguments.")
        raw = cast(list[dict[str, Any]], TEMPLATE_SCENARIOS[args.template])
    else:
        if not positions_list:
            raise SystemExit("Provide at least one --positions value or use --template.")
        length = max(
            len(positions_list),
            len(maxplies_list),
            len(simulations_list),
            len(max_batch_list),
            len(repetitions_list),
            len(seeds_list),
        )
        raw = []
        for idx in range(length):
            cfg_dict: dict[str, Any] = {
                "name": f"{args.scenario_name}_{idx}" if length > 1 else args.scenario_name,
                "positions": positions_list[idx % len(positions_list)],
                "max_random_plies": maxplies_list[idx % len(maxplies_list)],
                "simulations": simulations_list[idx % len(simulations_list)],
                "max_batch": max_batch_list[idx % len(max_batch_list)],
                "repetitions": repetitions_list[idx % len(repetitions_list)],
                "seed": seeds_list[idx % len(seeds_list)],
                "description": "Random playout positions",
            }
            raw.append(cfg_dict)

    scenarios = [MCTSScenario(**cast(dict[str, Any], cfg)) for cfg in raw]
    for s in scenarios:
        if s.positions <= 0 or s.max_random_plies < 0 or s.simulations <= 0 or s.repetitions <= 0 or s.max_batch <= 0:
            raise SystemExit("Scenario parameters must be positive (plies can be zero or greater).")

    reports: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        fens = generate_random_fens(scenario.positions, scenario.max_random_plies, scenario.seed)
        cpp_times: list[float] = []
        python_times: list[float] = []
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
        alignment = compute_alignment_metrics(fens, scenario, sample_size=args.alignment_sample)
        reports.append(
            {
                "scenario": scenario.to_dict(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version,
                "platform": platform.platform(),
                "stats": stats,
                "alignment": alignment,
            }
        )
        cpp_stats = cast(dict[str, Any], stats["cpp"])
        py_stats = cast(dict[str, Any], stats["python"])
        alignment_dict = cast(dict[str, Any], alignment)
        csv_rows.append(
            {
                "scenario": scenario.name,
                "positions": scenario.positions,
                "max_random_plies": scenario.max_random_plies,
                "simulations": scenario.simulations,
                "max_batch": scenario.max_batch,
                "repetitions": scenario.repetitions,
                "cpp_time_s": cpp_time,
                "python_time_s": python_time,
                "cpp_positions_per_s": cpp_stats["positions_per_second"],
                "python_positions_per_s": py_stats["positions_per_second"],
                "speedup_cpp_over_python": (python_time / cpp_time) if cpp_time > 0 else float("inf"),
                "alignment_mean_l1": alignment_dict.get("mean_l1", 0.0),
                "alignment_top_match_pct": alignment_dict.get("top_match_pct", 0.0),
            }
        )

    if csv_rows:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
