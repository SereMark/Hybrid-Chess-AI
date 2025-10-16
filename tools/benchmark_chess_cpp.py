# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import csv
import json
import platform
from importlib import import_module
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, cast

from _bench_common import import_chesscore, import_python_chess, prepare_extension_import_paths

prepare_extension_import_paths()
ccore: Any = import_chesscore()
chess: Any = import_python_chess()
chess_pgn = import_module("chess.pgn")

# --------------------------------------------------------------------------- helpers


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if not is_dataclass(obj):
        raise TypeError("Expected dataclass instance")
    return cast(dict[str, Any], json.loads(json.dumps(asdict(obj))))  # type: ignore[arg-type]


def render_markdown_report(report: dict[str, Any]) -> str:
    dataset = cast(dict[str, Any], report["dataset"])
    timings = cast(dict[str, Any], dataset["timings"])
    move_stats = cast(dict[str, Any], dataset["move_statistics"])
    correctness = cast(dict[str, Any], dataset["correctness"])
    lines: list[str] = []
    lines.append("# Chess Engine Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Python: {report['python_version']}")
    lines.append(f"Platform: {report['platform']}")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Name: {dataset['name']}")
    lines.append(f"- Positions: {dataset['positions']}")
    lines.append(f"- Loops per timing run: {dataset['loops']}")
    lines.append(f"- Timing repetitions: {dataset['repetitions']}")
    extras = cast(dict[str, Any], dataset.get("extras", {}))
    lines.append(f"- Description: {extras.get('description', 'N/A')}")
    lines.append("")
    lines.append("## Move Statistics")
    lines.append("")
    lines.append(f"- Mean legal moves: {move_stats['mean_moves']:.2f}")
    lines.append(f"- Median: {move_stats['median_moves']:.2f}")
    lines.append(f"- Min / Max: {move_stats['min_moves']} / {move_stats['max_moves']}")
    lines.append(f"- Std Dev: {move_stats['stdev_moves']:.2f}")
    lines.append(
        f"- Quartiles (Q1 / Q2 / Q3): "
        f"{move_stats['quartiles'][0]:.2f} / {move_stats['quartiles'][1]:.2f} / {move_stats['quartiles'][2]:.2f}"
    )
    lines.append("")
    lines.append("## Correctness")
    lines.append("")
    lines.append(f"- Positions checked: {correctness['total_positions']}")
    lines.append(f"- Mismatches: {correctness['mismatch_count']}")
    examples = cast(list[dict[str, Any]], correctness["examples"])
    if examples:
        lines.append("- Example mismatches:")
        for example in examples:
            lines.append(f"  - FEN: `{example['fen']}`")
            if example["missing_in_chesscore"]:
                missing_moves = cast(Iterable[str], example["missing_in_chesscore"])
                missing = ", ".join(missing_moves)
                lines.append(f"    Missing in chesscore: {missing}")
            if example["extra_in_chesscore"]:
                extra_moves = cast(Iterable[str], example["extra_in_chesscore"])
                extra = ", ".join(extra_moves)
                lines.append(f"    Extra in chesscore: {extra}")
    lines.append("")
    lines.append("## Timing Summary")
    lines.append("")
    for engine, summary in timings.items():
        lines.append(f"### {engine}")
        lines.append("")
        lines.append(f"- Mean time: {summary['mean_s']:.4f} s")
        lines.append(f"- Std Dev: {summary['stdev_s']:.4f} s")
        lines.append(f"- Min / Max: {summary['min_s']:.4f} / {summary['max_s']:.4f} s")
        lines.append(f"- Mean positions per second: {summary['mean_positions_per_second']:.0f}")
        lines.append(f"- Mean moves per second: {summary['mean_moves_per_second']:.0f}")
        lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class TimingSample:
    elapsed_s: float
    total_moves: int


@dataclass(frozen=True)
class TimingSummary:
    engine: str
    loops: int
    positions: int
    samples: list[TimingSample]
    mean_s: float
    stdev_s: float
    min_s: float
    max_s: float
    mean_positions_per_second: float
    min_positions_per_second: float
    max_positions_per_second: float
    mean_moves_per_second: float
    min_moves_per_second: float
    max_moves_per_second: float


@dataclass(frozen=True)
class CorrectnessReport:
    total_positions: int
    mismatch_count: int
    examples: list[dict[str, object]]


@dataclass(frozen=True)
class MoveStatistics:
    sample_size: int
    unique_positions: int
    min_moves: int
    max_moves: int
    mean_moves: float
    median_moves: float
    stdev_moves: float
    quartiles: tuple[float, float, float]
    stage_means: dict[str, float]


@dataclass(frozen=True)
class SingleGameMetrics:
    game_index: int
    plies: int
    captures: int
    promotions: int
    checks_given: int
    result: str
    termination: str
    finished: bool


@dataclass(frozen=True)
class GameAggregate:
    games: int
    finished_games: int
    unfinished_games: int
    avg_plies: float
    median_plies: float
    min_plies: int
    max_plies: int
    avg_captures: float
    avg_promotions: float
    avg_checks: float
    result_counts: dict[str, int]
    termination_counts: dict[str, int]


@dataclass(frozen=True)
class DatasetReport:
    name: str
    positions: int
    loops: int
    repetitions: int
    move_statistics: MoveStatistics
    correctness: CorrectnessReport
    timings: dict[str, TimingSummary]
    extras: dict[str, object]


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    positions: int
    max_random_plies: int
    loops: int
    repetitions: int
    seed: int
    description: str = "Random playout positions"
    mode: str = "random_positions"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "positions": self.positions,
            "max_random_plies": self.max_random_plies,
            "loops": self.loops,
            "repetitions": self.repetitions,
            "seed": self.seed,
            "description": self.description,
            "mode": self.mode,
        }


TEMPLATE_SCENARIOS: dict[str, list[dict]] = {
    "baseline": [
        {
            "name": "baseline_small",
            "positions": 200,
            "max_random_plies": 80,
            "loops": 80,
            "repetitions": 3,
            "seed": 9001,
        },
        {
            "name": "baseline_medium",
            "positions": 400,
            "max_random_plies": 120,
            "loops": 100,
            "repetitions": 3,
            "seed": 42,
        },
        {
            "name": "baseline_deep",
            "positions": 600,
            "max_random_plies": 200,
            "loops": 150,
            "repetitions": 3,
            "seed": 1337,
        },
    ],
    "quick": [
        {"name": "quick_small", "positions": 100, "max_random_plies": 60, "loops": 40, "repetitions": 2, "seed": 2025}
    ],
}

# --------------------------------------------------------------------------- core utils


def generate_random_positions(count: int, max_random_plies: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens: list[str] = []
    for _ in range(count):
        board = chess.Board()
        plies = rng.randint(0, max_random_plies)
        for _ in range(plies):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(rng.choice(legal_moves))
        fens.append(board.fen())
    return fens


def moves_to_uci(moves: Iterable[chess.Move]) -> list[str]:
    return sorted(move.uci() for move in moves)


def chesscore_moves_to_uci(moves: Sequence[ccore.Move]) -> list[str]:
    return sorted(ccore.uci_of_move(move) for move in moves)


def verify_correctness(fens: Sequence[str], *, limit: int | None = None) -> list[dict[str, object]]:
    discrepancies: list[dict[str, object]] = []
    pos = ccore.Position()
    for fen in fens:
        pos.from_fen(fen)
        ccore_moves = chesscore_moves_to_uci(pos.legal_moves())
        board = chess.Board(fen)
        py_moves = moves_to_uci(board.legal_moves)
        if ccore_moves != py_moves:
            discrepancies.append(
                {
                    "fen": fen,
                    "missing_in_chesscore": sorted(set(py_moves) - set(ccore_moves)),
                    "extra_in_chesscore": sorted(set(ccore_moves) - set(py_moves)),
                }
            )
            if limit is not None and len(discrepancies) >= limit:
                break
    return discrepancies


def benchmark_chesscore(fens: Sequence[str], loops: int) -> tuple[float, int]:
    pos = ccore.Position()
    total_moves = 0
    start = time.perf_counter()
    for _ in range(loops):
        for fen in fens:
            pos.from_fen(fen)
            moves = pos.legal_moves()
            total_moves += len(moves)
    elapsed = time.perf_counter() - start
    return elapsed, total_moves


def benchmark_python_chess(fens: Sequence[str], loops: int) -> tuple[float, int]:
    total_moves = 0
    start = time.perf_counter()
    for _ in range(loops):
        for fen in fens:
            board = chess.Board(fen)
            moves = list(board.legal_moves)
            total_moves += len(moves)
    elapsed = time.perf_counter() - start
    return elapsed, total_moves


def classify_stage(board: chess.Board) -> str:
    pieces = len(board.piece_map())
    ply_count = board.fullmove_number * 2 - (0 if board.turn == chess.WHITE else 1)
    if ply_count <= 14:
        return "opening"
    if pieces <= 10 or ply_count >= 70:
        return "endgame"
    return "middlegame"


def collect_move_statistics(fens: Sequence[str]) -> MoveStatistics:
    counts: list[int] = []
    stage_buckets: dict[str, list[int]] = {"opening": [], "middlegame": [], "endgame": []}
    for fen in fens:
        board = chess.Board(fen)
        num_moves = sum(1 for _ in board.legal_moves)
        counts.append(num_moves)
        stage_buckets.setdefault(classify_stage(board), []).append(num_moves)
    sample_size = len(counts)
    unique_positions = len(set(fens))
    if sample_size == 0:
        return MoveStatistics(
            sample_size=0,
            unique_positions=0,
            min_moves=0,
            max_moves=0,
            mean_moves=0.0,
            median_moves=0.0,
            stdev_moves=0.0,
            quartiles=(0.0, 0.0, 0.0),
            stage_means={k: 0.0 for k in stage_buckets},
        )
    mean_moves = statistics.fmean(counts)
    median_moves = statistics.median(counts)
    stdev_moves = statistics.stdev(counts) if sample_size > 1 else 0.0
    if sample_size >= 2:
        q1, q2, q3 = statistics.quantiles(counts, n=4, method="inclusive")
    else:
        q1 = q2 = q3 = float(counts[0])
    stage_means = {k: (statistics.fmean(v) if v else 0.0) for k, v in stage_buckets.items()}
    return MoveStatistics(
        sample_size=sample_size,
        unique_positions=unique_positions,
        min_moves=min(counts),
        max_moves=max(counts),
        mean_moves=mean_moves,
        median_moves=float(median_moves),
        stdev_moves=stdev_moves,
        quartiles=(float(q1), float(q2), float(q3)),
        stage_means=stage_means,
    )


def make_correctness_report(fens: Sequence[str], *, example_limit: int = 5) -> CorrectnessReport:
    mismatches = verify_correctness(fens, limit=example_limit)
    return CorrectnessReport(
        total_positions=len(fens),
        mismatch_count=len(mismatches),
        examples=mismatches[:5],
    )


def build_timing_summary(
    engine: str,
    bench_fn: Callable[[Sequence[str], int], tuple[float, int]],
    fens: Sequence[str],
    loops: int,
    repetitions: int,
) -> TimingSummary:
    samples: list[TimingSample] = []
    for _ in range(max(1, repetitions)):
        elapsed, total_moves = bench_fn(fens, loops)
        samples.append(TimingSample(elapsed_s=elapsed, total_moves=total_moves))
    elapsed_values = [s.elapsed_s for s in samples]
    positions = len(fens) * loops
    mean_s = statistics.fmean(elapsed_values)
    stdev_s = statistics.stdev(elapsed_values) if len(samples) > 1 else 0.0
    min_s = min(elapsed_values)
    max_s = max(elapsed_values)
    pos_rates = [positions / s if s > 0 else float("inf") for s in elapsed_values]
    move_rates = [s.total_moves / s.elapsed_s if s.elapsed_s > 0 else float("inf") for s in samples]
    return TimingSummary(
        engine=engine,
        loops=loops,
        positions=positions,
        samples=samples,
        mean_s=mean_s,
        stdev_s=stdev_s,
        min_s=min_s,
        max_s=max_s,
        mean_positions_per_second=statistics.fmean(pos_rates),
        min_positions_per_second=min(pos_rates),
        max_positions_per_second=max(pos_rates),
        mean_moves_per_second=statistics.fmean(move_rates),
        min_moves_per_second=min(move_rates),
        max_moves_per_second=max(move_rates),
    )


PROMOTION_MAP = {
    chess.QUEEN: ccore.Piece.QUEEN,
    chess.ROOK: ccore.Piece.ROOK,
    chess.BISHOP: ccore.Piece.BISHOP,
    chess.KNIGHT: ccore.Piece.KNIGHT,
}


def convert_move_to_chesscore(move: chess.Move) -> ccore.Move:
    if move.promotion is None:
        return ccore.Move(move.from_square, move.to_square)
    promo_piece = PROMOTION_MAP.get(move.promotion, ccore.Piece.QUEEN)
    return ccore.Move(move.from_square, move.to_square, promo_piece)


def aggregate_games(metrics: Sequence[SingleGameMetrics]) -> GameAggregate:
    games = len(metrics)
    if games == 0:
        return GameAggregate(
            games=0,
            finished_games=0,
            unfinished_games=0,
            avg_plies=0.0,
            median_plies=0.0,
            min_plies=0,
            max_plies=0,
            avg_captures=0.0,
            avg_promotions=0.0,
            avg_checks=0.0,
            result_counts={},
            termination_counts={},
        )
    plies = [m.plies for m in metrics]
    captures = [m.captures for m in metrics]
    promotions = [m.promotions for m in metrics]
    checks = [m.checks_given for m in metrics]
    result_counts = Counter(m.result for m in metrics)
    termination_counts = Counter(m.termination for m in metrics)
    finished_games = sum(1 for m in metrics if m.finished)
    return GameAggregate(
        games=games,
        finished_games=finished_games,
        unfinished_games=games - finished_games,
        avg_plies=statistics.fmean(plies),
        median_plies=float(statistics.median(plies)),
        min_plies=min(plies),
        max_plies=max(plies),
        avg_captures=statistics.fmean(captures),
        avg_promotions=statistics.fmean(promotions),
        avg_checks=statistics.fmean(checks),
        result_counts=dict(result_counts),
        termination_counts=dict(termination_counts),
    )


def simulate_random_games(
    num_games: int, max_plies: int, seed: int
) -> tuple[list[str], list[SingleGameMetrics], GameAggregate, list[dict[str, object]]]:
    rng = random.Random(seed)
    fens: list[str] = []
    metrics: list[SingleGameMetrics] = []
    mismatches: list[dict[str, object]] = []
    for game_index in range(num_games):
        board = chess.Board()
        pos = ccore.Position()
        captures = promotions = checks = plies_played = 0
        finished = False
        for ply in range(1, max_plies + 1):
            legal_moves = list(board.legal_moves)
            fen_before = board.fen()
            if not legal_moves:
                finished = True
                break
            move = rng.choice(legal_moves)
            move_uci = move.uci()
            fens.append(fen_before)
            ccore_moves = chesscore_moves_to_uci(pos.legal_moves())
            if move_uci not in ccore_moves:
                mismatches.append(
                    {
                        "game": game_index,
                        "ply": ply,
                        "fen": fen_before,
                        "move": move_uci,
                        "issue": "move missing in chesscore legal set",
                    }
                )
            pos.make_move(convert_move_to_chesscore(move))
            if board.is_capture(move):
                captures += 1
            if move.promotion is not None:
                promotions += 1
            board.push(move)
            plies_played += 1
            if board.is_check():
                checks += 1
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                finished = True
                break
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome else "*"
        termination = outcome.termination.name.lower() if outcome else ("max_plies" if not finished else "unknown")
        metrics.append(
            SingleGameMetrics(
                game_index=game_index,
                plies=plies_played,
                captures=captures,
                promotions=promotions,
                checks_given=checks,
                result=result,
                termination=termination,
                finished=bool(outcome),
            )
        )
    aggregate = aggregate_games(metrics)
    return fens, metrics, aggregate, mismatches


def read_pgn_games(
    pgn_texts: Sequence[str],
) -> tuple[list[str], list[str], list[SingleGameMetrics], GameAggregate, list[dict[str, object]], list[dict[str, str]]]:
    import io

    fens: list[str] = []
    played_fens: list[str] = []
    metrics: list[SingleGameMetrics] = []
    mismatches: list[dict[str, object]] = []
    meta: list[dict[str, str]] = []
    for game_index, pgn_text in enumerate(pgn_texts):
        handle = io.StringIO(pgn_text)
        game = chess_pgn.read_game(handle)
        if game is None:
            continue
        headers = dict(game.headers)
        meta.append(headers)
        if headers.get("SetUp") == "1" and headers.get("FEN"):
            board = chess.Board(headers["FEN"])
            pos = ccore.Position()
            pos.from_fen(headers["FEN"])
        else:
            board = chess.Board()
            pos = ccore.Position()
        captures = promotions = checks = 0
        plies_played = 0
        for move in game.mainline_moves():
            fen_before = board.fen()
            fens.append(fen_before)
            move_uci = move.uci()
            ccore_moves = chesscore_moves_to_uci(pos.legal_moves())
            if move_uci not in ccore_moves:
                mismatches.append(
                    {
                        "game": game_index,
                        "ply": plies_played + 1,
                        "fen": fen_before,
                        "move": move_uci,
                        "issue": "move missing in chesscore legal set",
                    }
                )
            pos.make_move(convert_move_to_chesscore(move))
            if board.is_capture(move):
                captures += 1
            if move.promotion is not None:
                promotions += 1
            board.push(move)
            played_fens.append(board.fen())
            plies_played += 1
            if board.is_check():
                checks += 1
        outcome = board.outcome(claim_draw=True)
        result = headers.get("Result") or (outcome.result() if outcome else "*")
        termination_raw = headers.get("Termination") or (outcome.termination.name if outcome else "unknown")
        termination = termination_raw.lower() if termination_raw else "unknown"
        metrics.append(
            SingleGameMetrics(
                game_index=game_index,
                plies=plies_played,
                captures=captures,
                promotions=promotions,
                checks_given=checks,
                result=result,
                termination=termination,
                finished=result in {"1-0", "0-1", "1/2-1/2"},
            )
        )
    aggregate = aggregate_games(metrics)
    return fens, played_fens, metrics, aggregate, mismatches, meta


def build_dataset(
    name: str,
    fens: Sequence[str],
    loops: int,
    repetitions: int,
    description: str,
    extra_payload: dict[str, object] | None = None,
    precomputed_move_stats: MoveStatistics | None = None,
    precomputed_correctness: CorrectnessReport | None = None,
) -> DatasetReport:
    move_stats = precomputed_move_stats or collect_move_statistics(fens)
    correctness = precomputed_correctness or make_correctness_report(fens)
    timings = {
        "chesscore": build_timing_summary("chesscore", benchmark_chesscore, fens, loops, repetitions),
        "python-chess": build_timing_summary("python-chess", benchmark_python_chess, fens, loops, repetitions),
    }
    extras = dict(extra_payload or {})
    extras.setdefault("description", description)
    return DatasetReport(
        name=name,
        positions=len(fens),
        loops=loops,
        repetitions=repetitions,
        move_statistics=move_stats,
        correctness=correctness,
        timings=timings,
        extras=extras,
    )


def format_rate(count: int, elapsed: float) -> str:
    return "∞" if elapsed <= 0.0 else f"{count / elapsed:,.0f} per second"


DEFAULT_POSITIONS = [200]
DEFAULT_MAX_RANDOM_PLIES = [40]
DEFAULT_LOOPS = [50]
DEFAULT_REPETITIONS = [3]
DEFAULT_SEEDS = [2025]

# --------------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, nargs="*", default=None, help="Number of random FENs per scenario")
    parser.add_argument("--max-random-plies", type=int, nargs="*", default=None, help="Upper bound on random plies")
    parser.add_argument("--loops", type=int, nargs="*", default=None, help="Benchmark loops per scenario")
    parser.add_argument("--repetitions", type=int, nargs="*", default=None, help="Timing repetitions per scenario")
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=None, help="Seeds to use for random position generation"
    )
    parser.add_argument("--scenario-name", type=str, default="random", help="Base name for ad-hoc scenarios")
    parser.add_argument("--template", choices=sorted(TEMPLATE_SCENARIOS.keys()), help="Use a preset suite of scenarios")

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "benchmark_summary.csv",
        help="Path for the CSV timing summary",
    )
    args = parser.parse_args()

    positions_list = list(args.positions) if args.positions else list(DEFAULT_POSITIONS)
    maxplies_list = list(args.max_random_plies) if args.max_random_plies else list(DEFAULT_MAX_RANDOM_PLIES)
    loops_list = list(args.loops) if args.loops else list(DEFAULT_LOOPS)
    reps_list = list(args.repetitions) if args.repetitions else list(DEFAULT_REPETITIONS)
    seeds_list = list(args.seeds) if args.seeds else list(DEFAULT_SEEDS)

    if args.template:
        if args.positions or args.max_random_plies or args.loops or args.repetitions or args.seeds:
            raise SystemExit("When --template is used, omit manual scenario arguments.")
        raw_scenarios = TEMPLATE_SCENARIOS[args.template]
    else:
        if not positions_list:
            raise SystemExit("Provide at least one --positions value or use --template.")
        raw_scenarios = []
        max_len = max(len(positions_list), len(maxplies_list), len(loops_list), len(reps_list), len(seeds_list))
        for idx in range(max_len):
            raw_scenarios.append(
                {
                    "name": f"{args.scenario_name}_{idx}" if max_len > 1 else args.scenario_name,
                    "positions": positions_list[idx % len(positions_list)],
                    "max_random_plies": maxplies_list[idx % len(maxplies_list)],
                    "loops": loops_list[idx % len(loops_list)],
                    "repetitions": reps_list[idx % len(reps_list)],
                    "seed": seeds_list[idx % len(seeds_list)],
                }
            )

    if any(cfg["positions"] <= 0 for cfg in raw_scenarios):
        raise SystemExit("--positions must all be positive")
    if any(cfg["max_random_plies"] < 0 for cfg in raw_scenarios):
        raise SystemExit("--max-random-plies must be non-negative")
    if any(cfg["loops"] <= 0 for cfg in raw_scenarios):
        raise SystemExit("--loops must be positive")
    if any(cfg["repetitions"] <= 0 for cfg in raw_scenarios):
        raise SystemExit("--repetitions must be positive")

    scenarios = [BenchmarkScenario(**cfg) for cfg in raw_scenarios]

    reports: list[dict[str, Any]] = []
    for scenario in scenarios:
        fens = generate_random_positions(scenario.positions, scenario.max_random_plies, scenario.seed)
        msg = (
            f"[{scenario.name}] Generated {len(fens)} positions up to "
            f"{scenario.max_random_plies} plies (seed={scenario.seed})."
        )
        print(msg)

        correctness = make_correctness_report(fens, example_limit=10)
        discrepancies = correctness.examples
        if discrepancies:
            print(f"[{scenario.name}] Correctness: {correctness.mismatch_count} mismatches.")
            for idx, mismatch in enumerate(discrepancies[:5], start=1):
                print(f"  [{idx}] FEN: {mismatch['fen']}")
                extra = cast(Iterable[str], mismatch["extra_in_chesscore"])
                missing = cast(Iterable[str], mismatch["missing_in_chesscore"])
                if extra:
                    print(f"      Extra in chesscore : {', '.join(extra)}")
                if missing:
                    print(f"      Missing in chesscore: {', '.join(missing)}")
            if len(discrepancies) > 5:
                print(f"  ...and {len(discrepancies) - 5} more.")
        else:
            print(f"[{scenario.name}] Correctness: all positions matched.")

        positions_evaluated = len(fens) * scenario.loops
        chesscore_time, chesscore_moves = benchmark_chesscore(fens, scenario.loops)
        python_time, python_moves = benchmark_python_chess(fens, scenario.loops)

        print("\nPerformance (legal move generation):")
        print(
            f"  chesscore    : {chesscore_time:.3f} s for {positions_evaluated:,} positions, "
            f"{chesscore_moves:,} moves -> {format_rate(positions_evaluated, chesscore_time)}"
        )
        print(
            f"  python-chess : {python_time:.3f} s for {positions_evaluated:,} positions, "
            f"{python_moves:,} moves -> {format_rate(positions_evaluated, python_time)}"
        )
        if chesscore_time > 0 and python_time > 0:
            print(f"  Speedup      : {python_time / chesscore_time:.2f}x (python-chess / chesscore)")

        dataset = build_dataset(
            scenario.name,
            fens,
            scenario.loops,
            scenario.repetitions,
            description=scenario.description,
            precomputed_move_stats=collect_move_statistics(fens),
            precomputed_correctness=correctness,
        )

        reports.append(
            {
                "scenario": scenario.to_dict(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version,
                "platform": platform.platform(),
                "dataset": dataclass_to_dict(dataset),
            }
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_rows: list[dict[str, object]] = []
        for report in reports:
            report_dict = report
            scenario_dict = cast(dict[str, Any], report_dict["scenario"])
            dataset_dict = cast(dict[str, Any], report_dict["dataset"])
            timings = cast(dict[str, Any], dataset_dict["timings"])
            correctness_dict = cast(dict[str, Any], dataset_dict["correctness"])
            csv_rows.append(
                {
                    "scenario": scenario_dict["name"],
                    "positions": scenario_dict["positions"],
                    "loops": scenario_dict["loops"],
                    "repetitions": scenario_dict["repetitions"],
                    "mismatches": correctness_dict["mismatch_count"],
                    "chesscore_time": timings["chesscore"]["mean_s"],
                    "python_time": timings["python-chess"]["mean_s"],
                    "speedup": (
                        timings["python-chess"]["mean_s"] / timings["chesscore"]["mean_s"]
                        if timings["chesscore"]["mean_s"] > 0
                        else float("inf")
                    ),
                }
            )
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Wrote CSV summary to {args.output_csv}")

    if any(
        cast(dict[str, Any], cast(dict[str, Any], report["dataset"])["correctness"])["mismatch_count"] > 0
        for report in reports
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
