# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from _bench_common import (
    Measurement,
    benchmark_callable,
    import_chesscore,
    import_python_chess,
    prepare_extension_import_paths,
    summarize_measurements,
    write_reports,
    render_markdown_table,
)

prepare_extension_import_paths()

chess: Any = import_python_chess()
ccore: Any = import_chesscore()


def generate_fens(count: int, max_random_plies: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens: list[str] = []
    for _ in range(count):
        board = chess.Board()
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


def uniform_evaluator(positions, moves_lists):
    policies = []
    values = []
    for moves in moves_lists:
        n = len(moves)
        policies.append([] if n == 0 else [1.0 / n] * n)
        values.append(0.0)
    return policies, values


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MCTS performance under different Dirichlet weights.")
    parser.add_argument("--positions", type=int, default=128, help="Number of random positions.")
    parser.add_argument("--max-random-plies", type=int, default=80, help="Maximum random plies per position.")
    parser.add_argument("--simulations", type=int, default=256, help="Simulations per search.")
    parser.add_argument("--max-batch", type=int, default=32, help="MCTS max_batch parameter.")
    parser.add_argument(
        "--weights", nargs="*", type=float, default=[0.0, 0.10, 0.25, 0.50], help="Dirichlet noise weights to compare."
    )
    parser.add_argument("--seed", type=int, default=2025, help="RNG seed.")
    parser.add_argument("--repeats", type=int, default=4, help="Timed iterations per weight.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per weight.")
    parser.add_argument(
        "--output-csv", type=Path, default=Path("benchmark_reports") / "mcts_dirichlet.csv", help="Write CSV summary."
    )
    args = parser.parse_args()

    fens = generate_fens(args.positions, args.max_random_plies, args.seed)

    measurements: list[Measurement] = []

    for weight in args.weights:

        def _run() -> None:
            engine = ccore.MCTS(args.simulations, 1.0, 0.3, weight)
            engine.set_c_puct_params(19652.0, 1.25)
            engine.set_fpu_reduction(0.1)
            for fen in fens:
                pos = ccore.Position()
                pos.from_fen(fen)
                engine.search_batched_legal(pos, uniform_evaluator, args.max_batch)

        measurement = benchmark_callable(
            f"dirichlet-{weight:.2f}",
            _run,
            warmup=args.warmup,
            repeat=args.repeats,
        )
        mean = measurement.summary()["mean_s"]
        measurement.extras.update(
            {
                "dirichlet_weight": weight,
                "positions": args.positions,
                "simulations": args.simulations,
                "positions_per_sec": (args.positions / mean) if mean > 0 else 0.0,
            }
        )
        measurements.append(measurement)

    report = summarize_measurements(
        measurements,
        metadata={
            "positions": args.positions,
            "max_random_plies": args.max_random_plies,
            "simulations": args.simulations,
            "max_batch": args.max_batch,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "seed": args.seed,
        },
    )

    print("MCTS Dirichlet benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
