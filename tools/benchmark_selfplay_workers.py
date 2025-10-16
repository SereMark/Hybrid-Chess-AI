from __future__ import annotations

import argparse
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

from _bench_common import (
    Measurement,
    benchmark_callable,
    prepare_extension_import_paths,
    summarize_measurements,
    write_reports,
    render_markdown_table,
)

prepare_extension_import_paths()

import config as C
from inference import BatchedEvaluator
from self_play import SelfPlayEngine


def configure_lightweight_selfplay(device: torch.device) -> None:
    """Reduce default simulation parameters for quicker benchmarks."""

    cpu_mode = device.type != "cuda"
    sims = 48 if not cpu_mode else 32
    sims_min = 24 if not cpu_mode else 16
    decay = 320 if not cpu_mode else 160

    C.MCTS = replace(
        C.MCTS,
        train_simulations=sims,
        train_simulations_min=sims_min,
        train_sim_decay_move_interval=decay,
    )
    C.SELFPLAY = replace(
        C.SELFPLAY,
        game_max_plies=60 if not cpu_mode else 40,
        temperature_moves=10 if not cpu_mode else 6,
        opening_random_moves=1,
    )


def run_benchmark(
    worker_counts: Sequence[int],
    games: int,
    repeats: int,
    warmup: int,
    device: torch.device,
    seed: int,
) -> list[Measurement]:
    configure_lightweight_selfplay(device)
    evaluator = BatchedEvaluator(device)
    engine = SelfPlayEngine(evaluator, seed=seed)
    engine._curriculum_prob = 0.0
    engine.opening_random_moves = 0
    engine.resign_enabled = False

    measurements: list[Measurement] = []

    try:
        for workers in worker_counts:
            engine.set_num_workers(workers)
            games_history: list[int] = []

            def _play() -> None:
                stats = engine.play_games(games)
                games_history.append(int(stats.get("games", 0)))

            measurement = benchmark_callable(
                f"selfplay-workers-{workers}",
                _play,
                warmup=warmup,
                repeat=repeats,
            )
            mean = statistics.fmean(measurement.samples)
            games_mean = statistics.fmean(games_history) if games_history else 0.0
            measurement.extras.update(
                {
                    "workers": workers,
                    "games_requested": games,
                    "games_mean": games_mean,
                    "games_per_sec": (games_mean / mean) if mean > 0 else 0.0,
                }
            )
            measurements.append(measurement)
    finally:
        engine.evaluator.close()
        evaluator.close()

    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SelfPlayEngine throughput across worker counts.")
    parser.add_argument("--workers", type=int, nargs="*", default=[1, 2, 4], help="Worker counts to test.")
    parser.add_argument("--games", type=int, default=4, help="Number of games per timed iteration.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed iterations per worker count.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per worker count.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for the evaluator model.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducible self-play sampling.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "selfplay_workers.csv",
        help="Write CSV summary.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    measurements = run_benchmark(args.workers, args.games, args.repeats, args.warmup, device, args.seed)

    report = summarize_measurements(
        measurements,
        metadata={
            "device": args.device,
            "games": args.games,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "seed": args.seed,
        },
    )

    print("Self-play worker benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
