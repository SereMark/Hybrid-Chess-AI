# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bench_common import (
    Measurement,
    benchmark_callable,
    prepare_extension_import_paths,
    render_markdown_table,
    summarize_measurements,
    write_reports,
)

prepare_extension_import_paths()

from inference import BatchedEvaluator  # noqa: E402
from self_play import SelfPlayEngine  # noqa: E402


def _sample_fens(engine: SelfPlayEngine, samples: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return [engine.sample_start_fen(rng) for _ in range(samples)]


def _prepare_engine(device: torch.device) -> SelfPlayEngine:
    evaluator = BatchedEvaluator(device)
    engine = SelfPlayEngine(evaluator)
    engine.set_num_workers(1)
    engine._curriculum_prob = 0.0
    return engine


def evaluate_scenario(device: torch.device, samples: int, seed: int, use_book: bool) -> tuple[list[str], Measurement]:
    engine = _prepare_engine(device)
    if not use_book:
        engine._opening_book = []
        engine._opening_cumulative = None

    def _run() -> None:
        _sample_fens(engine, samples, seed)

    try:
        measurement = benchmark_callable(
            "book-enabled" if use_book else "book-disabled",
            _run,
            warmup=1,
            repeat=5,
        )
        fens = _sample_fens(engine, samples, seed)
        unique_count = len(set(fens))
        baseline_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        baseline_freq = sum(fen == baseline_start for fen in fens)
        measurement.extras.update(
            {
                "use_opening_book": use_book,
                "sample_count": samples,
                "unique_fens": unique_count,
                "baseline_frequency": baseline_freq,
                "baseline_fraction": baseline_freq / samples if samples else 0.0,
            }
        )
        return fens, measurement
    finally:
        engine.evaluator.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the effect of the opening book on start position diversity."
    )
    parser.add_argument("--samples", type=int, default=512, help="Number of starting positions to sample.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for sampling.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for evaluator instantiation.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "opening_book.csv",
        help="Path for CSV summary.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    fens_with_book, measurement_book = evaluate_scenario(device, args.samples, args.seed, use_book=True)
    fens_without_book, measurement_no_book = evaluate_scenario(device, args.samples, args.seed, use_book=False)

    measurements = [measurement_book, measurement_no_book]
    report = summarize_measurements(
        measurements,
        metadata={
            "sample_count": args.samples,
            "seed": args.seed,
            "device": args.device,
            "unique_with_book": len(set(fens_with_book)),
            "unique_without_book": len(set(fens_without_book)),
        },
    )

    print("Opening book benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
