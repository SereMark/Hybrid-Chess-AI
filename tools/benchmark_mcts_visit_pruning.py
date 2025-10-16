# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from _bench_common import (
    Measurement,
    benchmark_callable,
    prepare_extension_import_paths,
    render_markdown_table,
    summarize_measurements,
    write_reports,
)

prepare_extension_import_paths()

import config as C  # noqa: E402
from self_play import _policy_targets  # type: ignore[attr-defined]  # noqa: E402


class DummyMove:
    __slots__ = ("from_square", "to_square", "promotion")

    def __init__(self, from_square: int, to_square: int, promotion: int = 0) -> None:
        self.from_square = int(from_square)
        self.to_square = int(to_square)
        self.promotion = int(promotion)


def _generate_counts(seed: int, policy_size: int, samples: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.poisson(60, size=policy_size).astype(np.float32) for _ in range(samples)]


def evaluate_clamp(clamp: int, counts_samples: list[np.ndarray], moves_template: list[Any]) -> Measurement:
    original_cfg = C.MCTS
    C.MCTS = replace(C.MCTS, visit_count_clamp=clamp)

    def _run() -> None:
        for counts in counts_samples:
            _policy_targets(moves_template, counts)

    try:
        measurement = benchmark_callable(f"clamp-{clamp}", _run, warmup=1, repeat=5)

        clipped_ratios: list[float] = []
        avg_policy_lengths: list[float] = []
        for counts in counts_samples:
            indices, weights = _policy_targets(moves_template, counts)
            clipped_total = weights.sum()
            unclipped_total = float(counts.sum())
            ratio = 0.0 if unclipped_total <= 0 else 1.0 - (clipped_total / unclipped_total)
            clipped_ratios.append(ratio)
            avg_policy_lengths.append(float(len(indices)))

        measurement.extras.update(
            {
                "visit_count_clamp": clamp,
                "mean_clipped_ratio": float(np.mean(clipped_ratios)),
                "std_clipped_ratio": float(np.std(clipped_ratios)),
                "mean_policy_entries": float(np.mean(avg_policy_lengths)),
                "sample_count": len(counts_samples),
            }
        )
        return measurement
    finally:
        C.MCTS = original_cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the impact of visit count clamping on policy target construction."
    )
    parser.add_argument("--policy-size", type=int, default=256, help="Simulated number of policy moves.")
    parser.add_argument("--samples", type=int, default=128, help="Number of synthetic visit count samples.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for synthetic visit counts.")
    parser.add_argument(
        "--clamps",
        type=int,
        nargs="*",
        default=[8, 32, C.MCTS.visit_count_clamp],
        help="Clamp values to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "mcts_visit_pruning.csv",
        help="Path for CSV summary.",
    )
    args = parser.parse_args()

    counts_samples = _generate_counts(args.seed, args.policy_size, args.samples)
    moves_template: list[Any] = [
        DummyMove(idx % 64, (idx * 13 + 7) % 64, promotion=0) for idx in range(args.policy_size)
    ]

    measurements = [evaluate_clamp(clamp, counts_samples, moves_template) for clamp in sorted(set(args.clamps))]

    report = summarize_measurements(
        measurements,
        metadata={
            "policy_size": args.policy_size,
            "sample_count": args.samples,
            "seed": args.seed,
        },
    )

    print("MCTS visit pruning benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
