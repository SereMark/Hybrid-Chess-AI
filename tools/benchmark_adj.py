# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


def _make_engine(device: torch.device) -> SelfPlayEngine:
    evaluator = BatchedEvaluator(device)
    engine = SelfPlayEngine(evaluator)
    engine.set_num_workers(1)
    engine._curriculum_prob = 0.0  # avoid curriculum noise
    return engine


def _capture_state(engine: SelfPlayEngine) -> dict[str, Any]:
    return {
        "phase": engine.adjudication_phase,
        "enabled": bool(engine.adjudication_enabled),
        "min_plies": int(engine.adjudication_min_plies),
        "value_margin": float(engine.adjudication_value_margin),
        "persist": int(engine.adjudication_persist),
        "material_margin": float(engine.adjudication_material_margin),
    }


def evaluate_iteration(device: torch.device, iteration: int) -> tuple[dict[str, Any], Measurement]:
    def _run() -> None:
        engine = _make_engine(device)
        try:
            engine.update_adjudication(iteration)
        finally:
            engine.evaluator.close()

    measurement = benchmark_callable(f"iteration-{iteration}", _run, warmup=1, repeat=5)

    engine = _make_engine(device)
    try:
        engine.update_adjudication(iteration)
        snapshot = _capture_state(engine)
    finally:
        engine.evaluator.close()
    measurement.extras.update({"iteration": iteration, **snapshot})
    return snapshot, measurement


def evaluate_schedule(device: torch.device, iterations: list[int]) -> list[Measurement]:
    measurements: list[Measurement] = []
    for iteration in iterations:
        _, measurement = evaluate_iteration(device, iteration)
        measurements.append(measurement)
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the adjudication schedule and transition timings across key iterations."
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for evaluator instantiation.")
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="*",
        default=[0, 16, 32, 64, 96, 128],
        help="Iterations to probe for adjudication state transitions.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "adjudication_schedule.csv",
        help="Path for CSV summary.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    iterations = sorted(set(args.iterations))
    measurements = evaluate_schedule(device, iterations)

    report = summarize_measurements(
        measurements,
        metadata={
            "device": args.device,
            "iterations": iterations,
        },
    )

    print("Adjudication benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
