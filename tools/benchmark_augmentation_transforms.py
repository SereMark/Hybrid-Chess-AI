from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bench_common import (
    Measurement,
    benchmark_callable,
    prepare_extension_import_paths,
    summarize_measurements,
    write_reports,
    render_markdown_table,
)

prepare_extension_import_paths()

from augmentation import Augment, POLICY_OUTPUT
from encoder import INPUT_PLANES


def make_random_batch(batch_size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    states = [np.random.rand(INPUT_PLANES, 8, 8).astype(np.float32) for _ in range(batch_size)]
    policies = [np.random.rand(POLICY_OUTPUT).astype(np.float32) for _ in range(batch_size)]
    return states, policies


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure augmentation transform latency.")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of positions per trial.")
    parser.add_argument("--repeats", type=int, default=20, help="Timed iterations per transform.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per transform.")
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=["mirror", "rot180", "vflip_cs"],
        help="Transforms to benchmark (mirror, rot180, vflip_cs).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "augmentation_transforms.csv",
        help="Write CSV summary.",
    )
    args = parser.parse_args()

    base_states, base_policies = make_random_batch(args.batch_size)

    measurements: list[Measurement] = []
    for transform in args.transforms:

        def _apply() -> None:
            Augment.apply(base_states, base_policies, transform)

        measurement = benchmark_callable(
            f"augment-{transform}",
            _apply,
            warmup=args.warmup,
            repeat=args.repeats,
        )
        measurement.extras.update(
            {
                "transform": transform,
                "batch_size": args.batch_size,
            }
        )
        measurements.append(measurement)

    report = summarize_measurements(
        measurements,
        metadata={
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
    )

    print("Augmentation transform benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
