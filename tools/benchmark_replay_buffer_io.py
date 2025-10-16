from __future__ import annotations

import argparse
import statistics
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

from encoder import INPUT_PLANES
from replay_buffer import ReplayBuffer
from augmentation import POLICY_OUTPUT


def random_entry() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    state = np.random.randint(0, 256, size=(INPUT_PLANES, 8, 8), dtype=np.uint8)
    policy_size = np.random.randint(1, 32)
    indices = np.random.choice(POLICY_OUTPUT, size=policy_size, replace=False).astype(np.int32)
    counts = np.random.randint(1, 10, size=policy_size).astype(np.uint16)
    value = int(np.random.randint(-128, 128))
    return state, indices, counts, value


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure replay buffer insert/sample throughput.")
    parser.add_argument("--capacity", type=int, default=4096, help="Replay buffer capacity.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for sampling.")
    parser.add_argument("--repeat", type=int, default=10, help="Timed iterations per scenario.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per scenario.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "replay_buffer.csv",
        help="Write CSV summary.",
    )
    args = parser.parse_args()

    buffer = ReplayBuffer(capacity=args.capacity, planes=INPUT_PLANES, height=8, width=8)

    entries = [random_entry() for _ in range(args.capacity)]

    def fill_buffer() -> None:
        buffer.clear()
        for state, indices, counts, value in entries:
            buffer.push(state, indices, counts, value)

    def sample_batches() -> None:
        if buffer.size < args.capacity:
            fill_buffer()
        for _ in range(max(1, args.capacity // args.batch_size)):
            _ = buffer.sample(args.batch_size, recent_ratio=0.5, recent_window_frac=0.25)

    fill_measure = benchmark_callable("buffer-fill", fill_buffer, warmup=args.warmup, repeat=args.repeat)
    sample_measure = benchmark_callable("buffer-sample", sample_batches, warmup=args.warmup, repeat=args.repeat)

    fill_mean = statistics.fmean(fill_measure.samples)
    sample_mean = statistics.fmean(sample_measure.samples)

    fill_measure.extras.update(
        {
            "capacity": args.capacity,
            "operations_per_sec": (args.capacity / fill_mean) if fill_mean > 0 else 0.0,
        }
    )
    sample_measure.extras.update(
        {
            "capacity": args.capacity,
            "batch_size": args.batch_size,
            "operations_per_sec": ((args.capacity // args.batch_size) / sample_mean) if sample_mean > 0 else 0.0,
        }
    )

    measurements = [fill_measure, sample_measure]
    report = summarize_measurements(
        measurements,
        metadata={
            "capacity": args.capacity,
            "batch_size": args.batch_size,
            "repeat": args.repeat,
            "warmup": args.warmup,
        },
    )

    print("Replay buffer benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
