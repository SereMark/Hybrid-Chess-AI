# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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

from encoder import INPUT_PLANES  # noqa: E402
from network import ChessNet, POLICY_OUTPUT  # noqa: E402
from train_loop import train_step  # noqa: E402
from utils import select_autocast_dtype  # noqa: E402


def random_batch(batch_size: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    states = [np.random.randint(0, 256, size=(INPUT_PLANES, 8, 8), dtype=np.uint8) for _ in range(batch_size)]
    indices: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for _ in range(batch_size):
        move_count = np.random.randint(16, 48)
        idx = np.random.choice(POLICY_OUTPUT, size=move_count, replace=False).astype(np.int32)
        cnt = np.random.randint(1, 16, size=move_count).astype(np.uint16)
        indices.append(idx)
        counts.append(cnt)
        values.append(np.array(np.random.randint(-127, 128), dtype=np.int8))
    return states, indices, counts, values


def build_trainer(device: torch.device, amp_enabled: bool, batch_size: int) -> SimpleNamespace:
    model = ChessNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    class _Scheduler:
        def step(self) -> None:
            return None

    trainer = SimpleNamespace()
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.scheduler = _Scheduler()
    trainer.device = device
    trainer.iteration = 0
    trainer.train_batch_size = batch_size
    trainer._amp_enabled = bool(amp_enabled)
    trainer._autocast_dtype = select_autocast_dtype(device)
    trainer.scaler = torch.amp.GradScaler(enabled=amp_enabled and device.type == "cuda")
    trainer.ema = None
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the cost of a single train_step invocation.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to benchmark.")
    parser.add_argument("--batch-size", type=int, default=32, help="Synthetic batch size.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "train_step.csv",
        help="Write CSV summary.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    batch = random_batch(args.batch_size)

    trainers = {
        "amp-off": build_trainer(device, amp_enabled=False, batch_size=args.batch_size),
        "amp-on": build_trainer(device, amp_enabled=True, batch_size=args.batch_size),
    }

    measurements: list[Measurement] = []
    for label, trainer in trainers.items():

        def _train_step() -> None:
            result = train_step(trainer, batch)
            trainer.iteration += 1
            if result is None:
                raise RuntimeError("train_step returned None (gradient overflow)")

        measurement = benchmark_callable(
            f"train-step-{label}",
            _train_step,
            warmup=args.warmup,
            repeat=args.repeats,
        )
        mean = statistics.fmean(measurement.samples)
        measurement.extras.update(
            {
                "label": label,
                "device": args.device,
                "batch_size": args.batch_size,
                "steps_per_sec": (1.0 / mean) if mean > 0 else 0.0,
                "amp_enabled": label.endswith("on"),
            }
        )
        measurements.append(measurement)

    report = summarize_measurements(
        measurements,
        metadata={
            "device": args.device,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
    )

    print("Train-step benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()