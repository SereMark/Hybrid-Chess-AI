from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Iterable, Sequence

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

from network import ChessNet, INPUT_PLANES


def detect_devices(explicit: Sequence[str] | None = None) -> list[str]:
    if explicit:
        return list(dict.fromkeys(explicit))
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def run_for_device(
    device_name: str,
    batch_sizes: Iterable[int],
    repeats: int,
    warmup: int,
    seed: int,
) -> list[Measurement]:
    try:
        device = torch.device(device_name)
    except Exception:
        return []

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = ChessNet().to(device)
    model.eval()
    model.requires_grad_(False)
    measurements: list[Measurement] = []

    for batch_size in batch_sizes:
        batch = torch.randn(batch_size, INPUT_PLANES, 8, 8, device=device, dtype=torch.float32)

        @torch.no_grad()
        def _forward() -> None:
            policy, value = model(batch)
            if policy.device.type == "cuda":
                torch.cuda.synchronize(device)
            elif policy.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            _ = float(policy.sum().item() + value.sum().item())

        measurement = benchmark_callable(
            f"{device_name}-batch{batch_size}",
            _forward,
            warmup=warmup,
            repeat=repeats,
        )
        mean = statistics.fmean(measurement.samples)
        measurement.extras.update(
            {
                "device": device_name,
                "batch_size": batch_size,
                "positions_per_sec": (batch_size / mean) if mean > 0 else 0.0,
                "dtype": "float32",
            }
        )
        measurements.append(measurement)

    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ChessNet inference throughput across batch sizes.")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=[1, 8, 32, 64, 128],
        help="Batch sizes to evaluate.",
    )
    parser.add_argument("--repeats", type=int, default=8, help="Timed iterations per batch size.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per batch size.")
    parser.add_argument("--devices", nargs="*", help="Optional specific devices (e.g. cpu cuda:0).")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for synthetic inputs.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "inference_batchsizes.csv",
        help="Path to write a CSV summary.",
    )
    args = parser.parse_args()

    devices = detect_devices(args.devices)
    if not devices:
        raise SystemExit("No devices available to benchmark.")

    measurements: list[Measurement] = []
    for device_name in devices:
        measurements.extend(run_for_device(device_name, args.batch_sizes, args.repeats, args.warmup, args.seed))

    report = summarize_measurements(
        measurements,
        metadata={
            "batch_sizes": list(args.batch_sizes),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "devices": devices,
            "seed": args.seed,
        },
    )

    print("Batch-size inference benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
