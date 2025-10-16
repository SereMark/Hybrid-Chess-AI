from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Sequence

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

from network import ChessNet, INPUT_PLANES


def available_devices(explicit: Sequence[str] | None = None) -> list[str]:
    """Return the list of torch devices to exercise."""

    if explicit:
        return list(dict.fromkeys(explicit))

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def run_benchmark(batch_size: int, repeats: int, warmup: int, devices: Sequence[str], seed: int) -> list[Measurement]:
    """Measure forward-pass latency for ChessNet across devices."""

    measurements: list[Measurement] = []

    for device_name in devices:
        try:
            device = torch.device(device_name)
        except Exception:
            continue

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = ChessNet().to(device)
        model.eval()
        model.requires_grad_(False)

        batch = torch.randn(batch_size, INPUT_PLANES, 8, 8, device=device, dtype=torch.float32)

        @torch.no_grad()
        def _forward() -> None:
            policy, value = model(batch)
            # Force synchronisation for CUDA/MPS
            if policy.device.type == "cuda":
                torch.cuda.synchronize(device)
            elif policy.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            _ = float(policy.sum().item() + value.sum().item())

        measurement = benchmark_callable(
            f"inference-{device_name}",
            _forward,
            warmup=warmup,
            repeat=repeats,
        )
        mean = statistics.fmean(measurement.samples)
        throughput = batch_size / mean if mean > 0 else 0.0
        measurement.extras.update(
            {
                "device": device_name,
                "batch_size": batch_size,
                "positions_per_sec": throughput,
                "dtype": "float32",
            }
        )
        measurements.append(measurement)

    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ChessNet inference throughput across devices.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for forward passes.")
    parser.add_argument("--repeats", type=int, default=10, help="Timed iterations per device.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per device.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for synthetic inputs.")
    parser.add_argument(
        "--devices",
        nargs="*",
        help="Optional explicit torch device list (e.g. cpu cuda:0). Defaults to detected devices.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "inference_devices.csv",
        help="Path to write a CSV summary.",
    )
    args = parser.parse_args()

    devices = available_devices(args.devices)
    if not devices:
        raise SystemExit("No torch devices available to benchmark.")

    measurements = run_benchmark(args.batch_size, args.repeats, args.warmup, devices, args.seed)
    report = summarize_measurements(
        measurements,
        metadata={
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "seed": args.seed,
        },
    )

    print("Device inference benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
