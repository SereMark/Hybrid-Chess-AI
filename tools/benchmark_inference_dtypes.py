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
    summarize_measurements,
    write_reports,
    render_markdown_table,
)

prepare_extension_import_paths()

from network import ChessNet, INPUT_PLANES


DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def run_benchmark(
    device_name: str,
    dtypes: Sequence[str],
    batch_size: int,
    repeats: int,
    warmup: int,
    seed: int,
) -> list[Measurement]:
    try:
        device = torch.device(device_name)
    except Exception:
        raise SystemExit(f"Invalid device: {device_name}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    measurements: list[Measurement] = []

    for dtype_name in dtypes:
        torch_dtype = DTYPE_MAP.get(dtype_name.lower())
        if torch_dtype is None:
            print(f"[skip] Unknown dtype '{dtype_name}'")
            continue

        model = ChessNet()
        try:
            model = model.to(device=device, dtype=torch_dtype)
        except Exception as exc:
            print(f"[skip] Unable to move model to {dtype_name} on {device_name}: {exc}")
            continue

        model.eval()
        model.requires_grad_(False)

        try:
            batch = torch.randn(batch_size, INPUT_PLANES, 8, 8, device=device, dtype=torch_dtype)
        except Exception as exc:
            print(f"[skip] Unable to allocate batch for dtype {dtype_name}: {exc}")
            continue

        @torch.no_grad()
        def _forward() -> None:
            policy, value = model(batch)
            if policy.device.type == "cuda":
                torch.cuda.synchronize(device)
            elif policy.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            _ = float(policy.sum().item() + value.sum().item())

        measurement = benchmark_callable(
            f"{device_name}-{dtype_name}",
            _forward,
            warmup=warmup,
            repeat=repeats,
        )
        mean = statistics.fmean(measurement.samples)
        measurement.extras.update(
            {
                "device": device_name,
                "dtype": dtype_name,
                "batch_size": batch_size,
                "positions_per_sec": (batch_size / mean) if mean > 0 else 0.0,
            }
        )
        measurements.append(measurement)

    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ChessNet inference throughput across data types.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to benchmark.")
    parser.add_argument(
        "--dtypes",
        nargs="*",
        default=["float32", "bfloat16", "float16"],
        help="Data types to test (subset of float32/bfloat16/float16).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for each forward pass.")
    parser.add_argument("--repeats", type=int, default=12, help="Timed iterations per dtype.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per dtype.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for synthetic inputs.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "inference_dtypes.csv",
        help="Write summary CSV.",
    )
    args = parser.parse_args()

    measurements = run_benchmark(args.device, args.dtypes, args.batch_size, args.repeats, args.warmup, args.seed)

    if not measurements:
        raise SystemExit("No measurements collected; all dtypes may have been skipped.")

    report = summarize_measurements(
        measurements,
        metadata={
            "device": args.device,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "seed": args.seed,
        },
    )

    print("Dtype inference benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
