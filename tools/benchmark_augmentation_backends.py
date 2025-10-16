# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
from pathlib import Path

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

from augmentation import Augment, POLICY_OUTPUT  # noqa: E402
from encoder import INPUT_PLANES  # noqa: E402


def make_random_batch(batch_size: int, seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    states = [rng.random((INPUT_PLANES, 8, 8), dtype=np.float32) for _ in range(batch_size)]
    policies = [rng.random(POLICY_OUTPUT, dtype=np.float32) for _ in range(batch_size)]
    return states, policies


def torch_augment(states: list[np.ndarray], policies: list[np.ndarray], transform: str) -> None:
    s = torch.from_numpy(np.stack(states)).to(torch.float32)
    p = torch.from_numpy(np.stack(policies)).to(torch.float32)

    perms = {
        "mirror": torch.from_numpy(Augment._policy_index_permutation("mirror")).long(),  # type: ignore[attr-defined]
        "rot180": torch.from_numpy(Augment._policy_index_permutation("rot180")).long(),  # type: ignore[attr-defined]
        "vflip_cs": torch.from_numpy(Augment._policy_index_permutation("vflip_cs")).long(),  # type: ignore[attr-defined]
    }
    plane_perm = torch.from_numpy(Augment._vflip_cs_plane_permutation(INPUT_PLANES)).long()  # type: ignore[attr-defined]
    turn_plane = Augment._feature_plane_indices().get("turn_plane", INPUT_PLANES)  # type: ignore[attr-defined]

    if transform == "mirror":
        s = torch.flip(s, dims=[-1])
        p = torch.index_select(p, 1, perms["mirror"])
    elif transform == "rot180":
        s = torch.flip(s, dims=[-2, -1])
        p = torch.index_select(p, 1, perms["rot180"])
    elif transform == "vflip_cs":
        s = torch.flip(s, dims=[-2])
        s = torch.index_select(s, 1, plane_perm)
        if 0 <= turn_plane < s.shape[1]:
            s[:, turn_plane] = 1.0 - s[:, turn_plane]
        p = torch.index_select(p, 1, perms["vflip_cs"])
    else:
        raise ValueError(transform)

    _ = (s.sum() + p.sum()).item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NumPy vs Torch augmentation backends.")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of positions per trial.")
    parser.add_argument("--repeats", type=int, default=20, help="Timed iterations per backend.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per backend.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducible inputs.")
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=["mirror", "rot180", "vflip_cs"],
        help="Transforms to benchmark.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_reports") / "augmentation_backends.csv",
        help="Write CSV summary.",
    )
    args = parser.parse_args()

    states, policies = make_random_batch(args.batch_size, args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    measurements: list[Measurement] = []
    for transform in args.transforms:

        def apply_numpy() -> None:
            Augment.apply(states, policies, transform)

        numpy_measure = benchmark_callable(
            f"numpy-{transform}",
            apply_numpy,
            warmup=args.warmup,
            repeat=args.repeats,
        )
        numpy_measure.extras.update(
            {
                "backend": "numpy",
                "transform": transform,
                "batch_size": args.batch_size,
            }
        )
        measurements.append(numpy_measure)

        def apply_torch() -> None:
            torch_augment(states, policies, transform)

        torch_measure = benchmark_callable(
            f"torch-{transform}",
            apply_torch,
            warmup=args.warmup,
            repeat=args.repeats,
        )
        torch_measure.extras.update(
            {
                "backend": "torch",
                "transform": transform,
                "batch_size": args.batch_size,
            }
        )
        measurements.append(torch_measure)

    report = summarize_measurements(
        measurements,
        metadata={
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "seed": args.seed,
        },
    )

    print("Augmentation backend benchmark (mean latency in ms):")
    print(render_markdown_table(measurements))

    write_reports(
        report,
        csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
