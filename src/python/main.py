"""CLI entrypoint for Hybrid Chess AI training."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from typing import Sequence

import config as C
import numpy as np
import torch
from checkpoint import save_checkpoint
from trainer import Trainer

RLOG = logging.getLogger("hybridchess.runtime")
TLOG = logging.getLogger("hybridchess.trainer")

CLI_DESCRIPTION = "Hybrid Chess AI training entrypoint"
__all__ = ["main"]


# ---------------------------------------------------------------------------#
# CLI parsing
# ---------------------------------------------------------------------------#


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    """Define CLI and parse arguments."""
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        dest="configs",
        default=[],
        help="Base configuration file(s), YAML. First replaces defaults; later ones overlay.",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        dest="overrides",
        default=[],
        help="Override configuration file(s), applied after all base configs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the most recent checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string for torch (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    return parser.parse_args(list(args))


def _apply_cli_configs(parsed: argparse.Namespace) -> None:
    """Load base configs in order, then apply overrides."""
    loaded: list[str] = []
    for idx, path in enumerate(parsed.configs):
        try:
            C.load_file(path, replace=(idx == 0))
            loaded.append(str(path))
        except Exception as exc:
            RLOG.error("Failed to load config %s: %s", path, exc)
            raise
    for path in parsed.overrides:
        try:
            C.load_file(path)
            loaded.append(str(path))
        except Exception as exc:
            RLOG.error("Failed to load override %s: %s", path, exc)
            raise
    if loaded:
        RLOG.info("Loaded configuration overlays: %s", ", ".join(loaded))


# ---------------------------------------------------------------------------#
# Runtime preparation
# ---------------------------------------------------------------------------#


def _resolve_thread_settings() -> tuple[int, int]:
    """Derive intra- and inter-op thread counts from CPU logical cores and config hints."""
    logical = os.cpu_count() or 1

    if C.TORCH.threads_intra > 0:
        intra = int(C.TORCH.threads_intra)
    else:
        intra = logical - 4 if logical >= 12 else (logical - 2 if logical > 4 else logical)
    intra = max(1, min(intra, logical))

    if C.TORCH.threads_inter > 0:
        inter = int(C.TORCH.threads_inter)
    else:
        inter = max(1, min(4, logical // 4 or 1))
    inter = max(1, min(inter, logical))

    return intra, inter


def _apply_environment_defaults(threads_intra: int, threads_inter: int, deterministic: bool) -> None:
    """Set minimal env defaults without overriding user-provided values."""
    os.environ.setdefault("OMP_NUM_THREADS", str(threads_intra))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, min(threads_intra, threads_inter * 2))))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _configure_logging() -> int:
    """Install a single stdout handler at the configured level."""
    level = getattr(logging, str(C.LOG.level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(handler)
    return level


def _set_backend_flag(module: object, attr: str, value: bool) -> None:
    """Assign module.attr = value only if the attribute exists (PyTorch-version-safe)."""
    if hasattr(module, attr):
        setattr(module, attr, value)


def _seed_everything(has_cuda: bool) -> None:
    """Seed Python, NumPy, and PyTorch when SEED != 0."""
    if C.SEED == 0:
        return
    import random as _py_random

    _py_random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    if has_cuda:
        torch.cuda.manual_seed_all(C.SEED)


def _configure_torch_backends(has_cuda: bool) -> None:
    """Set math precision and backend toggles according to config."""
    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None) if cuda_backend else None
    matmul_precision = str(C.TORCH.cuda_matmul_fp32_precision).lower()

    if has_cuda and (matmul_backend is None or not hasattr(matmul_backend, "fp32_precision")):
        raise RuntimeError("PyTorch build is missing torch.backends.cuda.matmul.fp32_precision; upgrade PyTorch.")
    if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
        matmul_backend.fp32_precision = matmul_precision

    if not has_cuda:
        TLOG.warning("CUDA unavailable; training will run on CPU")
        return
    cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
    if cudnn_conv is None or not hasattr(cudnn_conv, "fp32_precision"):
        raise RuntimeError("PyTorch build is missing torch.backends.cudnn.conv.fp32_precision; upgrade PyTorch.")
    cudnn_precision = str(C.TORCH.cudnn_conv_fp32_precision).lower()
    cudnn_conv.fp32_precision = cudnn_precision
    _set_backend_flag(
        matmul_backend,
        "allow_fp16_reduced_precision_reduction",
        bool(C.TORCH.cuda_allow_fp16_reduced_reduction),
    )
    _set_backend_flag(torch.backends.cudnn, "benchmark", bool(C.TORCH.cudnn_benchmark))


# ---------------------------------------------------------------------------#
# Entrypoint
# ---------------------------------------------------------------------------#


def main(argv: Sequence[str] | None = None) -> int:
    """Orchestrate setup and start training."""
    args = list(argv if argv is not None else sys.argv[1:])

    # Configure logging early so config-load errors are visible
    _configure_logging()
    parsed = _parse_args(args)
    _apply_cli_configs(parsed)
    # Reconfigure to reflect LOG.level from configs
    _configure_logging()

    # Determinism preference: explicit flag wins, otherwise seed!=0 implies deterministic
    det_cfg = getattr(C.TORCH, "deterministic", None)
    deterministic = bool(det_cfg is True or (det_cfg is None and C.SEED != 0))

    intra, inter = _resolve_thread_settings()
    _apply_environment_defaults(intra, inter, deterministic)

    has_cuda = torch.cuda.is_available()
    _configure_torch_backends(has_cuda)

    if deterministic:
        torch.use_deterministic_algorithms(True)

    torch.set_num_threads(intra)
    torch.set_num_interop_threads(inter)

    _seed_everything(has_cuda)

    trainer = Trainer(device=parsed.device, resume=parsed.resume)
    try:
        trainer.train()
    except KeyboardInterrupt:
        TLOG.warning("Interrupted; saving checkpoint")
        save_checkpoint(trainer)
        return 130
    finally:
        with contextlib.suppress(Exception):
            trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
