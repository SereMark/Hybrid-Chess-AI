"""CLI entrypoint for training the Hybrid Chess AI network."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Sequence, Tuple

import config as C
import numpy as np
import torch
from trainer import Trainer

__all__ = ["main"]

_ENV_BASE_DEFAULTS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False,max_split_size_mb:192",
    "CUDA_MODULE_LOADING": "LAZY",
    "CUDA_CACHE_MAXSIZE": "2147483648",
    "NVIDIA_TF32_OVERRIDE": "0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
}


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid Chess AI training entrypoint")
    parser.add_argument(
        "--config",
        "-c",
        action="append",
        dest="configs",
        default=[],
        help="Path to a base configuration file (YAML or JSON). Can be specified multiple times.",
    )
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        dest="overrides",
        default=[],
        help="Additional configuration override files applied after base configs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the most recent checkpoint.",
    )
    parser.add_argument(
        "--dry-config",
        action="store_true",
        help="Load configuration files and print the merged snapshot without starting training.",
    )
    return parser.parse_args(list(args))


def _apply_cli_configs(parsed: argparse.Namespace) -> None:
    loaded: list[str] = []
    for idx, path in enumerate(parsed.configs):
        try:
            C.load_file(path, replace=(idx == 0))
            loaded.append(str(path))
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.getLogger("hybridchess.runtime").error("Failed to load config %s: %s", path, exc)
            raise
    for path in parsed.overrides:
        try:
            C.load_file(path)
            loaded.append(str(path))
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.getLogger("hybridchess.runtime").error("Failed to load override %s: %s", path, exc)
            raise
    if loaded:
        logging.getLogger("hybridchess.runtime").info("Loaded configuration overlays: %s", ", ".join(loaded))


def _resolve_thread_settings() -> Tuple[int, int]:
    logical = os.cpu_count() or 1
    if C.TORCH.threads_intra > 0:
        intra = int(C.TORCH.threads_intra)
    else:
        intra = logical - 4 if logical >= 12 else logical - 2 if logical > 4 else logical
    intra = max(1, min(intra, logical))

    if C.TORCH.threads_inter > 0:
        inter = int(C.TORCH.threads_inter)
    else:
        inter = max(1, min(4, logical // 4 or 1))
    inter = max(1, min(inter, logical))
    return intra, inter


def _set_backend_flag(module: object, attr: str, value: bool) -> None:
    if hasattr(module, attr):
        setattr(module, attr, value)


def _apply_environment_defaults(threads_intra: int, threads_inter: int) -> None:
    env_defaults = dict(_ENV_BASE_DEFAULTS)
    env_defaults["OMP_NUM_THREADS"] = str(threads_intra)
    env_defaults["MKL_NUM_THREADS"] = str(max(1, min(threads_intra, threads_inter * 2)))
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)
    if C.SEED != 0:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _configure_logging() -> int:
    log_level = getattr(logging, str(C.LOG.level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(stdout_handler)
    return log_level


def _seed_everything(has_cuda: bool) -> None:
    if C.SEED == 0:
        return
    import random as _py_random

    _py_random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    if has_cuda:
        torch.cuda.manual_seed_all(C.SEED)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv if argv is not None else sys.argv[1:])
    parsed = _parse_args(raw_args)
    _apply_cli_configs(parsed)
    if parsed.dry_config:
        print(json.dumps(C.MANAGER.to_dict(), indent=2))
        return 0
    threads_intra, threads_inter = _resolve_thread_settings()
    _apply_environment_defaults(threads_intra, threads_inter)
    _configure_logging()
    has_cuda = torch.cuda.is_available()
    trainer_log = logging.getLogger("hybridchess.trainer")
    if has_cuda:
        torch.set_float32_matmul_precision(C.TORCH.matmul_float32_precision)
        _set_backend_flag(
            torch.backends.cuda.matmul,
            "allow_tf32",
            bool(C.TORCH.cuda_allow_tf32),
        )
        _set_backend_flag(
            torch.backends.cudnn,
            "allow_tf32",
            bool(C.TORCH.cudnn_allow_tf32),
        )
        _set_backend_flag(
            torch.backends.cuda.matmul,
            "allow_fp16_reduced_precision_reduction",
            bool(C.TORCH.cuda_allow_fp16_reduced_reduction),
        )
        _set_backend_flag(
            torch.backends.cudnn,
            "benchmark",
            bool(C.TORCH.cudnn_benchmark),
        )
    else:
        trainer_log.warning("CUDA unavailable; running training in CPU mode")
    if C.SEED != 0:
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads_intra)
    torch.set_num_interop_threads(threads_inter)
    logging.getLogger("hybridchess.runtime")
    _seed_everything(has_cuda)
    Trainer(resume=parsed.resume).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
