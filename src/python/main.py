"""CLI entrypoint for training the Hybrid Chess AI network."""

from __future__ import annotations

import logging
import os
import sys
from typing import Sequence, Tuple

import numpy as np
import torch

import config as C
from trainer import Trainer

__all__ = ["main"]

_ENV_BASE_DEFAULTS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False,max_split_size_mb:192",
    "CUDA_MODULE_LOADING": "LAZY",
    "CUDA_CACHE_MAXSIZE": "2147483648",
    "NVIDIA_TF32_OVERRIDE": "0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
}


def _resolve_thread_settings() -> Tuple[int, int]:
    logical = os.cpu_count() or 1
    if getattr(C.TORCH, "THREADS_INTRA", 0) > 0:
        intra = int(C.TORCH.THREADS_INTRA)
    else:
        intra = logical - 4 if logical >= 12 else logical - 2 if logical > 4 else logical
    intra = max(1, min(intra, logical))

    if getattr(C.TORCH, "THREADS_INTER", 0) > 0:
        inter = int(C.TORCH.THREADS_INTER)
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
    log_level = getattr(logging, str(C.LOG.LEVEL).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
        )
    )
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
    args = list(argv if argv is not None else sys.argv[1:])
    threads_intra, threads_inter = _resolve_thread_settings()
    _apply_environment_defaults(threads_intra, threads_inter)
    _configure_logging()
    has_cuda = torch.cuda.is_available()
    trainer_log = logging.getLogger("hybridchess.trainer")
    if has_cuda:
        torch.set_float32_matmul_precision(C.TORCH.MATMUL_FLOAT32_PRECISION)
        _set_backend_flag(
            torch.backends.cuda.matmul,
            "allow_tf32",
            bool(getattr(C.TORCH, "CUDA_ALLOW_TF32", False)),
        )
        _set_backend_flag(
            torch.backends.cudnn,
            "allow_tf32",
            bool(getattr(C.TORCH, "CUDNN_ALLOW_TF32", False)),
        )
        _set_backend_flag(
            torch.backends.cuda.matmul,
            "allow_fp16_reduced_precision_reduction",
            bool(getattr(C.TORCH, "CUDA_ALLOW_FP16_REDUCED_REDUCTION", True)),
        )
        _set_backend_flag(
            torch.backends.cudnn,
            "benchmark",
            bool(getattr(C.TORCH, "CUDNN_BENCHMARK", True)),
        )
    else:
        trainer_log.warning("CUDA unavailable; running training in CPU mode")
    if C.SEED != 0:
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads_intra)
    torch.set_num_interop_threads(threads_inter)
    logging.getLogger("hybridchess.runtime").info(
        "Torch intra-op threads=%d inter-op threads=%d", threads_intra, threads_inter
    )
    _seed_everything(has_cuda)
    resume_flag = any(arg in {"--resume", "resume"} for arg in args)
    Trainer(resume=resume_flag).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
