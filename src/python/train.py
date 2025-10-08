"""CLI entrypoint for training the Hybrid Chess AI network."""

from __future__ import annotations

import logging
import os
import sys
from typing import Sequence

import numpy as np
import torch

import config as C
from trainer import Trainer

__all__ = ["main"]

_ENV_DEFAULTS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256",
    "CUDA_MODULE_LOADING": "LAZY",
    "CUDA_CACHE_MAXSIZE": "2147483648",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NVIDIA_TF32_OVERRIDE": "1",
}


def _apply_environment_defaults() -> None:
    for key, value in _ENV_DEFAULTS.items():
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
    _apply_environment_defaults()
    _configure_logging()
    has_cuda = torch.cuda.is_available()
    trainer_log = logging.getLogger("hybridchess.trainer")
    if has_cuda:
        torch.set_float32_matmul_precision(C.TORCH.MATMUL_FLOAT32_PRECISION)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    else:
        trainer_log.warning("CUDA unavailable; running training in CPU mode")
    if C.SEED != 0:
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(C.TORCH.THREADS_INTRA)
    torch.set_num_interop_threads(C.TORCH.THREADS_INTER)
    _seed_everything(has_cuda)
    resume_flag = any(arg in {"--resume", "resume"} for arg in args)
    Trainer(resume=resume_flag).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
