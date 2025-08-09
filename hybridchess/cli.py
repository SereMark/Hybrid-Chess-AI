from __future__ import annotations

import logging

import torch

from .config import CONFIG
from .trainer import Trainer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.backends.cudnn.benchmark = CONFIG.use_cudnn_benchmark
    torch.set_num_threads(CONFIG.torch_num_threads)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(CONFIG.matmul_precision)
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
