from __future__ import annotations

import os
import socket
from datetime import datetime
from typing import Any

import config as C
import psutil
import torch


def format_time(seconds: float) -> str:
    value = float(seconds)
    if value < 60:
        return f"{value:.1f}s"
    if value < 3600:
        return f"{value / 60:.1f}m"
    return f"{value / 3600:.1f}h"


def format_si(value: int | float, digits: int = 1) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number >= 1_000_000_000:
        return f"{sign}{number / 1_000_000_000:.{digits}f}B"
    if number >= 1_000_000:
        return f"{sign}{number / 1_000_000:.{digits}f}M"
    if number >= 1_000:
        return f"{sign}{number / 1_000:.{digits}f}k"
    if digits <= 0:
        return f"{sign}{int(number)}"
    return f"{sign}{number:.{digits}f}"


def format_gb(value: float, digits: int = 1) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    return f"{number:.{digits}f}G"


def get_mem_info(proc: psutil.Process, device: torch.device, device_total_gb: float) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_gb": float(device_total_gb),
            "rss_gb": proc.memory_info().rss / 1024**3,
        }
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
        "total_gb": float(device_total_gb),
        "rss_gb": proc.memory_info().rss / 1024**3,
    }


def get_sys_info(proc: psutil.Process) -> dict[str, float]:
    vmem = psutil.virtual_memory()
    try:
        load1, load5, load15 = os.getloadavg()
    except Exception:
        load1 = load5 = load15 = 0.0
    return {
        "cpu_sys_pct": float(psutil.cpu_percent(0.0)),
        "cpu_proc_pct": float(proc.cpu_percent(0.0)),
        "ram_used_gb": float(vmem.used) / 1024**3,
        "ram_total_gb": float(vmem.total) / 1024**3,
        "ram_pct": float(vmem.percent),
        "load1": float(load1),
        "load5": float(load5),
        "load15": float(load15),
    }


def startup_summary(trainer: Any) -> str:
    has_cuda = bool(trainer.device.type == "cuda" and torch.cuda.is_available())
    amp_enabled = bool(getattr(trainer, "_amp_enabled", C.TORCH.AMP_ENABLED))
    autocast_mode = "fp16" if amp_enabled else "off"
    total_params_m = sum(p.numel() for p in trainer.model.parameters()) / 1e6
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 68,
        "Hybrid Chess AI Training",
        "=" * 68,
        f"start   : {timestamp} | host={socket.gethostname()} | pid={os.getpid()}",
        f"device  : {trainer.device} ({trainer.device_name}) | cuda={'yes' if has_cuda else 'no'} | autocast={autocast_mode}",
        f"model   : {total_params_m:.1f}M params ({C.MODEL.BLOCKS} blocks, {C.MODEL.CHANNELS} ch) | replay={C.REPLAY.BUFFER_CAPACITY:,}",
        f"selfplay: workers={C.SELFPLAY.NUM_WORKERS} sims={C.MCTS.TRAIN_SIMULATIONS_BASE}->{C.MCTS.TRAIN_SIMULATIONS_MIN} max_plies={C.SELFPLAY.GAME_MAX_PLIES}",
        f"train   : iters={C.TRAIN.TOTAL_ITERATIONS} games/iter={C.TRAIN.GAMES_PER_ITER} batch={C.TRAIN.BATCH_SIZE} lr={C.TRAIN.LR_INIT:.2e}->{C.TRAIN.LR_FINAL:.2e}",
        f"arena   : every {C.ARENA.EVAL_EVERY_ITERS} iters | games={C.ARENA.GAMES_PER_EVAL} | gate margin={C.ARENA.GATE_BASELINE_MARGIN}",
        f"logging : csv={C.LOG.METRICS_LOG_CSV_PATH}",
        "=" * 68,
        "",
    ]
    return "\n".join(lines)
