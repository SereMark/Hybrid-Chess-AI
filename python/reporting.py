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
    settings = getattr(trainer, "settings", None)
    if settings is not None:
        total_iterations = settings.training.total_iterations
        games_per_iter = settings.training.games_per_iteration
        batch_size = settings.training.batch_size
        lr_init = settings.training.lr_init
        lr_final = settings.training.lr_final
        selfplay_workers = settings.selfplay.num_workers
        sims_start = settings.mcts.train_simulations_base
        sims_end = settings.mcts.train_simulations_min
        max_plies = settings.selfplay.game_max_plies
        adj_margin = settings.selfplay.adjudicate_margin_start
        adj_min_plies = settings.selfplay.adjudicate_min_start
        dirichlet_late = getattr(C.SELFPLAY, "DIRICHLET_WEIGHT_LATE", 0.0)
        temp_moves = settings.selfplay.temp_moves
        arena_every = settings.arena.eval_every_iters
        arena_games = settings.arena.games_per_eval
        arena_margin = settings.arena.gate_baseline_margin
        csv_path = settings.logging.csv_path
    else:
        total_iterations = C.TRAIN.TOTAL_ITERATIONS
        games_per_iter = C.TRAIN.GAMES_PER_ITER
        batch_size = C.TRAIN.BATCH_SIZE
        lr_init = C.TRAIN.LR_INIT
        lr_final = C.TRAIN.LR_FINAL
        selfplay_workers = C.SELFPLAY.NUM_WORKERS
        sims_start = C.MCTS.TRAIN_SIMULATIONS_BASE
        sims_end = C.MCTS.TRAIN_SIMULATIONS_MIN
        max_plies = C.SELFPLAY.GAME_MAX_PLIES
        adj_margin = getattr(C.SELFPLAY, "ADJUDICATE_MATERIAL_MARGIN", 0.0)
        adj_min_plies = getattr(C.SELFPLAY, "ADJUDICATE_MIN_PLIES", 0)
        dirichlet_late = getattr(C.SELFPLAY, "DIRICHLET_WEIGHT_LATE", 0.0)
        temp_moves = C.SELFPLAY.TEMP_MOVES
        arena_every = C.ARENA.EVAL_EVERY_ITERS
        arena_games = C.ARENA.GAMES_PER_EVAL
        arena_margin = C.ARENA.GATE_BASELINE_MARGIN
        csv_path = C.LOG.METRICS_LOG_CSV_PATH
    lines = [
        "=" * 68,
        "Hybrid Chess AI Training",
        "=" * 68,
        f"start   : {timestamp} | host={socket.gethostname()} | pid={os.getpid()}",
        f"device  : {trainer.device} ({trainer.device_name}) | cuda={'yes' if has_cuda else 'no'} | autocast={autocast_mode}",
        f"model   : {total_params_m:.1f}M params ({C.MODEL.BLOCKS} blocks, {C.MODEL.CHANNELS} ch) | replay={C.REPLAY.BUFFER_CAPACITY:,}",
        f"selfplay: workers={selfplay_workers} sims={sims_start}->{sims_end} max_plies={max_plies}"
        f" adj={'on' if getattr(C.SELFPLAY, 'ADJUDICATE_ENABLED', False) else 'off'}"
        f" margin={adj_margin} min_plies={adj_min_plies}"
        f" dir_late={dirichlet_late:.2f} temp_moves={temp_moves}",
        f"train   : iters={total_iterations} games/iter={games_per_iter} batch={batch_size} lr={lr_init:.2e}->{lr_final:.2e}",
        f"arena   : every {arena_every} iters | games={arena_games} | gate margin={arena_margin}",
        f"logging : csv={csv_path}",
        "=" * 68,
        "",
    ]
    return "\n".join(lines)
