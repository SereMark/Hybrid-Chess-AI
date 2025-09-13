from __future__ import annotations

import os
import platform
import socket
import sys
import time
from contextlib import suppress
from datetime import datetime
from typing import Any

import psutil
import torch

import config as C


def format_time(seconds: float) -> str:
    s = float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


def format_si(n: int | float, digits: int = 1) -> str:
    try:
        x = float(n)
    except Exception:
        return str(n)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000_000:
        return f"{sign}{x / 1_000_000_000:.{digits}f}B"
    if x >= 1_000_000:
        return f"{sign}{x / 1_000_000:.{digits}f}M"
    if x >= 1_000:
        return f"{sign}{x / 1_000:.{digits}f}k"
    if digits <= 0:
        return f"{sign}{int(x)}"
    return f"{sign}{x:.{digits}f}"


def format_gb(x: float, digits: int = 1) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:.{digits}f}G"


def get_mem_info(proc: psutil.Process, device: torch.device, device_total_gb: float) -> dict[str, float]:
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
        "total_gb": float(device_total_gb),
        "rss_gb": proc.memory_info().rss / 1024**3,
    }


def get_sys_info(proc: psutil.Process) -> dict[str, float]:
    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
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
        "swap_used_gb": float(swap.used) / 1024**3,
        "swap_total_gb": float(swap.total) / 1024**3,
        "swap_pct": float(swap.percent),
        "load1": float(load1),
        "load5": float(load5),
        "load15": float(load15),
    }


def _read_cpu_model() -> str:
    try:
        if os.path.isfile("/proc/cpuinfo"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    cpu = platform.processor() or ""
    if not cpu:
        cpu = platform.machine()
    return cpu or "unknown"


def _format_kvs(tag: str, kv: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in kv.items():
        parts.append(f"{k}={v}")
    return f"[{tag:<8}] " + " | ".join(parts)


def _tag(t: str) -> str:
    return f"[{t:<8}] "


def _format_block(tag: str, kv: dict[str, Any]) -> str:
    if not kv:
        return f"[{tag:<8}]"
    keys = list(kv.keys())
    with suppress(Exception):
        keys = sorted(keys)
    width = max(6, max(len(str(k)) for k in keys))
    lines = [f"[{tag:<8}]"]
    for k in keys:
        v = kv[k]
        lines.append(f"  {k!s:<{width}}: {v}")
    return "\n".join(lines)


def format_iteration_summary(
    trainer: Any,
    iter_stats: dict[str, Any],
    arena_elapsed: float,
    next_ar: int,
    peak_alloc_gb: float,
    peak_reserved_gb: float,
    mem_info_summary: dict[str, float] | None = None,
    sys_info: dict[str, float] | None = None,
) -> list[str]:
    sp_time = float(iter_stats.get("selfplay_time", 0.0))
    tr_time = float(iter_stats.get("training_time", 0.0))
    full_iter_time = sp_time + tr_time + float(arena_elapsed)
    lr = float(iter_stats.get("learning_rate", 0.0))
    opt_steps = int(iter_stats.get("optimizer_steps", 0))
    train_bs = int(trainer.train_batch_size)
    train_time = float(iter_stats.get("training_time", 0.0))
    batches_per_sec = (opt_steps / train_time) if train_time > 0 else 0.0
    samples_per_sec = batches_per_sec * train_bs
    if mem_info_summary is None:
        mem_info_summary = get_mem_info(trainer._proc, trainer.device, trainer.device_total_gb)
    if sys_info is None:
        sys_info = get_sys_info(trainer._proc)
    iter_line = (
        _tag("Iter")
        + f"it {trainer.iteration}/{C.TRAIN.TOTAL_ITERATIONS} | time {format_time(full_iter_time)} | sp {format_time(sp_time)} | tr {format_time(tr_time)} | ar {format_time(arena_elapsed)} | "
        + f"elapsed {format_time(time.time() - trainer.start_time)} | next_ar {int(next_ar)} | games {trainer.total_games:,} | "
        + f"lr {lr:.2e} | steps {opt_steps} | batch {train_bs} | sps {format_si(samples_per_sec)}"
    )
    mem_line = (
        _tag("Mem")
        + f"GPU peak {format_gb(peak_alloc_gb)}/{format_gb(peak_reserved_gb)} | cur {format_gb(mem_info_summary['allocated_gb'])}/{format_gb(mem_info_summary['reserved_gb'])} | total {format_gb(mem_info_summary['total_gb'])} | "
        + f"RSS {format_gb(mem_info_summary['rss_gb'])} | RAM {format_gb(sys_info['ram_used_gb'])}/{format_gb(sys_info['ram_total_gb'])} ({sys_info['ram_pct']:.0f}%)"
    )
    return [iter_line, mem_line]


def startup_summary(trainer: Any) -> str:
    autocast_mode = "fp16" if C.TORCH.AMP_ENABLED else "off"
    tf32_effective = bool(torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32)
    try:
        nccl_v_raw = torch.cuda.nccl.version()
        if isinstance(nccl_v_raw, tuple) and len(nccl_v_raw) >= 3:
            nccl_v = ".".join(str(int(x)) for x in nccl_v_raw[:3])
        elif isinstance(nccl_v_raw, int):
            s = str(nccl_v_raw)
            if len(s) >= 3:
                if len(s) == 4:
                    nccl_v = f"{s[0]}.{s[1]}.{int(s[2:]):d}"
                elif len(s) == 3:
                    nccl_v = f"{s[0]}.{s[1]}.{s[2]}"
            else:
                nccl_v = s
        else:
            nccl_v = str(nccl_v_raw)
    except Exception:
        nccl_v = "unknown"

    total_params_m = sum(p.numel() for p in trainer.model.parameters()) / 1e6

    env_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
    env_modload = os.environ.get("CUDA_MODULE_LOADING", "default")
    env_cache = os.environ.get("CUDA_CACHE_MAXSIZE", "default")
    env_cublas_ws = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "default")
    env_omp = os.environ.get("OMP_NUM_THREADS", "default")
    env_mkl = os.environ.get("MKL_NUM_THREADS", "default")
    env_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "default")
    env_tf32 = os.environ.get("NVIDIA_TF32_OVERRIDE", "default")

    proc = trainer._proc if hasattr(trainer, "_proc") else psutil.Process(os.getpid())
    sysinfo = get_sys_info(proc)
    cpu_model = _read_cpu_model()
    cpu_phys = psutil.cpu_count(logical=False) or 0
    cpu_log = psutil.cpu_count(logical=True) or 0
    try:
        freq = psutil.cpu_freq()
        cpu_freq = f"{(freq.current / 1000.0):.2f}GHz" if freq and freq.current else "unknown"
    except Exception:
        cpu_freq = "unknown"
    hostname = socket.gethostname()
    os_name = platform.system()
    os_release = platform.release()
    os_ver = platform.version()
    py_ver = sys.version.split(" ")[0]
    pid = os.getpid()
    cwd = os.getcwd()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections: list[str] = []

    title = "Hybrid Chess AI Training"
    bar = "=" * max(72, len(title))

    sections.append(
        _format_block(
            "System",
            {
                "Host": hostname,
                "OS": f"{os_name} {os_release}",
                "Kernel": os_ver,
                "Python": py_ver,
                "PID": pid,
                "CWD": cwd,
                "Start": start_ts,
            },
        )
    )
    sections.append(
        _format_block(
            "CPU",
            {
                "Model": cpu_model,
                "Phys Cores": cpu_phys,
                "Log Cores": cpu_log,
                "Freq": cpu_freq,
                "Load1": f"{sysinfo['load1']:.2f}",
            },
        )
    )

    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 1
    try:
        cur_index = torch.cuda.current_device()
    except Exception:
        cur_index = 0
    for idx in range(device_count):
        dev = torch.device(f"cuda:{idx}")
        p = torch.cuda.get_device_properties(dev)
        try:
            free_b_i, total_b_i = torch.cuda.mem_get_info(dev)
        except Exception:
            free_b_i, total_b_i = 0, int(p.total_memory)
        kv_gpu: dict[str, Any] = {
            "Name": p.name,
            "Selected": (str(trainer.device) in (f"cuda:{idx}", "cuda") and idx == cur_index),
            "Index": idx,
            "CC": f"{p.major}.{p.minor}",
            "SMs": int(getattr(p, "multi_processor_count", 0)),
            "Warp": int(getattr(p, "warp_size", 32)),
            "VRAM": f"{(total_b_i / 1024**3):.1f}G (free {(free_b_i / 1024**3):.1f}G)",
        }
        l2_i = int(getattr(p, "l2_cache_size", 0))
        if l2_i > 0:
            kv_gpu["L2"] = f"{l2_i // 1024} KiB"
        bus_i = int(getattr(p, "memory_bus_width", 0))
        if bus_i > 0:
            kv_gpu["Mem Bus"] = f"{bus_i}-bit"
        clk_i = int(getattr(p, "memory_clock_rate", 0))
        if clk_i > 0:
            kv_gpu["Mem Clock"] = f"{clk_i} MHz"
        sections.append(_format_block(f"GPU {idx}", kv_gpu))
    sections.append("")

    sections.append(
        _format_block(
            "PyTorch",
            {
                "Torch": torch.__version__,
                "CUDA": getattr(torch.version, "cuda", "unknown"),
                "cuDNN": torch.backends.cudnn.version(),
                "NCCL": nccl_v,
                "Compile": ("on" if trainer._compiled else "off"),
                "Mode": C.TORCH.COMPILE_MODE,
                "FullGraph": C.TORCH.COMPILE_FULLGRAPH,
                "Dynamic": C.TORCH.COMPILE_DYNAMIC,
            },
        )
    )
    sections.append(
        _format_block(
            "Precision",
            {
                "Autocast": autocast_mode,
                "TF32": ("on" if tf32_effective else "off"),
                "Matmul": torch.get_float32_matmul_precision(),
                "GradScaler": ("on" if C.TORCH.AMP_ENABLED else "off"),
            },
        )
    )
    sections.append("")

    sections.append(
        _format_block(
            "Memory",
            {
                "Channels Last (train)": C.TORCH.MODEL_CHANNELS_LAST,
                "Channels Last (eval)": C.TORCH.EVAL_MODEL_CHANNELS_LAST,
                "Pin Memory (train)": C.TORCH.TRAIN_PIN_MEMORY,
                "Pin Memory (eval)": C.TORCH.EVAL_PIN_MEMORY,
            },
        )
    )
    sections.append(
        _format_block(
            "Threads",
            {
                "Torch Intra": torch.get_num_threads(),
                "Torch Inter": torch.get_num_interop_threads(),
                "CPU Cores": os.cpu_count() or 0,
                "Selfplay Workers": (
                    int(trainer.selfplay_engine.get_num_workers())
                    if hasattr(trainer, "selfplay_engine")
                    else C.SELFPLAY.NUM_WORKERS
                ),
            },
        )
    )
    sections.append(
        _format_block(
            "Env",
            {
                "PYTORCH_CUDA_ALLOC_CONF": env_alloc,
                "CUDA_MODULE_LOADING": env_modload,
                "CUDA_CACHE_MAXSIZE": env_cache,
                "CUBLAS_WORKSPACE_CONFIG": env_cublas_ws,
                "OMP_NUM_THREADS": env_omp,
                "MKL_NUM_THREADS": env_mkl,
                "CUDA_VISIBLE_DEVICES": env_cuda_vis,
                "NVIDIA_TF32_OVERRIDE": env_tf32,
            },
        )
    )
    sections.append("")

    sections.append(
        _format_block(
            "Model",
            {
                "Params": f"{total_params_m:.1f}M",
                "Blocks": C.MODEL.BLOCKS,
                "Channels": C.MODEL.CHANNELS,
                "Value Head": f"{C.MODEL.VALUE_CONV_CHANNELS}->{C.MODEL.VALUE_HIDDEN_DIM}",
            },
        )
    )
    sections.append(_format_block("Replay", {"Capacity": f"{C.REPLAY.BUFFER_CAPACITY:,}"}))
    sections.append(
        _format_block(
            "SelfPlay",
            {
                "Sims": f"{C.MCTS.TRAIN_SIMULATIONS_BASE}→≥{C.MCTS.TRAIN_SIMULATIONS_MIN}",
                "Temp": f"{C.SELFPLAY.TEMP_HIGH}->{C.SELFPLAY.TEMP_LOW}@{C.SELFPLAY.TEMP_MOVES}",
                "Resign": f"{C.RESIGN.VALUE_THRESHOLD} x{C.RESIGN.CONSECUTIVE_PLIES}",
                "Dirichlet": f"alpha={C.MCTS.DIRICHLET_ALPHA} w={C.MCTS.DIRICHLET_WEIGHT}",
                "Open Rand": C.SELFPLAY.OPENING_RANDOM_PLIES_MAX,
            },
        )
    )
    sections.append(
        _format_block(
            "Train",
            {
                "Iters": C.TRAIN.TOTAL_ITERATIONS,
                "Games/iter": C.TRAIN.GAMES_PER_ITER,
                "Batch": trainer.train_batch_size,
                "Sched Est": C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST,
            },
        )
    )
    sections.append(
        _format_block(
            "LR",
            {
                "Init": f"{C.TRAIN.LR_INIT:.2e}",
                "Warmup Steps": trainer.scheduler.warm,
                "Final": f"{C.TRAIN.LR_FINAL:.2e}",
                "Weight Decay": C.TRAIN.WEIGHT_DECAY,
                "Momentum": C.TRAIN.MOMENTUM,
            },
        )
    )
    sections.append(
        _format_block(
            "Loss",
            {
                "Policy": C.TRAIN.LOSS_POLICY_WEIGHT,
                "Value": f"{C.TRAIN.LOSS_VALUE_WEIGHT}->{C.TRAIN.LOSS_VALUE_WEIGHT_LATE}@{C.TRAIN.LOSS_VALUE_WEIGHT_SWITCH_ITER}",
                "Smooth": f"{C.TRAIN.LOSS_POLICY_LABEL_SMOOTH:.2f}",
                "Entropy": f"{C.TRAIN.LOSS_ENTROPY_COEF_INIT:.2e}→{C.TRAIN.LOSS_ENTROPY_COEF_MIN:.2e}@{C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS}",
                "Grad Clip": C.TRAIN.GRAD_CLIP_NORM,
            },
        )
    )
    sections.append(
        _format_block(
            "Augment",
            {
                "Mirror": f"{C.AUGMENT.MIRROR_PROB:.2f}",
                "Rot180": f"{C.AUGMENT.ROT180_PROB:.2f}",
                "VFlip-CS": f"{C.AUGMENT.VFLIP_CS_PROB:.2f}",
            },
        )
    )
    sections.append(
        _format_block(
            "EMA", {"Enabled": (C.TRAIN.EMA_ENABLED), "Decay": (C.TRAIN.EMA_DECAY if C.TRAIN.EMA_ENABLED else 0)}
        )
    )
    sections.append(
        _format_block(
            "Eval",
            {
                "Batch Max": C.EVAL.BATCH_SIZE_MAX,
                "Coalesce ms": C.EVAL.COALESCE_MS,
                "Cache": f"{C.EVAL.CACHE_CAPACITY:,}",
                "Arena Cache": f"{C.EVAL.ARENA_CACHE_CAPACITY:,}",
                "DType": ("fp16" if C.EVAL.CACHE_USE_FP16 else "fp32"),
            },
        )
    )
    sections.append(
        _format_block(
            "Arena",
            {
                "Every": C.ARENA.EVAL_EVERY_ITERS,
                "Games": C.ARENA.GAMES_PER_EVAL,
                "Baseline p": f"{100.0 * C.ARENA.GATE_BASELINE_P:.0f}%",
                "Z": f"{C.ARENA.GATE_Z_EARLY}->{C.ARENA.GATE_Z_LATE} (sw@{C.ARENA.GATE_Z_SWITCH_ITER})",
                "Min Games": C.ARENA.GATE_MIN_GAMES,
                "Deterministic": ("on" if C.ARENA.DETERMINISTIC else "off"),
            },
        )
    )
    sections.append(_format_block("Expect", {"Total Games": f"{C.TRAIN.TOTAL_ITERATIONS * C.TRAIN.GAMES_PER_ITER:,}"}))
    sections.append("")

    sections.append(_format_block("CFG.LOG", vars(C.LOG)))
    sections.append(_format_block("CFG.TORCH", vars(C.TORCH)))
    sections.append(_format_block("CFG.DATA", vars(C.DATA)))
    sections.append(_format_block("CFG.MODEL", vars(C.MODEL)))
    sections.append(_format_block("CFG.EVAL", vars(C.EVAL)))
    sections.append(_format_block("CFG.SELF", vars(C.SELFPLAY)))
    sections.append(_format_block("CFG.REPL", vars(C.REPLAY)))
    sections.append(_format_block("CFG.SAMP", vars(C.SAMPLING)))
    sections.append(_format_block("CFG.AUG", vars(C.AUGMENT)))
    sections.append(_format_block("CFG.MCTS", vars(C.MCTS)))
    sections.append(_format_block("CFG.RESN", vars(C.RESIGN)))
    sections.append(_format_block("CFG.TRAIN", vars(C.TRAIN)))
    sections.append(_format_block("CFG.ARENA", vars(C.ARENA)))

    return "\n" + bar + f"\n{title}\n" + bar + "\n" + "\n".join(sections) + "\n" + bar
