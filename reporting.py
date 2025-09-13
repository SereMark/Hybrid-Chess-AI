from __future__ import annotations

import os
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


def startup_summary(trainer: Any) -> str:
    props = torch.cuda.get_device_properties(trainer.device)
    sm = f"{props.major}.{props.minor}"
    mp = int(getattr(props, "multi_processor_count", 0))
    autocast_mode = "fp16" if C.TORCH.AMP_ENABLED else "off"
    tf32_effective = bool(torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32)
    total_params_m = sum(p.numel() for p in trainer.model.parameters()) / 1e6
    env_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
    env_modload = os.environ.get("CUDA_MODULE_LOADING", "default")
    env_cache = os.environ.get("CUDA_CACHE_MAXSIZE", "default")

    sections: list[str] = []
    sections.append(
        f"[GPU     ] {trainer.device_name} (device {trainer.device}) | SM {sm} | VRAM {trainer.device_total_gb:.1f}G | MPs {mp}"
    )
    sections.append(
        f"[PyTorch ] {torch.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()} | compile={'on' if trainer._compiled else 'off'}(mode={C.TORCH.COMPILE_MODE}, full={C.TORCH.COMPILE_FULLGRAPH}, dyn={C.TORCH.COMPILE_DYNAMIC})"
    )
    sections.append(
        f"[Precision] autocast={autocast_mode} | TF32={'on' if tf32_effective else 'off'} | matmul={torch.get_float32_matmul_precision()} | GradScaler={'on' if C.TORCH.AMP_ENABLED else 'off'}"
    )
    sections.append(
        f"[Memory  ] channels_last train={C.TORCH.MODEL_CHANNELS_LAST} eval={C.TORCH.EVAL_MODEL_CHANNELS_LAST} | pin_memory train={C.TORCH.TRAIN_PIN_MEMORY} eval={C.TORCH.EVAL_PIN_MEMORY}"
    )
    sections.append(f"[Allocator] {env_alloc} | cuda_module_loading={env_modload} | cuda_cache_max={env_cache}")
    try:
        sp_workers = int(trainer.selfplay_engine.get_num_workers())
    except Exception:
        sp_workers = int(C.SELFPLAY.NUM_WORKERS)
    sections.append(
        f"[Threads ] torch intra={torch.get_num_threads()} inter={torch.get_num_interop_threads()} | CPU cores={os.cpu_count()} | selfplay workers={sp_workers}"
    )
    sections.append(
        f"[Model   ] params={total_params_m:.1f}M | blocks={C.MODEL.BLOCKS} | channels={C.MODEL.CHANNELS} | value_head=({C.MODEL.VALUE_CONV_CHANNELS}->{C.MODEL.VALUE_HIDDEN_DIM})"
    )
    sections.append(f"[Replay  ] capacity={C.REPLAY.BUFFER_CAPACITY:,}")
    sections.append(
        f"[SelfPlay] sims {C.MCTS.TRAIN_SIMULATIONS_BASE}→≥{C.MCTS.TRAIN_SIMULATIONS_MIN} | temp {C.SELFPLAY.TEMP_HIGH}->{C.SELFPLAY.TEMP_LOW}@{C.SELFPLAY.TEMP_MOVES} | resign {C.RESIGN.VALUE_THRESHOLD} x{C.RESIGN.CONSECUTIVE_PLIES} | dirichlet alpha={C.MCTS.DIRICHLET_ALPHA} w={C.MCTS.DIRICHLET_WEIGHT}"
    )
    sections.append(
        f"[Train   ] iters={C.TRAIN.TOTAL_ITERATIONS} | games/iter={C.TRAIN.GAMES_PER_ITER} | batch={trainer.train_batch_size} | sched_est={C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST}"
    )
    sections.append(
        f"[LR      ] init={C.TRAIN.LR_INIT:.2e} | warmup={trainer.scheduler.warm} | final={C.TRAIN.LR_FINAL:.2e} | weight_decay={C.TRAIN.WEIGHT_DECAY} | momentum={C.TRAIN.MOMENTUM}"
    )
    sections.append(
        f"[Loss    ] policy={C.TRAIN.LOSS_POLICY_WEIGHT} | value={C.TRAIN.LOSS_VALUE_WEIGHT}->{C.TRAIN.LOSS_VALUE_WEIGHT_LATE}@{C.TRAIN.LOSS_VALUE_WEIGHT_SWITCH_ITER} | smooth={C.TRAIN.LOSS_POLICY_LABEL_SMOOTH:.2f} | entropy={C.TRAIN.LOSS_ENTROPY_COEF_INIT:.2e}→{C.TRAIN.LOSS_ENTROPY_COEF_MIN:.2e}@{C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS}"
    )
    sections.append(
        f"[Augment ] mirror={C.AUGMENT.MIRROR_PROB:.2f} | rot180={C.AUGMENT.ROT180_PROB:.2f} | vflip_cs={C.AUGMENT.VFLIP_CS_PROB:.2f}"
    )
    sections.append(
        f"[EMA     ] {'on' if C.TRAIN.EMA_ENABLED else 'off'} | decay={C.TRAIN.EMA_DECAY if C.TRAIN.EMA_ENABLED else 0}"
    )
    sections.append(
        f"[Eval    ] batch_max={C.EVAL.BATCH_SIZE_MAX} @ {C.EVAL.COALESCE_MS}ms | cache={C.EVAL.CACHE_CAPACITY:,} | dtype={'fp16' if C.EVAL.CACHE_USE_FP16 else 'fp32'}"
    )
    sections.append(
        f"[Arena   ] every={C.ARENA.EVAL_EVERY_ITERS} | games={C.ARENA.GAMES_PER_EVAL} | baseline_p={100.0 * C.ARENA.GATE_BASELINE_P:.0f}% | Z {C.ARENA.GATE_Z_EARLY}->{C.ARENA.GATE_Z_LATE} (switch@{C.ARENA.GATE_Z_SWITCH_ITER}) | min_games={C.ARENA.GATE_MIN_GAMES} | det={'on' if C.ARENA.DETERMINISTIC else 'off'}"
    )
    sections.append(f"[Expect  ] total_games={C.TRAIN.TOTAL_ITERATIONS * C.TRAIN.GAMES_PER_ITER:,}")
    title = "Hybrid Chess AI Training"
    bar = "=" * max(72, len(title))
    return "\n" + bar + f"\n{title}\n" + bar + "\n" + "\n".join(sections) + "\n" + bar
