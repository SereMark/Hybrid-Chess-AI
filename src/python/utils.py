"""Utility helpers for logging, formatting, and FEN manipulation."""

from __future__ import annotations

import contextlib
import csv
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import config as C
import numpy as np
import psutil
import torch
from torch import nn

__all__ = [
    "prepare_model",
    "select_autocast_dtype",
    "select_inference_dtype",
    "MetricsReporter",
    "flip_fen_perspective",
    "sanitize_fen",
    "format_gb",
    "format_si",
    "format_time",
    "get_mem_info",
    "get_sys_info",
    "startup_summary",
]

_CASTLING_ORDER = "KQkq"
_FILES = "abcdefgh"


def _cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(checker):
        return False
    with contextlib.suppress(Exception):
        return bool(checker())
    return False


def select_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.bfloat16
    return torch.bfloat16 if _cuda_supports_bf16() else torch.float16


def select_inference_dtype(device: torch.device, *, cpu_dtype: torch.dtype = torch.float32) -> torch.dtype:
    if device.type != "cuda":
        return cpu_dtype
    return select_autocast_dtype(device)


def prepare_model(
    model: nn.Module,
    device: torch.device,
    *,
    channels_last: bool = False,
    dtype: Optional[torch.dtype] = None,
    eval_mode: bool = False,
    freeze: bool = False,
) -> nn.Module:
    model = model.to(device)
    if channels_last:
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.weight.data = module.weight.data.to(memory_format=torch.channels_last)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.eval() if eval_mode else model.train()
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def _flip_square(square: str) -> str:
    if not square or square == "-":
        return "-"
    sq = square.strip()
    if len(sq) != 2:
        return "-"
    file_ch, rank_ch = sq[0], sq[1]
    if file_ch not in _FILES or not rank_ch.isdigit():
        return "-"
    rank_val = int(rank_ch)
    if not 1 <= rank_val <= 8:
        return "-"
    flipped_file = _FILES[::-1][_FILES.index(file_ch)]
    flipped_rank = str(9 - rank_val)
    return f"{flipped_file}{flipped_rank}"


def _swap_castling_rights(rights: str) -> str:
    if not rights or rights == "-":
        return "-"
    swapped: list[str] = []
    for ch in rights:
        if ch in _CASTLING_ORDER:
            swapped.append(ch.lower() if ch.isupper() else ch.upper())
    swapped = list(dict.fromkeys(swapped))
    ordered = [c for c in _CASTLING_ORDER if c in swapped]
    return "".join(ordered) if ordered else "-"


def sanitize_fen(fen: str) -> str:
    parts = fen.strip().split()
    if len(parts) < 6:
        defaults = ["w", "-", "-", "0", "1"]
        parts += defaults[len(parts) - 1 :]
    tail: list[str]
    if len(parts) > 6:
        head = parts[:6]
        tail = parts[6:]
    else:
        head = parts
        tail = []
    if len(head) < 6:
        head += ["-"] * (6 - len(head))
        head[1] = "w"
        head[4] = "0"
        head[5] = "1"
    sanitized = " ".join(head)
    if tail:
        sanitized = " ".join([sanitized, *tail])
    return sanitized


def flip_fen_perspective(fen: str) -> str:
    board, side, castling, ep, halfmove, fullmove, *rest = sanitize_fen(fen).split()
    rows = board.split("/")
    flipped_rows: list[str] = []
    for row in reversed(rows):
        new_row_chars: list[str] = []
        for ch in row:
            if ch.isdigit():
                new_row_chars.append(ch)
            elif ch.isalpha():
                new_row_chars.append(ch.swapcase())
            else:
                new_row_chars.append(ch)
        flipped_rows.append("".join(new_row_chars))
    flipped_board = "/".join(flipped_rows)
    flipped_side = "b" if side.lower() == "w" else "w"
    flipped_castling = _swap_castling_rights(castling)
    flipped_ep = _flip_square(ep)
    flipped_fullmove = fullmove
    flipped_halfmove = halfmove
    components: list[str] = [
        flipped_board,
        flipped_side,
        flipped_castling,
        flipped_ep,
        flipped_halfmove,
        flipped_fullmove,
    ]
    if rest:
        components.extend(rest)
    return " ".join(components)


def iter_with_perspectives(fens: Iterable[str]) -> list[str]:
    out: list[str] = []
    for fen in fens:
        cleaned = sanitize_fen(fen)
        out.append(cleaned)
        out.append(flip_fen_perspective(cleaned))
    return out


def _json_safe(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass(slots=True)
class MetricsReporter:
    """Append-only reporter writing both CSV and JSONL metrics."""

    csv_path: str
    jsonl_path: str | None = None

    def append(self, row: Mapping[str, object], field_order: Sequence[str] | None = None) -> None:
        self._append_csv(row, field_order)
        self._append_jsonl(row)

    def append_json(self, row: Mapping[str, object]) -> None:
        """Record an event solely in the JSONL stream (no CSV impact)."""
        self._append_jsonl(row)

    def _append_csv(self, row: Mapping[str, object], field_order: Sequence[str] | None) -> None:
        if not self.csv_path:
            return

        try:
            directory = os.path.dirname(self.csv_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            fieldnames = list(field_order) if field_order is not None else list(row.keys())
            write_header = not os.path.isfile(self.csv_path)

            with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            raise RuntimeError("Failed to append metrics row") from exc

    def _append_jsonl(self, row: Mapping[str, object]) -> None:
        if not self.jsonl_path:
            return
        try:
            directory = os.path.dirname(self.jsonl_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            safe_row = {str(key): _json_safe(value) for key, value in row.items()}
            with open(self.jsonl_path, "a", encoding="utf-8") as handle:
                json.dump(safe_row, handle, separators=(",", ":"))
                handle.write("\n")
        except Exception as exc:
            raise RuntimeError("Failed to append metrics JSONL row") from exc


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
    info = proc.memory_full_info()
    rss = info.rss / 1024**3
    vms = getattr(info, "vms", info.rss) / 1024**3
    uss = getattr(info, "uss", info.rss) / 1024**3
    if device.type != "cuda":
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_gb": float(device_total_gb),
            "rss_gb": rss,
            "vms_gb": vms,
            "uss_gb": uss,
        }
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
        "total_gb": float(device_total_gb),
        "rss_gb": rss,
        "vms_gb": vms,
        "uss_gb": uss,
    }


def get_sys_info(proc: psutil.Process) -> dict[str, float]:
    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    if hasattr(os, "getloadavg"):
        try:
            load1, load5, load15 = os.getloadavg()
        except Exception:
            load1 = load5 = load15 = 0.0
    else:
        load1 = load5 = load15 = 0.0
    return {
        "cpu_sys_pct": float(psutil.cpu_percent(interval=None)),
        "cpu_proc_pct": float(proc.cpu_percent(interval=None)),
        "ram_used_gb": float(vmem.used) / 1024**3,
        "ram_total_gb": float(vmem.total) / 1024**3,
        "ram_pct": float(vmem.percent),
        "swap_pct": float(swap.percent),
        "load1": float(load1),
        "load5": float(load5),
        "load15": float(load15),
    }


def startup_summary(trainer: Any) -> str:
    has_cuda = bool(trainer.device.type == "cuda" and torch.cuda.is_available())
    amp_enabled = bool(getattr(trainer, "_amp_enabled", C.TORCH.amp_enabled))
    autocast_mode = "fp16" if amp_enabled else "off"
    total_params_m = sum(p.numel() for p in trainer.model.parameters()) / 1e6
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device_name = getattr(trainer, "device_name", str(trainer.device))
    lines = [
        f"[{timestamp}] Hybrid Chess AI training",
        f"device={device_name} ({trainer.device}) cuda={'yes' if has_cuda else 'no'} autocast={autocast_mode} "
        f"params={total_params_m:.2f}M replay={C.REPLAY.capacity:,}",
        " ".join(
            [
                f"train iters={C.TRAIN.total_iterations}",
                f"games/iter={C.TRAIN.games_per_iter}",
                f"batch={trainer.train_batch_size}",
                f"lr={C.TRAIN.learning_rate_init:.2e}->{C.TRAIN.learning_rate_final:.2e}",
                f"grad_clip={C.TRAIN.grad_clip_norm}",
            ]
        ),
        " ".join(
            [
                f"self-play workers={C.SELFPLAY.num_workers}",
                f"sims={C.MCTS.train_simulations}->{C.MCTS.train_simulations_min}",
                f"temp_moves={C.SELFPLAY.temperature_moves}",
                f"max_plies={C.SELFPLAY.game_max_plies}",
                f"adjudication_margin={C.SELFPLAY.adjudication_value_margin}",
            ]
        ),
        " ".join(
            [
                f"arena every={C.ARENA.eval_every_iters}",
                f"games={C.ARENA.games_per_eval}",
                f"gate_baseline={C.ARENA.gate_baseline_p:.3f}",
                f"margin={C.ARENA.gate_margin:.3f}",
            ]
        ),
        f"metrics_csv={getattr(getattr(trainer, 'metrics', None), 'csv_path', '')}",
    ]
    return " | ".join(lines)
