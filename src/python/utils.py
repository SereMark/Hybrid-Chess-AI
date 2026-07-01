from __future__ import annotations

import contextlib
import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import config as C
import numpy as np
import torch
from torch import nn

__all__ = [
    "DEFAULT_START_FEN",
    "select_autocast_dtype",
    "select_inference_dtype",
    "prepare_model",
    "sanitize_fen",
    "flip_fen_perspective",
    "select_visit_count_move",
    "MetricsReporter",
    "format_time",
    "startup_summary",
]

_CASTLING_ORDER = "KQkq"
_FILES = "abcdefgh"

DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


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
    if dtype is not None:
        model = model.to(dtype=dtype)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    model = model.eval() if eval_mode else model.train()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
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
    board = parts[0] if len(parts) > 0 else "8/8/8/8/8/8/8/8"
    side = parts[1] if len(parts) > 1 else "w"
    castling = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"
    halfmove = parts[4] if len(parts) > 4 else "0"
    fullmove = parts[5] if len(parts) > 5 else "1"
    return " ".join([board, side, castling, ep, halfmove, fullmove] + parts[6:])


def _revrow(row: str) -> str:
    cells: list[str] = []
    for ch in row:
        if ch.isdigit():
            cells.extend(["1"] * int(ch))
        else:
            cells.append(ch)
    cells = [c.swapcase() if c.isalpha() else c for c in cells[::-1]]
    out: list[str] = []
    run = 0
    for c in cells:
        if c == "1":
            run += 1
        else:
            if run:
                out.append(str(run))
                run = 0
            out.append(c)
    if run:
        out.append(str(run))
    return "".join(out)


def flip_fen_perspective(fen: str) -> str:
    board, side, castling, ep, halfmove, fullmove, *rest = sanitize_fen(fen).split()
    rows = board.split("/")
    flipped_board = "/".join(_revrow(r) for r in rows[::-1])
    flipped_side = "b" if side.lower() == "w" else "w"
    flipped_castling = _swap_castling_rights(castling)
    flipped_ep = _flip_square(ep)
    components: list[str] = [
        flipped_board,
        flipped_side,
        flipped_castling,
        flipped_ep,
        halfmove,
        fullmove,
    ]
    if rest:
        components.extend(rest)
    return " ".join(components)


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
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(x) for x in value]
    return str(value)


@dataclass(slots=True)
class MetricsReporter:
    csv_path: str
    jsonl_path: str | None = None

    def append(self, row: Mapping[str, object], field_order: Sequence[str] | None = None) -> None:
        self._append_csv(row, field_order)
        self._append_jsonl(row)

    def append_json(self, row: Mapping[str, object]) -> None:
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
            with open(self.csv_path, "a", newline="", encoding="utf-8") as h:
                writer = csv.DictWriter(h, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            raise RuntimeError("Failed to append metric row to file") from exc

    def _append_jsonl(self, row: Mapping[str, object]) -> None:
        if not self.jsonl_path:
            return
        try:
            directory = os.path.dirname(self.jsonl_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            safe_row = {str(k): _json_safe(v) for k, v in row.items()}
            with open(self.jsonl_path, "a", encoding="utf-8") as h:
                json.dump(safe_row, h, separators=(",", ":"))
                h.write("\n")
        except Exception as exc:
            raise RuntimeError("Failed to append metric row to JSONL file") from exc


def select_visit_count_move(
    visit_counts: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if visit_counts.ndim != 1 or visit_counts.size == 0:
        return 0
    if temperature <= C.SELFPLAY.deterministic_temp_eps:
        return int(np.argmax(visit_counts))

    scaled = np.asarray(np.maximum(visit_counts, 0.0), dtype=np.float64)
    scaled = scaled ** (1.0 / max(float(temperature), 1e-6))
    total = float(scaled.sum())
    if not np.isfinite(total) or total <= 0.0:
        return int(np.argmax(visit_counts))

    probabilities = np.asarray(scaled / total, dtype=np.float64)
    try:
        return int(rng.choice(len(probabilities), p=probabilities))
    except Exception:
        return int(np.argmax(visit_counts))


def format_time(seconds: float) -> str:
    v = float(seconds)
    if v < 60:
        return f"{v:.1f}s"
    if v < 3600:
        return f"{v / 60:.1f}m"
    return f"{v / 3600:.1f}h"


def startup_summary(trainer: Any) -> str:
    has_cuda = bool(trainer.device.type == "cuda" and torch.cuda.is_available())
    amp_enabled = bool(getattr(trainer, "_amp_enabled", C.TORCH.amp_enabled))
    autocast_mode = "fp16" if amp_enabled else "disabled"
    params_m = sum(p.numel() for p in trainer.model.parameters()) / 1e6
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device_name = getattr(trainer, "device_name", str(trainer.device))
    lines = [
        f"[{timestamp}] Hybrid Chess AI Training",
        f"device={device_name} ({trainer.device}) cuda={'yes' if has_cuda else 'no'} autocast={autocast_mode} "
        f"params={params_m:.2f}M replay_buffer={C.REPLAY.capacity:,}",
        " ".join(
            [
                f"train_iters={C.TRAIN.total_iterations}",
                f"games/iter={C.TRAIN.games_per_iter}",
                f"batch_size={trainer.train_batch_size}",
                f"lr={C.TRAIN.learning_rate_init:.2e}->{C.TRAIN.learning_rate_final:.2e}",
                f"grad_clip={C.TRAIN.grad_clip_norm}",
            ]
        ),
        " ".join(
            [
                f"selfplay_threads={C.SELFPLAY.num_workers}",
                f"simulations={C.MCTS.train_simulations}->{C.MCTS.train_simulations_min}",
                f"temp_moves={C.SELFPLAY.temperature_moves}",
                f"max_moves={C.SELFPLAY.game_max_plies}",
                f"adj_margin={C.SELFPLAY.adjudication_value_margin}",
                f"adj_warmup={C.SELFPLAY.adjudication_warmup_iters}",
                f"adj_ramp={C.SELFPLAY.adjudication_ramp_iters}",
            ]
        ),
        " ".join(
            [
                f"arena every={C.ARENA.eval_every_iters} iters",
                f"games={C.ARENA.games_per_eval}",
            ]
        ),
        f"metrics_csv={getattr(getattr(trainer, 'metrics', None), 'csv_path', '')}",
    ]
    return " | ".join(lines)
