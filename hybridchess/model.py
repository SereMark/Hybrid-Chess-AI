from __future__ import annotations

import threading
import time
from collections import deque, OrderedDict
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    BLOCKS,
    BOARD_SIZE,
    CHANNELS,
    EVAL_BATCH_TIMEOUT_MS,
    EVAL_CACHE_SIZE,
    EVAL_CHANNELS_LAST,
    EVAL_CLOSE_JOIN_TIMEOUT_S,
    EVAL_MAX_BATCH,
    INPUT_PLANES,
    NSQUARES,
    POLICY_OUTPUT,
    VALUE_CONV_CHANNELS,
    VALUE_HIDDEN_DIM,
)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class ChessNet(nn.Module):
    def __init__(self, blocks: int | None = None, channels: int | None = None) -> None:
        super().__init__()
        blocks = blocks or BLOCKS
        channels = channels or CHANNELS
        policy_planes = POLICY_OUTPUT // NSQUARES
        self.conv_in = nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, policy_planes, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_planes)
        self.policy_fc = nn.Linear(policy_planes * NSQUARES, POLICY_OUTPUT)
        self.value_conv = nn.Conv2d(channels, VALUE_CONV_CHANNELS, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(VALUE_CONV_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_CONV_CHANNELS * NSQUARES, VALUE_HIDDEN_DIM)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN_DIM, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        p = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return p, v


class BatchedEvaluator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model_lock = threading.Lock()
        any_mod: Any = ChessNet().to(self.device)
        self.eval_model: nn.Module = any_mod
        if EVAL_CHANNELS_LAST:
            self.eval_model = cast(
                nn.Module,
                getattr(self.eval_model, "to")(memory_format=torch.channels_last),
            )
        self.eval_model = self.eval_model.eval()
        if self.device.type == "cuda":
            self.eval_model.half()
        for p in self.eval_model.parameters():
            p.requires_grad_(False)
        self.cache_lock = threading.Lock()
        self.cache: OrderedDict[int, tuple[np.ndarray, float | np.floating[Any]]] = OrderedDict()
        self.cache_cap = EVAL_CACHE_SIZE
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, float] = {
            "requests_total": 0,
            "cache_hits_total": 0,
            "cache_misses_total": 0,
            "batches_total": 0,
            "eval_positions_total": 0,
            "batch_size_max": 0,
        }
        self._pending_lock = threading.Condition()
        self._pending: deque[_EvalRequest] = deque()
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._batch_worker, name="EvalBatchWorker", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._shutdown.set()
        with self._pending_lock:
            for r in list(self._pending):
                r.policy = np.zeros((POLICY_OUTPUT,), dtype=np.float16)
                r.value = 0.0
                r.event.set()
            self._pending.clear()
            self._pending_lock.notify_all()
        try:
            self._thread.join(timeout=EVAL_CLOSE_JOIN_TIMEOUT_S)
        except Exception:
            pass
        with self.cache_lock:
            self.cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def refresh_from(self, src_model: torch.nn.Module) -> None:
        with self.model_lock:
            base = getattr(src_model, "_orig_mod", src_model)
            if hasattr(base, "module"):
                base = base.module
            self.eval_model.load_state_dict(base.state_dict(), strict=True)
            if self.device.type == "cuda":
                self.eval_model.half()
            self.eval_model.eval()
            for p in self.eval_model.parameters():
                p.requires_grad_(False)
        with self.cache_lock:
            self.cache.clear()

    def _encode_batch(self, positions: list[Any]) -> torch.Tensor:
        import chesscore as ccore

        x_np = np.zeros((0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32) if not positions else ccore.encode_batch(positions)
        x = torch.from_numpy(x_np)
        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
        return x.contiguous(memory_format=torch.channels_last) if x.dim() == 4 else x.contiguous()

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self.model_lock, torch.inference_mode():
            if self.device.type == "cuda":
                x = x.to(dtype=torch.float16)
            return cast(tuple[torch.Tensor, torch.Tensor], self.eval_model(x))

    def _position_key(self, pos: Any) -> int | None:
        try:
            h_attr = getattr(pos, "hash", None)
            if callable(h_attr):
                h_val = h_attr()
            else:
                h_val = h_attr
            if h_val is None:
                return None
            if isinstance(h_val, np.generic):
                h_val = h_val.item()
            if isinstance(h_val, (int, np.integer)):
                return int(h_val)
            if isinstance(h_val, str):
                try:
                    return int(h_val)
                except Exception:
                    return None
            return None
        except Exception:
            return None

    def infer_positions(self, positions: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        if not positions:
            return np.zeros((0, POLICY_OUTPUT), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        with self._metrics_lock:
            self._metrics["requests_total"] += len(positions)
        cached_p: list[np.ndarray | None] = [None] * len(positions)
        cached_v: list[float] = [0.0] * len(positions)
        reqs: list[tuple[int, _EvalRequest]] = []
        uniq_by_key: dict[int, int] = {}
        uniq_by_obj: dict[int, int] = {}
        misses = 0
        with self.cache_lock:
            for i, pos in enumerate(positions):
                key = self._position_key(pos)
                if key is not None and key in self.cache:
                    pol, val = self.cache.pop(key)
                    self.cache[key] = (pol, val)
                    cached_p[i] = pol
                    cached_v[i] = float(val)
                    with self._metrics_lock:
                        self._metrics["cache_hits_total"] += 1
                else:
                    if key is not None:
                        if key in uniq_by_key:
                            reqs.append((i, reqs[uniq_by_key[key]][1]))
                        else:
                            r = _EvalRequest(pos)
                            uniq_by_key[key] = len(reqs)
                            reqs.append((i, r))
                            misses += 1
                    else:
                        obj_id = id(pos)
                        if obj_id in uniq_by_obj:
                            reqs.append((i, reqs[uniq_by_obj[obj_id]][1]))
                        else:
                            r = _EvalRequest(pos)
                            uniq_by_obj[obj_id] = len(reqs)
                            reqs.append((i, r))
        unique_reqs: list[_EvalRequest] = []
        seen: set[int] = set()
        for _, r in reqs:
            if id(r) not in seen:
                unique_reqs.append(r)
                seen.add(id(r))
        with self._metrics_lock:
            self._metrics["cache_misses_total"] += misses
        for r in unique_reqs:
            with self._pending_lock:
                self._pending.append(r)
                self._pending_lock.notify()
        for idx, r in reqs:
            r.event.wait()
            cached_p[idx] = r.policy
            cached_v[idx] = r.value
        if reqs:
            with self.cache_lock:
                for idx, r in reqs:
                    key = self._position_key(positions[idx])
                    if key is not None and key not in self.cache:
                        pol_opt = cached_p[idx]
                        val = cached_v[idx]
                        if pol_opt is not None and not getattr(r, "error", False):
                            self.cache[key] = (
                                pol_opt.astype(np.float16),
                                np.float16(val),
                            )
                while len(self.cache) > self.cache_cap:
                    self.cache.popitem(last=False)
        pol_np = np.stack([(p if p is not None else np.zeros((POLICY_OUTPUT,), dtype=np.float32)) for p in cached_p]).astype(np.float32)
        val_np = np.asarray(cached_v, dtype=np.float32)
        return pol_np, val_np

    def _batch_worker(self) -> None:
        timeout_s = max(0.0, float(EVAL_BATCH_TIMEOUT_MS) / 1000.0)
        while not self._shutdown.is_set():
            with self._pending_lock:
                while not self._pending and not self._shutdown.is_set():
                    self._pending_lock.wait()
                if self._shutdown.is_set():
                    break
                batch: list[_EvalRequest] = [self._pending.popleft()]
                start = time.time()
                while len(batch) < EVAL_MAX_BATCH and not self._shutdown.is_set():
                    rem = timeout_s - (time.time() - start)
                    if rem <= 0.0:
                        break
                    if not self._pending:
                        self._pending_lock.wait(timeout=rem)
                        if not self._pending:
                            break
                    while self._pending and len(batch) < EVAL_MAX_BATCH:
                        batch.append(self._pending.popleft())
            if self._shutdown.is_set():
                break
            try:
                x = self._encode_batch([r.position for r in batch])
                p_t, v_t = self._forward(x)
                p_logits = p_t.float()
                p_logits = p_logits - p_logits.amax(dim=1, keepdim=True)
                pol = F.softmax(p_logits, dim=1).detach().to(dtype=torch.float16).cpu().numpy()
                val = v_t.detach().to(dtype=torch.float16).cpu().numpy()
                for i, r in enumerate(batch):
                    r.policy = pol[i]
                    r.value = float(val[i])
                    r.error = False
                    r.event.set()
                with self._metrics_lock:
                    self._metrics["batches_total"] += 1
                    self._metrics["eval_positions_total"] += len(batch)
                    if len(batch) > int(self._metrics["batch_size_max"]):
                        self._metrics["batch_size_max"] = float(len(batch))
            except Exception:
                for r in batch:
                    r.policy = np.zeros((POLICY_OUTPUT,), dtype=np.float16)
                    r.value = 0.0
                    r.error = True
                    r.event.set()

    def get_metrics(self) -> dict[str, float]:
        with self._metrics_lock:
            return dict(self._metrics)


class _EvalRequest:
    def __init__(self, position: Any) -> None:
        self.position = position
        self.event = threading.Event()
        self.policy: np.ndarray | None = None
        self.value: float = 0.0
        self.error: bool = False
