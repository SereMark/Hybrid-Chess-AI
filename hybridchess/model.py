from __future__ import annotations

import contextlib
import numbers
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    AMP_PREFER_BFLOAT16,
    BOARD_SIZE,
    EVAL_BATCH_COALESCE_MS,
    EVAL_BATCH_SIZE_MAX,
    EVAL_CACHE_CAPACITY,
    EVAL_CACHE_USE_FP16,
    EVAL_MODEL_CHANNELS_LAST,
    EVAL_PIN_MEMORY,
    EVAL_WORKER_JOIN_TIMEOUT_S,
    INPUT_PLANES,
    MODEL_BLOCKS,
    MODEL_CHANNELS,
    MODEL_VALUE_CONV_CHANNELS,
    MODEL_VALUE_HIDDEN_DIM,
    NSQUARES,
    POLICY_OUTPUT,
)


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU x2 with skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + skip)


class ChessNet(nn.Module):
    """Policy + value network over encoded positions."""

    def __init__(self, num_blocks: int | None = None, channels: int | None = None) -> None:
        super().__init__()
        num_blocks = num_blocks or MODEL_BLOCKS
        channels = channels or MODEL_CHANNELS
        policy_planes = POLICY_OUTPUT // NSQUARES
        # Trunk
        self.conv_in = nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.residual_stack = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        # Policy head
        self.policy_conv = nn.Conv2d(channels, policy_planes, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_planes)
        self.policy_fc = nn.Linear(policy_planes * NSQUARES, POLICY_OUTPUT)
        # Value head
        self.value_conv = nn.Conv2d(channels, MODEL_VALUE_CONV_CHANNELS, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(MODEL_VALUE_CONV_CHANNELS)
        self.value_fc1 = nn.Linear(MODEL_VALUE_CONV_CHANNELS * NSQUARES, MODEL_VALUE_HIDDEN_DIM)
        self.value_fc2 = nn.Linear(MODEL_VALUE_HIDDEN_DIM, 1)
        # Init weights
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
        x = self.residual_stack(x)
        # Policy
        policy_logits = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = policy_logits.flatten(1)
        policy_logits = self.policy_fc(policy_logits)
        # Value
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.flatten(1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)
        return policy_logits, value


@dataclass
class _EvalRequest:
    """Internal single evaluation request container."""

    position: Any
    event: threading.Event = field(default_factory=threading.Event)
    policy: np.ndarray | None = None
    value: float = 0.0
    error: bool = False


class BatchedEvaluator:
    """Threaded batched inference with a small LRU cache."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model_lock = threading.Lock()
        model_any: Any = ChessNet().to(self.device)
        self.eval_model: nn.Module = model_any
        if EVAL_MODEL_CHANNELS_LAST:
            self.eval_model = cast(
                nn.Module,
                self.eval_model.to(memory_format=torch.channels_last),
            )
        self.eval_model = self.eval_model.eval()
        if self.device.type == "cuda":
            use_bf16 = AMP_PREFER_BFLOAT16
            self.eval_model = cast(
                nn.Module,
                self.eval_model.to(dtype=(torch.bfloat16 if use_bf16 else torch.float16)),
            )
        for p in self.eval_model.parameters():
            p.requires_grad_(False)
        # Simple LRU cache
        self.cache_lock = threading.Lock()
        self.cache: OrderedDict[int, tuple[np.ndarray, float | np.floating[Any]]] = OrderedDict()
        self.cache_capacity = EVAL_CACHE_CAPACITY
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
                r.policy = np.zeros((POLICY_OUTPUT,), dtype=(np.float16 if EVAL_CACHE_USE_FP16 else np.float32))
                r.value = 0.0
                r.event.set()
            self._pending.clear()
            self._pending_lock.notify_all()
        with contextlib.suppress(Exception):
            self._thread.join(timeout=EVAL_WORKER_JOIN_TIMEOUT_S)
        with self.cache_lock:
            self.cache.clear()

    def __enter__(self) -> BatchedEvaluator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def refresh_from(self, src_model: torch.nn.Module) -> None:
        with self.model_lock:
            base = getattr(src_model, "_orig_mod", src_model)
            if hasattr(base, "module"):
                base = base.module
            self.eval_model.load_state_dict(base.state_dict(), strict=True)
            if self.device.type == "cuda":
                use_bf16 = AMP_PREFER_BFLOAT16
                self.eval_model = cast(
                    nn.Module,
                    self.eval_model.to(dtype=(torch.bfloat16 if use_bf16 else torch.float16)),
                )
            self.eval_model.eval()
            for p in self.eval_model.parameters():
                p.requires_grad_(False)
        with self.cache_lock:
            self.cache.clear()

    def _encode_batch(self, positions: list[Any]) -> torch.Tensor:
        import chesscore as ccore

        # Encode batch of positions to NCHW float32
        x_np = np.zeros((0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32) if not positions else ccore.encode_batch(positions)
        x = torch.from_numpy(x_np)
        if self.device.type == "cuda":
            if EVAL_PIN_MEMORY:
                x = x.pin_memory()
            x = x.to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
        return x.contiguous(memory_format=torch.channels_last) if x.dim() == 4 else x.contiguous()

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Single forward pass (thread-safe, inference-only)
        with self.model_lock, torch.inference_mode():
            if self.device.type == "cuda":
                x = x.to(dtype=(torch.bfloat16 if (AMP_PREFER_BFLOAT16) else torch.float16))
            return cast(tuple[torch.Tensor, torch.Tensor], self.eval_model(x))

    def _position_key(self, pos: Any) -> int | None:
        """Best-effort stable cache key; prefers position.hash()."""
        try:
            h_attr = getattr(pos, "hash", None)
            h_val = h_attr() if callable(h_attr) else h_attr
            if h_val is None:
                return None
            if isinstance(h_val, np.generic):
                h_val = cast(np.generic, h_val).item()
            if isinstance(h_val, numbers.Integral):
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
        """Return (policy, value) arrays with caching and de-duplication."""
        if not positions:
            return np.zeros((0, POLICY_OUTPUT), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        with self._metrics_lock:
            self._metrics["requests_total"] += len(positions)
        cached_policy: list[np.ndarray | None] = [None] * len(positions)
        cached_value: list[float] = [0.0] * len(positions)
        index_request_pairs: list[tuple[int, _EvalRequest]] = []  # (original index, shared request)
        key_to_request_index: dict[int, int] = {}
        obj_to_request_index: dict[int, int] = {}
        cache_misses_count = 0
        with self.cache_lock:
            for i, pos in enumerate(positions):
                key = self._position_key(pos)
                if key is not None and key in self.cache:
                    pol, val = self.cache.pop(key)
                    self.cache[key] = (pol, val)  # recency bump (LRU-ish)
                    cached_policy[i] = pol
                    cached_value[i] = float(val)
                    with self._metrics_lock:
                        self._metrics["cache_hits_total"] += 1
                else:
                    if key is not None:
                        if key in key_to_request_index:
                            index_request_pairs.append((i, index_request_pairs[key_to_request_index[key]][1]))
                        else:
                            r = _EvalRequest(pos)
                            key_to_request_index[key] = len(index_request_pairs)
                            index_request_pairs.append((i, r))
                            cache_misses_count += 1
                    else:
                        obj_id = id(pos)
                        if obj_id in obj_to_request_index:
                            index_request_pairs.append((i, index_request_pairs[obj_to_request_index[obj_id]][1]))
                        else:
                            r = _EvalRequest(pos)
                            obj_to_request_index[obj_id] = len(index_request_pairs)
                            index_request_pairs.append((i, r))
        unique_requests: list[_EvalRequest] = []
        seen: set[int] = set()
        for _, r in index_request_pairs:
            if id(r) not in seen:
                unique_requests.append(r)
                seen.add(id(r))
        with self._metrics_lock:
            self._metrics["cache_misses_total"] += cache_misses_count
        for r in unique_requests:
            with self._pending_lock:
                self._pending.append(r)
                self._pending_lock.notify()
        for idx, r in index_request_pairs:
            r.event.wait()
            cached_policy[idx] = r.policy
            cached_value[idx] = r.value
        if index_request_pairs:
            with self.cache_lock:
                for idx, r in index_request_pairs:
                    key = self._position_key(positions[idx])
                    if key is not None and key not in self.cache:
                        policy_opt = cached_policy[idx]
                        value_scalar = cached_value[idx]
                        if policy_opt is not None and not getattr(r, "error", False):
                            if EVAL_CACHE_USE_FP16:
                                self.cache[key] = (
                                    policy_opt.astype(np.float16),
                                    np.float16(value_scalar),
                                )
                            else:
                                self.cache[key] = (
                                    policy_opt.astype(np.float32),
                                    np.float32(value_scalar),
                                )
                while len(self.cache) > self.cache_capacity:
                    self.cache.popitem(last=False)
        policy = np.stack([(p if p is not None else np.zeros((POLICY_OUTPUT,), dtype=np.float32)) for p in cached_policy]).astype(np.float32)
        value = np.asarray(cached_value, dtype=np.float32)
        return policy, value

    def _batch_worker(self) -> None:
        """Coalesce pending requests into batches up to size/time limits."""
        coalesce_timeout_s = max(0.0, float(EVAL_BATCH_COALESCE_MS) / 1000.0)
        while not self._shutdown.is_set():
            with self._pending_lock:
                while not self._pending and not self._shutdown.is_set():
                    self._pending_lock.wait()
                if self._shutdown.is_set():
                    break
                batch: list[_EvalRequest] = [self._pending.popleft()]
                coalesce_start = time.time()
                while len(batch) < EVAL_BATCH_SIZE_MAX and not self._shutdown.is_set():
                    remaining = coalesce_timeout_s - (time.time() - coalesce_start)
                    if remaining <= 0.0:
                        break
                    if not self._pending:
                        self._pending_lock.wait(timeout=remaining)
                        if not self._pending:
                            break
                    while self._pending and len(batch) < EVAL_BATCH_SIZE_MAX:
                        batch.append(self._pending.popleft())
            if self._shutdown.is_set():
                break
            try:
                x = self._encode_batch([r.position for r in batch])
                policy_out_t, value_out_t = self._forward(x)
                policy_logits = policy_out_t.float()
                policy_logits = policy_logits - policy_logits.amax(dim=1, keepdim=True)
                out_dtype = torch.float16 if EVAL_CACHE_USE_FP16 else torch.float32
                policies_np = F.softmax(policy_logits, dim=1).detach().to(dtype=out_dtype).cpu().numpy()
                values_np = value_out_t.detach().to(dtype=out_dtype).cpu().numpy()
                for i, r in enumerate(batch):
                    r.policy = policies_np[i]
                    r.value = float(values_np[i])
                    r.error = False
                    r.event.set()
                with self._metrics_lock:
                    self._metrics["batches_total"] += 1
                    self._metrics["eval_positions_total"] += len(batch)
                    if len(batch) > int(self._metrics["batch_size_max"]):
                        self._metrics["batch_size_max"] = float(len(batch))
            except Exception:
                for r in batch:
                    r.policy = np.zeros((POLICY_OUTPUT,), dtype=(np.float16 if EVAL_CACHE_USE_FP16 else np.float32))
                    r.value = 0.0
                    r.error = True
                    r.event.set()

    def get_metrics(self) -> dict[str, float]:
        """Snapshot of internal evaluator counters."""
        with self._metrics_lock:
            return dict(self._metrics)
