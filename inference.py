from __future__ import annotations

import dataclasses
import numbers
import threading
import time
from collections import OrderedDict, deque
from contextlib import suppress
from typing import Any, cast

import chesscore as ccore
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import config as C
from network import BOARD_SIZE, INPUT_PLANES, POLICY_OUTPUT, ChessNet


@dataclasses.dataclass
class _EvalRequest:
    position: Any
    event: threading.Event = dataclasses.field(default_factory=threading.Event)
    policy: np.ndarray | None = None
    value: float = 0.0
    error: bool = False


class BatchedEvaluator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model_lock = threading.Lock()
        model_any: Any = ChessNet().to(self.device)
        self.eval_model: nn.Module = model_any
        if C.TORCH.EVAL_MODEL_CHANNELS_LAST:
            self.eval_model = cast(
                nn.Module,
                self.eval_model.to(memory_format=torch.channels_last),
            )
        self.eval_model = self.eval_model.eval()
        self.eval_model = cast(nn.Module, self.eval_model.to(dtype=torch.float16))
        for p in self.eval_model.parameters():
            p.requires_grad_(False)
        self.cache_lock = threading.Lock()
        self.cache: OrderedDict[int, tuple[np.ndarray, float | np.floating[Any]]] = OrderedDict()
        self._cache_capacity = int(C.EVAL.CACHE_CAPACITY)
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
        self._batch_size_cap = int(C.EVAL.BATCH_SIZE_MAX)
        self._coalesce_ms = int(C.EVAL.COALESCE_MS)
        self._thread = threading.Thread(target=self._batch_worker, name="EvalBatchWorker", daemon=True)
        self._thread.start()

    def set_batching_params(self, batch_size_max: int | None = None, coalesce_ms: int | None = None) -> None:
        if batch_size_max is not None:
            self._batch_size_cap = int(max(1, batch_size_max))
        if coalesce_ms is not None:
            self._coalesce_ms = int(max(0, coalesce_ms))

    def close(self) -> None:
        self._shutdown.set()
        with self._pending_lock:
            for r in list(self._pending):
                r.policy = np.zeros(
                    (POLICY_OUTPUT,),
                    dtype=(np.float16 if C.EVAL.CACHE_USE_FP16 else np.float32),
                )
                r.value = 0.0
                r.event.set()
            self._pending.clear()
            self._pending_lock.notify_all()

        with suppress(Exception):
            self._thread.join(timeout=C.EVAL.WORKER_JOIN_TIMEOUT_S)
        with self.cache_lock:
            self.cache.clear()

    def set_cache_capacity(self, capacity: int) -> None:
        cap = int(max(0, capacity))
        with self.cache_lock:
            self._cache_capacity = cap
            while len(self.cache) > self._cache_capacity:
                self.cache.popitem(last=False)

    @property
    def cache_capacity(self) -> int:
        return int(self._cache_capacity)

    @cache_capacity.setter
    def cache_capacity(self, value: int) -> None:
        self.set_cache_capacity(int(value))

    def __enter__(self) -> BatchedEvaluator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self):
        self.close()

    def refresh_from(self, src_model: torch.nn.Module) -> None:
        with self.model_lock:
            base = getattr(src_model, "_orig_mod", src_model)
            if hasattr(base, "module"):
                base = base.module
            self.eval_model.load_state_dict(base.state_dict(), strict=True)
            self.eval_model = cast(nn.Module, self.eval_model.to(dtype=torch.float16))
            self.eval_model.eval()
            for p in self.eval_model.parameters():
                p.requires_grad_(False)
        with self.cache_lock:
            self.cache.clear()

    def _encode_batch(self, positions: list[Any]) -> torch.Tensor:
        x_np = (
            np.zeros((0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            if not positions
            else ccore.encode_batch(positions)
        )
        x = torch.from_numpy(x_np)
        if C.TORCH.EVAL_PIN_MEMORY:
            x = x.pin_memory()
        x = x.to(self.device, non_blocking=True)
        return x.contiguous(memory_format=torch.channels_last) if x.dim() == 4 else x.contiguous()

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self.model_lock, torch.inference_mode():
            x = x.to(dtype=torch.float16)
            return cast(tuple[torch.Tensor, torch.Tensor], self.eval_model(x))

    def _position_key(self, pos: Any) -> int | None:
        try:
            h_attr = getattr(pos, "hash", None)
            h_val = h_attr() if callable(h_attr) else h_attr
            if h_val is None:
                return None
            if isinstance(h_val, np.generic):
                h_val = cast(Any, h_val).item()
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
        if not positions:
            return np.zeros((0, POLICY_OUTPUT), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        with self._metrics_lock:
            self._metrics["requests_total"] += len(positions)
        cached_policy: list[np.ndarray | None] = [None] * len(positions)
        cached_value: list[float] = [0.0] * len(positions)
        index_request_pairs: list[tuple[int, _EvalRequest]] = []
        key_to_request_index: dict[int, int] = {}
        obj_to_request_index: dict[int, int] = {}
        cache_misses_count = 0
        with self.cache_lock:
            for i, pos in enumerate(positions):
                key = self._position_key(pos)
                if key is not None and key in self.cache:
                    pol, val = self.cache.pop(key)
                    self.cache[key] = (pol, val)
                    cached_policy[i] = pol
                    cached_value[i] = float(val)
                    with self._metrics_lock:
                        self._metrics["cache_hits_total"] += 1
                elif key is not None:
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
                            if C.EVAL.CACHE_USE_FP16:
                                self.cache[key] = (policy_opt.astype(np.float16), np.float16(value_scalar))
                            else:
                                self.cache[key] = (policy_opt.astype(np.float32), np.float32(value_scalar))
                while len(self.cache) > self._cache_capacity:
                    self.cache.popitem(last=False)
        policy = np.stack(
            [(p if p is not None else np.zeros((POLICY_OUTPUT,), dtype=np.float32)) for p in cached_policy]
        ).astype(np.float32)
        value = np.asarray(cached_value, dtype=np.float32)
        return policy, value

    def _batch_worker(self) -> None:
        while not self._shutdown.is_set():
            with self._pending_lock:
                while not self._pending and not self._shutdown.is_set():
                    self._pending_lock.wait()
                if self._shutdown.is_set():
                    break
                batch: list[_EvalRequest] = [self._pending.popleft()]
                coalesce_start = time.time()
                coalesce_timeout_s = max(0.0, float(self._coalesce_ms) / 1000.0)
                while len(batch) < self._batch_size_cap and not self._shutdown.is_set():
                    remaining = coalesce_timeout_s - (time.time() - coalesce_start)
                    if remaining <= 0.0:
                        break
                    if not self._pending:
                        self._pending_lock.wait(timeout=remaining)
                        if not self._pending:
                            break
                    while self._pending and len(batch) < self._batch_size_cap:
                        batch.append(self._pending.popleft())
            if self._shutdown.is_set():
                break
            try:
                x = self._encode_batch([r.position for r in batch])
                policy_out_t, value_out_t = self._forward(x)
                policy_logits = policy_out_t.float()
                policy_logits = policy_logits - policy_logits.amax(dim=1, keepdim=True)
                out_dtype = torch.float16 if C.EVAL.CACHE_USE_FP16 else torch.float32
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
                    r.policy = np.zeros(
                        (POLICY_OUTPUT,),
                        dtype=(np.float16 if C.EVAL.CACHE_USE_FP16 else np.float32),
                    )
                    r.value = 0.0
                    r.error = True
                    r.event.set()

    def get_metrics(self) -> dict[str, float]:
        with self._metrics_lock:
            return dict(self._metrics)
