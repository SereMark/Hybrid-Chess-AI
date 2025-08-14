from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_PLANES = 119
PLANES_PER_POSITION = 14
HISTORY_LENGTH = 8
POLICY_OUTPUT = 73 * 64
BLOCKS = 8
CHANNELS = 160
VALUE_CONV_CHANNELS = 8
VALUE_HIDDEN_DIM = 512
EVAL_CACHE_SIZE = 20000
EVAL_MAX_BATCH = 1024
EVAL_BATCH_TIMEOUT_MS = 2


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
        policy_planes = POLICY_OUTPUT // 64
        self.conv_in = nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, policy_planes, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_planes)
        self.policy_fc = nn.Linear(policy_planes * 64, POLICY_OUTPUT)
        self.value_conv = nn.Conv2d(channels, VALUE_CONV_CHANNELS, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(VALUE_CONV_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_CONV_CHANNELS * 64, VALUE_HIDDEN_DIM)
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
        self.eval_model: nn.Module = any_mod.to(
            memory_format=torch.channels_last
        ).eval()
        if self.device.type == "cuda":
            self.eval_model.half()
        for p in self.eval_model.parameters():
            p.requires_grad_(False)
        self.cache_lock = threading.Lock()
        self.cache: dict[int, tuple[np.ndarray, float | np.floating[Any]]] = {}
        self.cache_order: deque[int] = deque()
        self.cache_cap = EVAL_CACHE_SIZE
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, float] = {
            "requests_total": 0,
            "cache_hits_total": 0,
            "cache_misses_total": 0,
            "queued_unique_total": 0,
            "batches_total": 0,
            "eval_positions_total": 0,
            "batch_size_max": 0,
            "pending_queue_max": 0,
            "encode_time_s_total": 0.0,
            "forward_time_s_total": 0.0,
            "wait_time_s_total": 0.0,
        }
        self._pending_lock = threading.Condition()
        self._pending: deque[_EvalRequest] = deque()
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._batch_worker, name="EvalBatchWorker", daemon=True
        )
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
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        with self.cache_lock:
            self.cache.clear()
            self.cache_order.clear()

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
            self.cache_order.clear()

    def _encode_batch(self, positions: list[Any]) -> torch.Tensor:
        import chesscore as ccore

        x_np = (
            np.zeros((0, INPUT_PLANES, 8, 8), dtype=np.float32)
            if not positions
            else ccore.encode_batch(positions)
        )
        x = torch.from_numpy(x_np).pin_memory().to(self.device, non_blocking=True)
        return x.contiguous(memory_format=torch.channels_last)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with (
            self.model_lock,
            torch.inference_mode(),
            torch.autocast(device_type="cuda", enabled=(self.device.type == "cuda")),
        ):
            x = (
                x.contiguous(memory_format=torch.channels_last)
                if x.dim() == 4
                else x.contiguous()
            )
            if self.device.type == "cuda":
                x = x.to(dtype=torch.float16)
            return self.eval_model(x)

    def infer_positions(self, positions: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        if not positions:
            return np.zeros((0, POLICY_OUTPUT), dtype=np.float32), np.zeros(
                (0,), dtype=np.float32
            )
        with self._metrics_lock:
            self._metrics["requests_total"] += len(positions)
        cached_p: list[np.ndarray | None] = [None] * len(positions)
        cached_v: list[float] = [0.0] * len(positions)
        reqs: list[tuple[int, _EvalRequest]] = []
        uniq: dict[int, int] = {}
        with self.cache_lock:
            for i, pos in enumerate(positions):
                h = int(getattr(pos, "hash", 0))
                if h and h in self.cache:
                    pol, val = self.cache[h]
                    cached_p[i] = pol
                    cached_v[i] = float(val)
                    with self._metrics_lock:
                        self._metrics["cache_hits_total"] += 1
                else:
                    if h and h in uniq:
                        reqs.append((i, reqs[uniq[h]][1]))
                    else:
                        r = _EvalRequest(pos)
                        uniq[h] = len(reqs)
                        reqs.append((i, r))
        unique_reqs: list[_EvalRequest] = []
        seen: set[int] = set()
        for _, r in reqs:
            if id(r) not in seen:
                unique_reqs.append(r)
                seen.add(id(r))
        with self._metrics_lock:
            self._metrics["cache_misses_total"] += len(unique_reqs)
            self._metrics["queued_unique_total"] += len(unique_reqs)
        for r in unique_reqs:
            with self._pending_lock:
                r.t_submit = time.time()
                self._pending.append(r)
                self._pending_lock.notify()
        for idx, r in reqs:
            r.event.wait()
            cached_p[idx] = r.policy
            cached_v[idx] = r.value
        if reqs:
            with self.cache_lock:
                for idx, _ in reqs:
                    h = int(getattr(positions[idx], "hash", 0))
                    if h and h not in self.cache:
                        pol_opt = cached_p[idx]
                        val = cached_v[idx]
                        if pol_opt is not None:
                            self.cache[h] = (
                                pol_opt.astype(np.float16),
                                np.float16(val),
                            )
                            self.cache_order.append(h)
                while len(self.cache_order) > self.cache_cap:
                    k = self.cache_order.popleft()
                    self.cache.pop(k, None)
        pol_np = np.stack(
            [
                (p if p is not None else np.zeros((POLICY_OUTPUT,), dtype=np.float32))
                for p in cached_p
            ]
        ).astype(np.float32)
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
                with self._metrics_lock:
                    if len(self._pending) > int(self._metrics["pending_queue_max"]):
                        self._metrics["pending_queue_max"] = float(len(self._pending))
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
            t_enc0 = time.time()
            x = self._encode_batch([r.position for r in batch])
            t_enc1 = time.time()
            p_t, v_t = self._forward(x)
            t_fwd1 = time.time()
            pol = (
                torch.softmax(p_t, dim=1).detach().to(dtype=torch.float16).cpu().numpy()
            )
            val = v_t.detach().to(dtype=torch.float16).cpu().numpy()
            now = time.time()
            accum_wait = 0.0
            for i, r in enumerate(batch):
                r.policy = pol[i]
                r.value = float(val[i])
                if hasattr(r, "t_submit"):
                    accum_wait += max(0.0, now - float(getattr(r, "t_submit", now)))
                r.event.set()
            with self._metrics_lock:
                self._metrics["batches_total"] += 1
                self._metrics["eval_positions_total"] += len(batch)
                if len(batch) > int(self._metrics["batch_size_max"]):
                    self._metrics["batch_size_max"] = float(len(batch))
                self._metrics["encode_time_s_total"] += max(0.0, t_enc1 - t_enc0)
                self._metrics["forward_time_s_total"] += max(0.0, t_fwd1 - t_enc1)
                self._metrics["wait_time_s_total"] += float(accum_wait)

    def get_metrics(self) -> dict[str, float]:
        with self._metrics_lock:
            return dict(self._metrics)


class _EvalRequest:
    def __init__(self, position: Any) -> None:
        self.position = position
        self.event = threading.Event()
        self.policy: np.ndarray | None = None
        self.value: float = 0.0
        self.t_submit: float = 0.0
