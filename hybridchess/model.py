from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONFIG


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self, blocks: int | None = None, channels: int | None = None) -> None:
        super().__init__()
        blocks = blocks or CONFIG.blocks
        channels = channels or CONFIG.channels
        policy_planes = CONFIG.policy_output // 64

        self.conv_in = nn.Conv2d(
            CONFIG.input_planes, channels, 3, padding=1, bias=False
        )
        self.bn_in = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(channels, policy_planes, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_planes)
        self.policy_fc = nn.Linear(policy_planes * 64, CONFIG.policy_output)

        self.value_conv = nn.Conv2d(channels, CONFIG.value_conv_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(CONFIG.value_conv_channels)
        self.value_fc1 = nn.Linear(
            CONFIG.value_conv_channels * 64, CONFIG.value_hidden_dim
        )
        self.value_fc2 = nn.Linear(CONFIG.value_hidden_dim, 1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.flatten(1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.flatten(1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)

        return policy, value


class BatchedEvaluator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model_lock = threading.Lock()
        self.eval_model: nn.Module = ChessNet().to(self.device)
        self.eval_model.eval()
        self.cache_lock = threading.Lock()
        self.cache: dict[int, tuple[np.ndarray, float]] = {}
        self.cache_order: deque[int] = deque()
        self.cache_cap = CONFIG.eval_cache_size
        self._pending_lock = threading.Condition()
        self._pending: deque["_EvalRequest"] = deque()
        worker = threading.Thread(
            target=self._batch_worker, name="EvalBatchWorker", daemon=True
        )
        worker.start()

    def refresh_from(self, src_model: torch.nn.Module) -> None:
        with self.model_lock:
            self.eval_model.load_state_dict(src_model.state_dict(), strict=True)
            self.eval_model.eval()
        with self.cache_lock:
            self.cache.clear()
            self.cache_order.clear()

    def _encode_batch(self, positions: list[Any]) -> torch.Tensor:
        import chesscore as ccore

        if not positions:
            x_np = np.zeros((0, CONFIG.input_planes, 8, 8), dtype=np.float32)
        else:
            x_np = ccore.encode_batch(positions)
        x_cpu = torch.from_numpy(x_np)
        if CONFIG.eval_pin_memory:
            x_cpu = x_cpu.pin_memory()
        x = x_cpu.to(self.device, non_blocking=True)
        if CONFIG.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self.model_lock:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", enabled=True):
                    policy_logits, value = self.eval_model(x)
            return policy_logits, value

    def infer_positions(self, positions: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        if not positions:
            return (
                np.zeros((0, CONFIG.policy_output), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        cached_policies: list[np.ndarray | None] = [None] * len(positions)
        cached_values: list[float] = [0.0] * len(positions)
        requests: list[tuple[int, _EvalRequest]] = []
        with self.cache_lock:
            for i, pos in enumerate(positions):
                h = int(getattr(pos, "hash", 0))
                if h and h in self.cache:
                    pol, val = self.cache[h]
                    cached_policies[i] = pol
                    cached_values[i] = val
                else:
                    req = _EvalRequest(pos)
                    requests.append((i, req))
        for _, req in requests:
            with self._pending_lock:
                self._pending.append(req)
                self._pending_lock.notify()
        for idx, req in requests:
            req.event.wait()
            cached_policies[idx] = req.policy
            cached_values[idx] = req.value
        if requests:
            with self.cache_lock:
                for idx, _ in requests:
                    h = int(getattr(positions[idx], "hash", 0))
                    if h and h not in self.cache:
                        pol = cached_policies[idx]
                        val = cached_values[idx]
                        if pol is not None:
                            self.cache[h] = (pol.astype(np.float16), float(val))
                            self.cache_order.append(h)
                while len(self.cache_order) > self.cache_cap:
                    k = self.cache_order.popleft()
                    self.cache.pop(k, None)
        pol_np = np.stack(
            [
                (
                    p
                    if p is not None
                    else np.zeros((CONFIG.policy_output,), dtype=np.float32)
                )
                for p in cached_policies
            ]
        ).astype(np.float32)
        val_np = np.asarray(cached_values, dtype=np.float32)
        return pol_np, val_np

    def _batch_worker(self) -> None:
        timeout_s = max(0.0, float(CONFIG.eval_batch_timeout_ms) / 1000.0)
        while True:
            with self._pending_lock:
                while not self._pending:
                    self._pending_lock.wait()
                batch: list[_EvalRequest] = []
                batch.append(self._pending.popleft())
                start_time = time.time()
                while len(batch) < CONFIG.eval_max_batch:
                    remaining = timeout_s - (time.time() - start_time)
                    if remaining <= 0.0:
                        break
                    if not self._pending:
                        self._pending_lock.wait(timeout=remaining)
                        if not self._pending:
                            break
                    while self._pending and len(batch) < CONFIG.eval_max_batch:
                        batch.append(self._pending.popleft())
            positions = [r.position for r in batch]
            x = self._encode_batch(positions)
            policy_logits_t, values_t = self._forward(x)
            pol_np = torch.softmax(policy_logits_t, dim=1).detach().cpu().numpy()
            val_np = values_t.detach().cpu().numpy()
            for i, req in enumerate(batch):
                req.policy = pol_np[i]
                req.value = float(val_np[i])
                req.event.set()


class _EvalRequest:
    def __init__(self, position: Any) -> None:
        self.position = position
        self.event = threading.Event()
        self.policy: np.ndarray | None = None
        self.value: float = 0.0
