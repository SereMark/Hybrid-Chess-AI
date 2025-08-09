from __future__ import annotations

import threading
from collections import deque
from queue import Empty, Queue
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONFIG


def unwrap_compiled_module(module: nn.Module) -> nn.Module:
    m = module
    while hasattr(m, "_orig_mod"):
        m = getattr(m, "_orig_mod")  # type: ignore[attr-defined]
    return m


def strip_orig_mod_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if keys and keys[0].startswith("_orig_mod."):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def get_module_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    base = unwrap_compiled_module(module)
    return base.state_dict()


def load_module_state_dict(
    module: nn.Module, state_dict: dict[str, torch.Tensor], strict: bool = True
) -> None:
    base = unwrap_compiled_module(module)
    cleaned = strip_orig_mod_prefix(state_dict)
    base.load_state_dict(cleaned, strict=strict)


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

        self.value_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)

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

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _EvalRequest:
    def __init__(self, position: Any) -> None:
        self.position = position
        try:
            self.hash: int | None = int(getattr(position, "hash", 0))
        except Exception:
            self.hash = None
        self.policy: np.ndarray | None = None
        self.value: float | None = None
        self.event = threading.Event()


class BatchedEvaluator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model_lock = threading.Lock()
        self.eval_model = ChessNet().to(self.device)
        if CONFIG.use_torch_compile:
            self.eval_model = cast(torch.nn.Module, torch.compile(self.eval_model))
        self.eval_model.eval()
        self.queue: "Queue[_EvalRequest]" = Queue()
        self.max_batch = CONFIG.eval_max_batch
        self.timeout_ms = CONFIG.eval_batch_timeout_ms
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.cache_lock = threading.Lock()
        self.cache: dict[int, tuple[np.ndarray, float]] = {}
        self.cache_order: deque[int] = deque()
        self.cache_cap = CONFIG.eval_cache_size

    def __enter__(self) -> "BatchedEvaluator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def refresh_from(self, src_model: torch.nn.Module) -> None:
        with self.model_lock:
            load_module_state_dict(
                self.eval_model, get_module_state_dict(src_model), strict=True
            )
            self.eval_model.eval()
        with self.cache_lock:
            self.cache.clear()
            self.cache_order.clear()

    def _encode_batch(self, positions: list[Any]) -> torch.Tensor:
        import chesscore as chessai

        if not positions:
            x_np = np.zeros((0, CONFIG.input_planes, 8, 8), dtype=np.float32)
        else:
            x_np = chessai.encode_batch(positions)
        x_cpu = torch.from_numpy(x_np)
        if self.device.type == "cuda":
            x_cpu = x_cpu.pin_memory()
        x = x_cpu.to(self.device, non_blocking=True)
        if CONFIG.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self.model_lock:
            with torch.inference_mode():
                with torch.amp.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
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
        to_run: list[Any] = []
        to_idx: list[int] = []
        with self.cache_lock:
            for i, pos in enumerate(positions):
                h = int(getattr(pos, "hash", 0))
                if h and h in self.cache:
                    pol, val = self.cache[h]
                    cached_policies[i] = pol
                    cached_values[i] = val
                else:
                    to_run.append(pos)
                    to_idx.append(i)
        if to_run:
            x = self._encode_batch(to_run)
            policy_logits_t, values_t = self._forward(x)
            pol_run = torch.softmax(policy_logits_t, dim=1).detach().cpu().numpy()
            val_run = values_t.detach().cpu().numpy()
            with self.cache_lock:
                for j, idx in enumerate(to_idx):
                    cached_policies[idx] = pol_run[j]
                    cached_values[idx] = float(val_run[j])
                    h = int(getattr(positions[idx], "hash", 0))
                    if h:
                        if h not in self.cache:
                            self.cache[h] = (pol_run[j].astype(np.float16), float(val_run[j]))
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

    def _worker(self) -> None:
        while not self.stop_flag.is_set():
            batch: list[_EvalRequest] = []
            deadline = None
            try:
                req = self.queue.get(timeout=self.timeout_ms / 1000.0)
                batch.append(req)
                import time as _time

                deadline = _time.monotonic() + (self.timeout_ms / 1000.0)
                while len(batch) < self.max_batch:
                    remaining = deadline - _time.monotonic() if deadline else 0.0
                    if remaining <= 0:
                        break
                    try:
                        batch.append(self.queue.get(timeout=remaining))
                    except Empty:
                        break
            except Empty:
                continue

            positions = [r.position for r in batch]
            x = self._encode_batch(positions)
            policy_logits, values = self._forward(x)
            policy = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()
            values_np = values.detach().cpu().numpy()
            for i, r in enumerate(batch):
                r.policy = policy[i]
                r.value = float(values_np[i])
                r.event.set()
            with self.cache_lock:
                for i, r in enumerate(batch):
                    h = int(getattr(r.position, "hash", 0))
                    if h and h not in self.cache:
                        self.cache[h] = (policy[i].astype(np.float16), float(values_np[i]))
                        self.cache_order.append(h)
                while len(self.cache_order) > self.cache_cap:
                    k = self.cache_order.popleft()
                    self.cache.pop(k, None)

    def evaluate(self, position: Any) -> tuple[np.ndarray, float]:
        h = int(getattr(position, "hash", 0))
        if h:
            with self.cache_lock:
                if h in self.cache:
                    pol, val = self.cache[h]
                    return pol.astype(np.float32), val
        req = _EvalRequest(position)
        self.queue.put(req)
        req.event.wait()
        assert req.policy is not None and req.value is not None
        return req.policy.astype(np.float32), req.value

    def shutdown(self) -> None:
        self.stop_flag.set()
        self.thread.join(timeout=1.0)
