from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Any

import numpy as np
import torch
from typing import cast

from .config import CONFIG
from .nn import ChessNet


class _EvalRequest:
    def __init__(self, position: Any) -> None:
        self.position = position
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

    def __enter__(self) -> "BatchedEvaluator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def refresh_from(self, src_model: torch.nn.Module) -> None:
        with self.model_lock:
            self.eval_model.load_state_dict(src_model.state_dict(), strict=False)
            self.eval_model.eval()

    def _encode_batch(self, positions: list[Any]) -> torch.Tensor:
        import chessai

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
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    policy_logits, value = self.eval_model(x)
            return policy_logits, value

    def _worker(self) -> None:
        while not self.stop_flag.is_set():
            batch: list[_EvalRequest] = []
            try:
                req = self.queue.get(timeout=self.timeout_ms / 1000.0)
                batch.append(req)
                while len(batch) < self.max_batch:
                    try:
                        batch.append(self.queue.get_nowait())
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

    def evaluate(self, position: Any) -> tuple[np.ndarray, float]:
        req = _EvalRequest(position)
        self.queue.put(req)
        req.event.wait()
        assert req.policy is not None and req.value is not None
        return req.policy, req.value

    def shutdown(self) -> None:
        self.stop_flag.set()
        self.thread.join(timeout=1.0)
