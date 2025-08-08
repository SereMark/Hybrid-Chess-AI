from __future__ import annotations

import threading
from queue import Queue, Empty
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import autocast

from .config import CONFIG
from .nn import ChessNet


class PositionEncoder:
    def __init__(self) -> None:
        self.planes = CONFIG.input_planes
        self.history_length = CONFIG.history_length
        self.planes_per_position = CONFIG.planes_per_position
        self.turn_plane = self.history_length * self.planes_per_position
        self.fullmove_plane = self.turn_plane + 1
        self.castling_start_plane = self.fullmove_plane + 1
        self.halfmove_plane = self.castling_start_plane + 4

    def encode_single(
        self, position: Any, history: list[Any] | None = None
    ) -> np.ndarray:
        import chessai

        tensor = np.zeros((CONFIG.input_planes, 8, 8), dtype=np.float32)
        positions = [position] + (history[: self.history_length - 1] if history else [])
        while len(positions) < self.history_length:
            positions.append(positions[-1] if positions else position)

        for t in range(self.history_length):
            pos = positions[t] if t < len(positions) else positions[-1]
            base_plane = t * self.planes_per_position

            for piece in range(6):
                for color in range(2):
                    plane_idx = base_plane + piece * 2 + color
                    bb = pos.pieces[piece][color]
                    while bb:
                        square = (bb & -bb).bit_length() - 1
                        bb &= bb - 1
                        row, col = divmod(square, 8)
                        tensor[plane_idx, row, col] = 1.0

            if t < len(positions):
                repetitions = self._count_reps(pos, positions[: t + 1])
                if repetitions >= 1:
                    tensor[base_plane + 12] = 1.0
                if repetitions >= 2:
                    tensor[base_plane + 13] = 1.0

        if position.turn == chessai.WHITE:
            tensor[self.turn_plane] = 1.0

        tensor[self.fullmove_plane] = min(position.fullmove / 100.0, 1.0)

        castling_rights = [position.castling & (1 << i) for i in range(4)]
        for i, has_right in enumerate(castling_rights):
            if has_right:
                tensor[self.castling_start_plane + i] = 1.0

        tensor[self.halfmove_plane] = min(position.halfmove / 100.0, 1.0)
        return tensor

    def _count_reps(self, target_pos: Any, history: list[Any]) -> int:
        thash = target_pos.hash
        return sum(1 for pos in history if pos.hash == thash) - 1

    def encode_batch(
        self, positions: list[Any], histories: list[list[Any]] | None = None
    ) -> np.ndarray:
        if not positions:
            return np.zeros((0, self.planes, 8, 8), dtype=np.float32)

        batch_size = len(positions)
        tensor = np.zeros((batch_size, self.planes, 8, 8), dtype=np.float32)
        for i, position in enumerate(positions):
            history = histories[i] if histories else None
            tensor[i] = self.encode_single(position, history)
        return tensor


class _EvalRequest:
    def __init__(self, position: Any) -> None:
        self.position = position
        self.policy: np.ndarray | None = None
        self.value: float | None = None
        self.event = threading.Event()


class BatchedEvaluator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.encoder = PositionEncoder()
        self.model_lock = threading.Lock()
        self.eval_model = ChessNet().to(self.device)
        self.eval_model.eval()
        self.queue: "Queue[_EvalRequest]" = Queue()
        self.max_batch = CONFIG.eval_max_batch
        self.timeout_ms = CONFIG.eval_batch_timeout_ms
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def refresh_from(self, src_model: torch.nn.Module) -> None:
        with self.model_lock:
            self.eval_model.load_state_dict(src_model.state_dict(), strict=False)
            self.eval_model.eval()

    def _encode_batch(self, positions: list[Any]) -> torch.Tensor:
        x_np = self.encoder.encode_batch(positions)
        x = torch.from_numpy(x_np).to(self.device, non_blocking=True)
        if CONFIG.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self.model_lock:
            with torch.no_grad():
                with autocast(enabled=self.device.type == "cuda"):
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
