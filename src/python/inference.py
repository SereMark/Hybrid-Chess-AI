"""Asynchronous batched evaluator for Hybrid Chess inference."""

from __future__ import annotations

import numbers
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from typing import Any, Iterable, Optional, cast

import config as C
import encoder
import numpy as np
import torch
from network import BOARD_SIZE, INPUT_PLANES, ChessNet
from torch import nn
from utils import prepare_model, select_inference_dtype

__all__ = ["BatchedEvaluator"]


class BatchedEvaluator:
    """Thread-safe evaluator that batches encode requests and caches results."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model_lock = threading.Lock()

        self._dtype: torch.dtype = select_inference_dtype(self.device)
        self.eval_model: nn.Module = prepare_model(
            ChessNet(),
            self.device,
            channels_last=C.TORCH.eval_model_channels_last,
            dtype=self._dtype,
            eval_mode=True,
            freeze=True,
        )

        self._np_dtype: Any
        self._cache_dtype: torch.dtype

        if self._dtype == torch.bfloat16:
            self._np_dtype = np.float32
            self._cache_dtype = torch.float32
        elif self._dtype == torch.float16:
            self._np_dtype = np.float16
            self._cache_dtype = torch.float16 if C.EVAL.use_fp16_cache else torch.float32
        else:
            self._np_dtype = np.float32
            self._cache_dtype = torch.float32

        self.cache_lock = threading.Lock()
        self._val_cap = int(max(1, C.EVAL.value_cache_capacity))
        self._enc_cap = int(max(1, C.EVAL.encode_cache_capacity))
        self._out_cap = int(max(1, C.EVAL.cache_capacity))
        self._val_cache: OrderedDict[int, float] = OrderedDict()
        self._enc_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._out_cache: OrderedDict[int, tuple[np.ndarray, float]] = OrderedDict()

        self._batch_cap = int(C.EVAL.batch_size_max)
        self._coalesce_ms = int(C.EVAL.coalesce_ms)
        self._shutdown = threading.Event()
        self._cv = threading.Condition()
        self._queue: list[_EvalRequest] = []
        self._thread = threading.Thread(target=self._coalesce_loop, name="EvalCoalesce", daemon=True)
        self._thread.start()

        self._m_lock = threading.Lock()
        self._metrics = {
            "requests_total": 0.0,
            "cache_hits_total": 0.0,
            "cache_misses_total": 0.0,
            "batches_total": 0.0,
            "eval_positions_total": 0.0,
            "batch_size_max": 0.0,
        }

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self._shutdown.set()
        with suppress(Exception), self._cv:
            self._cv.notify_all()
        with suppress(Exception):
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
        with suppress(Exception), self.cache_lock:
            self._enc_cache.clear()
            self._val_cache.clear()
            self._out_cache.clear()

    def __enter__(self) -> "BatchedEvaluator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    # ------------------------------------------------------------------ configuration
    def set_batching_params(self, batch_size_max: Optional[int] = None, coalesce_ms: Optional[int] = None) -> None:
        if batch_size_max is not None:
            self._batch_cap = int(max(1, batch_size_max))
        if coalesce_ms is not None:
            self._coalesce_ms = int(max(0, coalesce_ms))

    def set_cache_capacity(
        self,
        capacity: Optional[int] = None,
        *,
        value_capacity: Optional[int] = None,
        encode_capacity: Optional[int] = None,
    ) -> None:
        if capacity is not None:
            self._out_cap = int(max(1, capacity))
        if value_capacity is not None:
            self._val_cap = int(max(1, value_capacity))
        if encode_capacity is not None:
            self._enc_cap = int(max(1, encode_capacity))

    @property
    def cache_capacity(self) -> int:
        return int(self._out_cap)

    @cache_capacity.setter
    def cache_capacity(self, value: int) -> None:
        self.set_cache_capacity(int(value))

    def clear_caches(self) -> None:
        with self.cache_lock:
            self._enc_cache.clear()
            self._val_cache.clear()
            self._out_cache.clear()

    # ------------------------------------------------------------------ public API
    def refresh_from(self, model: nn.Module) -> None:
        with self.model_lock:
            self.eval_model.load_state_dict(model.state_dict(), strict=True)
            self.clear_caches()

    def infer_positions_legal(
        self, positions: Iterable[Any], moves_per_position: Iterable[Iterable[Any]]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        positions_list = list(positions)
        moves_list = [list(moves) for moves in moves_per_position]
        if len(positions_list) != len(moves_list):
            raise ValueError("positions and moves must have the same length")
        with self.model_lock:
            return self._infer_positions_legal_locked(positions_list, moves_list)

    def infer_values(self, positions: Iterable[Any]) -> np.ndarray:
        positions_list = list(positions)
        with self.model_lock:
            return self._infer_values_locked(positions_list)

    # ------------------------------------------------------------------ metrics
    def metrics_snapshot(self) -> dict[str, float]:
        with self._m_lock:
            return {k: float(v) for k, v in self._metrics.items()}

    def get_metrics(self) -> dict[str, float]:
        """Compatibility shim returning internal metrics."""
        return self.metrics_snapshot()

    # ------------------------------------------------------------------ internals (locked by model_lock)
    def _infer_positions_legal_locked(
        self, positions: list[Any], moves_per_position: list[list[Any]]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        keys = [self._position_key(pos) for pos in positions]
        probs_out: list[np.ndarray] = [np.zeros((len(moves),), dtype=np.float32) for moves in moves_per_position]
        values_out = np.zeros((len(positions),), dtype=np.float32)

        hits: list[int] = []
        misses: list[int] = []
        with self.cache_lock:
            for idx, key in enumerate(keys):
                if key is not None and key in self._out_cache:
                    logits_cached, value_cached = self._out_cache[key]
                    moves = moves_per_position[idx]
                    probs_out[idx] = self._probs_from_cached_logits(logits_cached, moves)
                    values_out[idx] = float(value_cached)
                    hits.append(idx)
                else:
                    misses.append(idx)

        if misses:
            miss_positions = [positions[idx] for idx in misses]
            miss_moves = [moves_per_position[idx] for idx in misses]
            probs_computed, values_computed = self._infer_and_cache(
                miss_positions, miss_moves, [keys[idx] for idx in misses]
            )
            for dst, probs, val in zip(misses, probs_computed, values_computed, strict=False):
                probs_out[dst] = probs.astype(np.float32, copy=False)
                values_out[dst] = float(val)

        if hits:
            with self._m_lock:
                self._metrics["cache_hits_total"] += len(hits)

        with self._m_lock:
            self._metrics["requests_total"] += len(positions)

        return probs_out, values_out

    def _infer_values_locked(self, positions: list[Any]) -> np.ndarray:
        keys = [self._position_key(pos) for pos in positions]
        out = np.zeros((len(positions),), dtype=np.float32)
        hits: list[int] = []
        misses: list[int] = []

        with self.cache_lock:
            for idx, key in enumerate(keys):
                if key is not None and key in self._val_cache:
                    out[idx] = float(self._val_cache[key])
                    hits.append(idx)
                else:
                    misses.append(idx)

        if misses:
            enc = self._encode_positions([positions[idx] for idx in misses])
            batch = torch.from_numpy(enc).to(self.device, dtype=torch.float32, non_blocking=False)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self._dtype,
                    enabled=self._dtype != torch.float32,
                ):
                    _, values = self.eval_model(batch)
            out_values = values.to(dtype=self._cache_dtype).cpu().numpy()
            with self.cache_lock:
                for dst, value in zip(misses, out_values, strict=False):
                    out[dst] = float(value)
                    key = keys[dst]
                    if key is not None:
                        self._val_cache[key] = float(value)
                        while len(self._val_cache) > self._val_cap:
                            self._val_cache.popitem(last=False)

            with self._m_lock:
                self._metrics["cache_misses_total"] += len(misses)
                self._metrics["eval_positions_total"] += len(misses)
        if hits:
            with self._m_lock:
                self._metrics["cache_hits_total"] += len(hits)
        return out

    def _infer_and_cache(
        self,
        positions: list[Any],
        moves_per_position: list[list[Any]],
        keys: list[Optional[int]],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        enc = self._encode_positions(positions)
        move_indices = [
            np.asarray(encoder.encode_move_indices_batch([moves])[0], dtype=np.int64) for moves in moves_per_position
        ]
        max_moves = max((indices.size for indices in move_indices), default=0)
        indices = np.zeros((len(positions), max_moves), dtype=np.int64)
        mask = np.zeros((len(positions), max_moves), dtype=bool)
        for row, idx_arr in enumerate(move_indices):
            length = idx_arr.size
            indices[row, :length] = idx_arr
            mask[row, :length] = True

        batch_inputs = torch.from_numpy(enc).to(self.device, dtype=torch.float32, non_blocking=False)
        indices_t = torch.from_numpy(indices).to(self.device, dtype=torch.long)
        mask_t = torch.from_numpy(mask).to(self.device, dtype=torch.bool)

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._dtype,
                enabled=self._dtype != torch.float32,
            ):
                logits, values = self.eval_model(batch_inputs)

        logits = logits.float()
        masked = logits.gather(1, indices_t)
        neg_inf = torch.finfo(masked.dtype).min
        masked = torch.where(mask_t, masked, torch.full_like(masked, neg_inf))
        maxv = masked.max(dim=1, keepdim=True).values
        ex = torch.exp(masked - maxv)
        ex = torch.where(mask_t, ex, torch.zeros_like(ex))
        denom = ex.sum(dim=1, keepdim=True).clamp_min(1e-9)
        probs = ex / denom

        probs_np = probs.to(dtype=self._cache_dtype).cpu().numpy()
        values_np = values.to(dtype=self._cache_dtype).cpu().numpy()
        logits_np = logits.to(dtype=self._cache_dtype).cpu().numpy()

        result_probs: list[np.ndarray] = []
        for row, idx_arr in enumerate(move_indices):
            result_probs.append(probs_np[row, : idx_arr.size])

        with self.cache_lock:
            for row, key in enumerate(keys):
                if key is None:
                    continue
                self._out_cache[key] = (logits_np[row].reshape(-1), float(values_np[row]))
                self._val_cache[key] = float(values_np[row])
                while len(self._out_cache) > self._out_cap:
                    self._out_cache.popitem(last=False)
                while len(self._val_cache) > self._val_cap:
                    self._val_cache.popitem(last=False)

        with self._m_lock:
            self._metrics["cache_misses_total"] += len(positions)
            self._metrics["batches_total"] += 1
            self._metrics["eval_positions_total"] += len(positions)
            if len(positions) > self._metrics["batch_size_max"]:
                self._metrics["batch_size_max"] = float(len(positions))

        return result_probs, values_np.astype(np.float32, copy=False)

    def _position_key(self, position: Any) -> Optional[int]:
        try:
            hash_attr = getattr(position, "hash", None)
            value = hash_attr() if callable(hash_attr) else hash_attr
            if value is None:
                return None
            if isinstance(value, np.generic):
                value = cast(Any, value).item()
            if isinstance(value, numbers.Integral):
                return int(value)
            if isinstance(value, str):
                return int(value)
        except Exception:
            return None
        return None

    def _encode_positions(self, positions: list[Any]) -> np.ndarray:
        if not positions:
            return np.zeros((0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=self._np_dtype)
        key_candidates = [self._position_key(pos) for pos in positions]
        with self.cache_lock:
            cached_encodings: list[np.ndarray | None] = []
            missing_positions: list[Any] = []
            missing_indices: list[int] = []
            for idx, key in enumerate(key_candidates):
                if key is not None and key in self._enc_cache:
                    cached_encodings.append(self._enc_cache[key])
                else:
                    missing_positions.append(positions[idx])
                    missing_indices.append(idx)
                    cached_encodings.append(None)

        if missing_positions:
            encoded_missing = encoder.encode_batch(missing_positions).astype(self._np_dtype, copy=False)
            with self.cache_lock:
                for offset, idx in enumerate(missing_indices):
                    arr = encoded_missing[offset]
                    cached_encodings[idx] = arr
                    key = key_candidates[idx]
                    if key is not None:
                        self._enc_cache[key] = arr
                        while len(self._enc_cache) > self._enc_cap:
                            self._enc_cache.popitem(last=False)

        arrays: list[np.ndarray] = []
        for arr in cached_encodings:
            if arr is None:
                raise RuntimeError("failed to encode position")
            arrays.append(arr)
        return np.stack(arrays, axis=0)

    @staticmethod
    def _probs_from_cached_logits(logits: np.ndarray, moves: list[Any]) -> np.ndarray:
        if not moves:
            return np.zeros((0,), dtype=np.float32)

        encoded = encoder.encode_move_indices_batch([moves])[0].astype(np.int64, copy=False)
        probs = np.zeros((len(moves),), dtype=np.float32)
        if encoded.size == 0:
            return probs

        logits_vec = np.asarray(logits, dtype=np.float32).reshape(-1)
        valid = (encoded >= 0) & (encoded < logits_vec.shape[0])
        if not np.any(valid):
            return probs

        selected = logits_vec[encoded[valid]]
        maxv = float(np.max(selected))
        expd = np.exp(selected - maxv)
        denom = float(expd.sum())
        if not np.isfinite(denom) or denom <= 0.0:
            probs[valid] = 1.0 / float(valid.sum())
        else:
            probs[valid] = (expd / max(denom, 1e-9)).astype(np.float32, copy=False)
        return probs

    # ------------------------------------------------------------------ background thread
    def _coalesce_loop(self) -> None:
        while not self._shutdown.is_set():
            batch: list[_EvalRequest] = []
            with self._cv:
                while not self._queue and not self._shutdown.is_set():
                    self._cv.wait(timeout=0.01)
                if self._shutdown.is_set():
                    break
                if self._queue:
                    batch.append(self._queue.pop(0))
                total = batch[0].size if batch else 0
                deadline = time.monotonic() + max(0.0, float(self._coalesce_ms) / 1000.0)
                while total < self._batch_cap and time.monotonic() < deadline:
                    if not self._queue:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._cv.wait(timeout=min(remaining, 0.002))
                        continue
                    if self._queue[0].size <= (self._batch_cap - total):
                        request = self._queue.pop(0)
                        batch.append(request)
                        total += request.size
                    else:
                        break
            if not batch:
                continue
            try:
                flat_positions: list[Any] = []
                flat_moves: list[list[Any]] = []
                sizes: list[int] = []
                for request in batch:
                    flat_positions.extend(request.positions)
                    flat_moves.extend(request.moves)
                    sizes.append(request.size)
                probs_list, values_arr = self._infer_positions_legal_locked(flat_positions, flat_moves)
                offset = 0
                for request, size in zip(batch, sizes, strict=False):
                    request.out_probs = probs_list[offset : offset + size]
                    request.out_values = values_arr[offset : offset + size]
                    request.event.set()
                    offset += size
            except Exception:
                for request in batch:
                    request.out_probs = [np.zeros((0,), dtype=np.float32) for _ in range(request.size)]
                    request.out_values = np.zeros((request.size,), dtype=np.float32)
                    request.event.set()


class _EvalRequest:
    __slots__ = ("positions", "moves", "size", "out_probs", "out_values", "event")

    def __init__(self, positions: list[Any], moves: list[list[Any]]) -> None:
        self.positions = positions
        self.moves = moves
        self.size = len(positions)
        self.out_probs: Optional[list[np.ndarray]] = None
        self.out_values: Optional[np.ndarray] = None
        self.event = threading.Event()
