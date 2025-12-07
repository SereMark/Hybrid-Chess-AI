from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from typing import Any, Iterable, Optional

import config as C
import encoder
import numpy as np
import torch
from network import ChessNet
from torch import nn
from utils import prepare_model, select_inference_dtype

__all__ = ["BatchedEvaluator"]


class BatchedEvaluator:
    def __init__(self, device: torch.device) -> None:
        self.log = logging.getLogger("hybridchess.inference")
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

        self._np_dtype: np.dtype[Any]
        self._cache_dtype: torch.dtype
        self._host_infer_dtype: torch.dtype

        if self._dtype == torch.bfloat16:
            self._np_dtype = np.dtype(np.float32)
            self._cache_dtype = torch.float32
            self._host_infer_dtype = torch.float32
            self._should_autocast = self.device.type == "cuda"
        elif self._dtype == torch.float16:
            self._np_dtype = np.dtype(np.float16)
            self._cache_dtype = torch.float16 if C.EVAL.use_fp16_cache else torch.float32
            self._host_infer_dtype = torch.float32 if self.device.type == "cpu" else torch.float16
            self._should_autocast = self.device.type == "cuda"
        else:
            self._np_dtype = np.dtype(np.float32)
            self._cache_dtype = torch.float32
            self._host_infer_dtype = torch.float32
            self._should_autocast = False

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
        self._thread: threading.Thread | None = threading.Thread(
            target=self._coalesce_loop, name="EvalCoalesce", daemon=True
        )
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

    def close(self) -> None:
        self._shutdown.set()
        with suppress(Exception), self._cv:
            self._cv.notify_all()
        with suppress(Exception):
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        self._thread = None
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
        with self.cache_lock:
            if capacity is not None:
                self._out_cap = int(max(1, capacity))
                self._evict_until_within(self._out_cache, self._out_cap)
            if value_capacity is not None:
                self._val_cap = int(max(1, value_capacity))
                self._evict_until_within(self._val_cache, self._val_cap)
            if encode_capacity is not None:
                self._enc_cap = int(max(1, encode_capacity))
                self._evict_until_within(self._enc_cache, self._enc_cap)

    def _touch_lru(self, cache: OrderedDict[int, Any], key: int) -> None:
        if key in cache:
            cache.move_to_end(key)

    @staticmethod
    def _evict_until_within(cache: OrderedDict[int, Any], capacity: int) -> None:
        while cache and len(cache) > capacity:
            cache.popitem(last=False)

    def clear_caches(self) -> None:
        with self.cache_lock:
            self._enc_cache.clear()
            self._val_cache.clear()
            self._out_cache.clear()

    def refresh_from(self, model: nn.Module) -> None:
        with self.model_lock:
            self.eval_model.load_state_dict(model.state_dict(), strict=True)
            self.clear_caches()

    def infer_positions_legal(
        self,
        positions: Iterable[Any],
        moves_per_position: Iterable[Iterable[Any]] | np.ndarray,
        counts: Iterable[int] | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        positions_list = list(positions)

        if counts is not None:
            if not isinstance(moves_per_position, np.ndarray):
                raise TypeError(
                    "Ha a counts argumentum meg van adva, a moves_per_position értékének numpy tömbnek kell lennie"
                )

            flat_moves = moves_per_position
            moves_list: list[Any] = []
            offset = 0
            for c in counts:
                moves_list.append(flat_moves[offset : offset + c])
                offset += c
        else:
            moves_list = [list(moves) for moves in moves_per_position]

        if len(positions_list) != len(moves_list):
            raise ValueError(
                "a positions ({}) és a moves ({}) listáknak azonos hosszúságúnak kell lenniük".format(
                    len(positions_list), len(moves_list)
                )
            )

        probs, values = self._dispatch_eval(positions_list, moves_list)
        return probs, values

    def infer_values(self, positions: Iterable[Any]) -> np.ndarray:
        positions_list = list(positions)
        if not positions_list:
            return np.zeros((0,), dtype=np.float32)

        keys = [self._position_key(pos) for pos in positions_list]
        out = np.zeros((len(positions_list),), dtype=np.float32)
        misses: list[int] = []

        with self.cache_lock:
            for idx, key in enumerate(keys):
                if key is not None and key in self._val_cache:
                    out[idx] = float(self._val_cache[key])
                    self._touch_lru(self._val_cache, key)
                else:
                    misses.append(idx)

        if misses:
            miss_positions = [positions_list[idx] for idx in misses]
            updated = self._infer_values_locked(miss_positions)
            out[np.asarray(misses, dtype=np.intp)] = updated

        with self._m_lock:
            self._metrics["requests_total"] += len(positions_list)
            self._metrics["cache_hits_total"] += len(positions_list) - len(misses)
        return out

    def metrics_snapshot(self) -> dict[str, float]:
        with self._m_lock:
            return {k: float(v) for k, v in self._metrics.items()}

    def get_metrics(self) -> dict[str, float]:
        return self.metrics_snapshot()

    def _position_key(self, position: Any) -> int | None:
        h = getattr(position, "hash", None)
        return int(h) if h is not None else None

    def _dispatch_eval(self, positions_list: list[Any], moves_list: list[Any]) -> tuple[list[np.ndarray], np.ndarray]:
        if len(positions_list) != len(moves_list):
            raise ValueError("a positions és a moves listáknak azonos hosszúságúnak kell lenniük")
        if not positions_list:
            return [], np.zeros((0,), dtype=np.float32)
        if self._shutdown.is_set():
            raise RuntimeError("Az értékelő le lett állítva")

        if self._batch_cap <= 1 and self._coalesce_ms <= 0:
            with self.model_lock:
                probs, values = self._infer_positions_legal_locked(positions_list, moves_list)
            probs = [np.asarray(arr, dtype=np.float32, copy=False) for arr in probs]
            values = np.asarray(values, dtype=np.float32, copy=False)
            return probs, values

        request = _EvalRequest(positions_list, moves_list)
        with self._cv:
            if self._shutdown.is_set():
                raise RuntimeError("Az értékelő le lett állítva")
            self._queue.append(request)
            self._cv.notify()

        timeout = max(0.01, (float(self._coalesce_ms) / 1000.0) + 0.01)
        if not request.event.wait(timeout):
            removed = False
            with self._cv:
                for idx, pending in enumerate(self._queue):
                    if pending is request:
                        self._queue.pop(idx)
                        removed = True
                        break
            if removed:
                with self.model_lock:
                    probs_raw, values_raw = self._infer_positions_legal_locked(positions_list, moves_list)
                request.out_probs = probs_raw
                request.out_values = values_raw
                request.event.set()
            else:
                request.event.wait()

        if request.out_probs is None or request.out_values is None:
            raise RuntimeError("Az értékelő kérése sikertelen volt")
        probs_np = [np.asarray(arr, dtype=np.float32, copy=False) for arr in request.out_probs]
        values_np = np.asarray(request.out_values, dtype=np.float32, copy=False)
        return probs_np, values_np

    def _infer_values_locked(self, positions: list[Any]) -> np.ndarray:
        keys = [self._position_key(pos) for pos in positions]
        dummy_moves: list[list[Any]] = [[] for _ in positions]
        _, values = self._infer_and_cache(positions, dummy_moves, keys)
        return values

    def _infer_positions_legal_locked(
        self, positions: list[Any], moves_per_position: list[Any]
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
                    turn = int(getattr(positions[idx], "turn", 0))
                    probs_out[idx] = self._probs_from_cached_logits(logits_cached, moves, turn)
                    values_out[idx] = float(value_cached)
                    self._touch_lru(self._out_cache, key)
                    if key in self._val_cache:
                        self._touch_lru(self._val_cache, key)
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

    def _infer_and_cache(
        self,
        positions: list[Any],
        moves_per_position: list[Any],
        keys: list[Optional[int]],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        enc = encoder.encode_batch(positions)
        batch_inputs = torch.from_numpy(enc).to(
            self.device,
            dtype=self._host_infer_dtype,
            non_blocking=True,
        )
        if C.TORCH.eval_model_channels_last and self.device.type == "cuda":
            batch_inputs = batch_inputs.contiguous(memory_format=torch.channels_last)
        autocast_enabled = self._should_autocast
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._dtype,
                enabled=autocast_enabled,
            ):
                logits, values = self.eval_model(batch_inputs)

        logits = logits.to(dtype=torch.float32 if autocast_enabled else logits.dtype)
        values_np = values.to(dtype=self._cache_dtype).cpu().numpy()
        logits_np = logits.to(dtype=self._cache_dtype).cpu().numpy()

        is_pre_encoded = False
        if moves_per_position:
            first_list = moves_per_position[0]
            if len(first_list) > 0:
                first_item = first_list[0]
                if isinstance(first_item, (int, np.integer)):
                    is_pre_encoded = True
            elif isinstance(first_list, np.ndarray):
                is_pre_encoded = True

        if is_pre_encoded:
            move_indices = [
                m if isinstance(m, np.ndarray) else np.asarray(m, dtype=np.int64) for m in moves_per_position
            ]
        else:
            turns = [int(getattr(p, "turn", 0)) for p in positions]
            encoded_moves = encoder.encode_canonical_move_indices_batch(moves_per_position, turns)
            move_indices = [np.asarray(arr, dtype=np.int64) for arr in encoded_moves]

        max_moves = max((indices.size for indices in move_indices), default=0)

        if max_moves > 0:
            indices = np.zeros((len(positions), max_moves), dtype=np.int64)
            mask = np.zeros((len(positions), max_moves), dtype=bool)
            for row, idx_arr in enumerate(move_indices):
                valid_indices = idx_arr[idx_arr >= 0]
                length = valid_indices.size
                if length > 0:
                    indices[row, :length] = valid_indices
                    mask[row, :length] = True

            indices_t = torch.from_numpy(indices).to(self.device, dtype=torch.long)
            mask_t = torch.from_numpy(mask).to(self.device, dtype=torch.bool)

            num_logits = logits.shape[1]
            valid_t = (indices_t >= 0) & (indices_t < num_logits) & mask_t
            safe_indices_t = torch.where(valid_t, indices_t, torch.zeros_like(indices_t))

            masked = logits.gather(1, safe_indices_t)
            neg_inf = torch.tensor(float("-inf"), dtype=masked.dtype, device=masked.device)
            masked = torch.where(valid_t, masked, neg_inf)
            maxv = masked.max(dim=1, keepdim=True).values
            ex = torch.exp(masked - maxv)
            ex = torch.where(valid_t, ex, torch.zeros_like(ex))
            denom = ex.sum(dim=1, keepdim=True)
            denom_safe = denom.clamp_min(1e-9)
            probs = torch.where(denom > 0, ex / denom_safe, torch.zeros_like(ex))
            probs_np = probs.to(dtype=self._cache_dtype).cpu().numpy()

            result_probs = []
            for row, idx_arr in enumerate(move_indices):
                valid_count = (idx_arr >= 0).sum()
                row_probs = probs_np[row, :valid_count]

                if idx_arr.size != valid_count:
                    full_probs = np.zeros(idx_arr.shape, dtype=probs_np.dtype)
                    full_probs[idx_arr >= 0] = row_probs
                    result_probs.append(full_probs)
                else:
                    result_probs.append(row_probs)
        else:
            result_probs = [np.zeros((0,), dtype=np.float32) for _ in move_indices]

        with self.cache_lock:
            for row, key in enumerate(keys):
                if key is None:
                    continue
                self._out_cache[key] = (logits_np[row].reshape(-1), float(values_np[row]))
                self._val_cache[key] = float(values_np[row])
                self._touch_lru(self._out_cache, key)
                self._touch_lru(self._val_cache, key)
                self._evict_until_within(self._out_cache, self._out_cap)
                self._evict_until_within(self._val_cache, self._val_cap)

        with self._m_lock:
            self._metrics["cache_misses_total"] += len(positions)
            self._metrics["batches_total"] += 1
            self._metrics["eval_positions_total"] += len(positions)
            if len(positions) > self._metrics["batch_size_max"]:
                self._metrics["batch_size_max"] = float(len(positions))

        return result_probs, values_np.astype(np.float32, copy=False)

    @staticmethod
    def _probs_from_cached_logits(logits: np.ndarray, moves: list[Any], turn: int = 0) -> np.ndarray:
        if len(moves) == 0:
            return np.zeros((0,), dtype=np.float32)

        if isinstance(moves, (np.ndarray, list)) and len(moves) > 0:
            first = moves[0]
            if isinstance(first, (int, np.integer)):
                encoded = np.asarray(moves, dtype=np.int64)
            else:
                encoded_batch = encoder.encode_canonical_move_indices_batch([moves], [turn])
                encoded = encoded_batch[0].astype(np.int64, copy=False)
        else:
            encoded = np.asarray([], dtype=np.int64)

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

    def _coalesce_loop(self) -> None:
        try:
            while True:
                batch: list[_EvalRequest] = []
                with self._cv:
                    while not self._queue and not self._shutdown.is_set():
                        self._cv.wait()
                    if self._shutdown.is_set() and not self._queue:
                        break
                    if self._queue:
                        batch.append(self._queue.pop(0))
                    total = batch[0].size if batch else 0
                    if batch and batch[0].no_wait:
                        deadline = time.monotonic()
                    else:
                        deadline = time.monotonic() + max(0.0, float(self._coalesce_ms) / 1000.0)
                    while total < self._batch_cap and time.monotonic() < deadline:
                        if not self._queue:
                            if self._shutdown.is_set():
                                break
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            self._cv.wait(timeout=remaining)
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
                    with self.model_lock:
                        probs_list, values_arr = self._infer_positions_legal_locked(flat_positions, flat_moves)
                    offset = 0
                    for request, size in zip(batch, sizes, strict=False):
                        request.out_probs = probs_list[offset : offset + size]
                        request.out_values = values_arr[offset : offset + size]
                        request.event.set()
                        offset += size
                except Exception:
                    self.log.exception("Az értékelő kötegelt feldolgozása sikertelen volt")
                    for request in batch:
                        request.out_probs = [np.zeros((0,), dtype=np.float32) for _ in range(request.size)]
                        request.out_values = np.zeros((request.size,), dtype=np.float32)
                        request.event.set()
        finally:
            with self._cv:
                pending = list(self._queue)
                self._queue.clear()
            for request in pending:
                request.out_probs = [np.zeros((0,), dtype=np.float32) for _ in range(request.size)]
                request.out_values = np.zeros((request.size,), dtype=np.float32)
                request.event.set()


class _EvalRequest:
    __slots__ = ("positions", "moves", "size", "out_probs", "out_values", "event", "no_wait")

    def __init__(self, positions: list[Any], moves: list[list[Any]]) -> None:
        self.positions = positions
        self.moves = moves
        self.size = len(positions)
        self.out_probs: Optional[list[np.ndarray]] = None
        self.out_values: Optional[np.ndarray] = None
        self.event = threading.Event()
        self.no_wait = False
