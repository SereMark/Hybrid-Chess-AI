"""Batched inference with caching and request coalescing."""
from __future__ import annotations

import numbers
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from typing import Any, cast

import numpy as np
import torch
from torch import nn

import config as C
import encoder
from network import BOARD_SIZE, INPUT_PLANES, POLICY_OUTPUT, ChessNet
from utils import prepare_model, select_inference_dtype

__all__ = ["BatchedEvaluator"]


class BatchedEvaluator:
    """Thread-safe evaluator with encode/output caches and soft batching."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model_lock = threading.Lock()

        # Model
        self._dtype: torch.dtype = select_inference_dtype(self.device)
        self.eval_model: nn.Module = prepare_model(
            ChessNet(),
            self.device,
            channels_last=C.TORCH.eval_model_channels_last,
            dtype=self._dtype,
            eval_mode=True,
            freeze=True,
        )

        # Dtypes for host tensors and cache
        if self._dtype == torch.bfloat16:
            self._np_dtype = np.float32  # keep encode precision
            self._cache_dtype = torch.float32
        elif self._dtype == torch.float16:
            self._np_dtype = np.float16
            self._cache_dtype = torch.float16 if C.EVAL.use_fp16_cache else torch.float32
        else:
            self._np_dtype = np.float32
            self._cache_dtype = torch.float32

        # Caches
        self.cache_lock = threading.Lock()
        self._val_cap = int(max(1, C.EVAL.value_cache_capacity))
        self._enc_cap = int(max(1, C.EVAL.encode_cache_capacity))
        self._out_cap = int(max(1, C.EVAL.cache_capacity))
        self._val_cache: OrderedDict[int, float] = OrderedDict()
        self._enc_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._out_cache: OrderedDict[int, tuple[np.ndarray, float]] = OrderedDict()

        # Batching
        self._batch_cap = int(C.EVAL.batch_size_max)
        self._coalesce_ms = int(C.EVAL.coalesce_ms)
        self._shutdown = threading.Event()
        self._cv = threading.Condition()
        self._queue: list[_EvalRequest] = []
        self._thread = threading.Thread(target=self._coalesce_loop, name="EvalCoalesce", daemon=True)
        self._thread.start()

        # Metrics
        self._m_lock = threading.Lock()
        self._m = {
            "requests_total": 0.0,
            "cache_hits_total": 0.0,
            "cache_misses_total": 0.0,
            "batches_total": 0.0,
            "eval_positions_total": 0.0,
            "batch_size_max": 0.0,
        }

    # ---------------------------- lifecycle
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

    def __del__(self):
        with suppress(Exception):
            self.close()

    # ---------------------------- config
    def set_batching_params(self, batch_size_max: int | None = None, coalesce_ms: int | None = None) -> None:
        if batch_size_max is not None:
            self._batch_cap = int(max(1, batch_size_max))
        if coalesce_ms is not None:
            self._coalesce_ms = int(max(0, coalesce_ms))

    def set_cache_capacity(self, capacity: int | None = None, *, value_capacity: int | None = None, encode_capacity: int | None = None) -> None:
        if capacity is not None:
            self._out_cap = int(max(1, capacity))
        if value_capacity is not None:
            self._val_cap = int(max(1, value_capacity))
        if encode_capacity is not None:
            self._enc_cap = int(max(1, encode_capacity))

    @property
    def cache_capacity(self) -> int:
        return int(self._enc_cap)

    @cache_capacity.setter
    def cache_capacity(self, v: int) -> None:
        self.set_cache_capacity(int(v))

    def clear_caches(self) -> None:
        with self.cache_lock:
            self._enc_cache.clear()
            self._val_cache.clear()
            self._out_cache.clear()

    # ---------------------------- model refresh
    def refresh_from(self, src_model: torch.nn.Module) -> None:
        with self.model_lock:
            base = getattr(src_model, "_orig_mod", src_model)
            if hasattr(base, "module"):
                base = base.module
            model = prepare_model(ChessNet(), self.device, channels_last=C.TORCH.eval_model_channels_last)
            model.load_state_dict(cast(nn.Module, base).state_dict(), strict=True)
            self.eval_model = model
            with suppress(Exception):
                self._fuse_eval_model()
                model = self.eval_model
            self.eval_model = prepare_model(
                model,
                self.device,
                channels_last=C.TORCH.eval_model_channels_last,
                dtype=self._dtype,
                eval_mode=True,
                freeze=True,
            )
        self.clear_caches()

    # ---------------------------- public inference
    def infer_positions_legal(self, positions: list[Any], moves_per_position: list[list[Any]]) -> tuple[list[np.ndarray], np.ndarray]:
        if self._batch_cap <= 1 or self._coalesce_ms <= 0:
            return self._infer_positions_legal_direct(positions, moves_per_position)
        req = _EvalRequest(positions, moves_per_position)
        with self._cv:
            self._queue.append(req)
            self._cv.notify()
        req.ev.wait()
        return cast(tuple[list[np.ndarray], np.ndarray], (req.out_pol, req.out_val))

    def infer_values(self, positions: list[Any]) -> np.ndarray:
        if not positions:
            return np.zeros((0,), dtype=np.float32)

        pending: list[tuple[int, Any]] = []
        values: list[float | None] = [None] * len(positions)

        with self.cache_lock:
            for i, p in enumerate(positions):
                k = self._position_key(p)
                if k is not None and k in self._val_cache:
                    v = self._val_cache.pop(k)
                    self._val_cache[k] = v
                    values[i] = float(v)
                else:
                    pending.append((i, p))

        if pending:
            x = self._encode_batch([p for _, p in pending])
            _, v_t = self._forward(x)
            v_np = v_t.detach().to(dtype=torch.float32).cpu().numpy()
            with self.cache_lock:
                for (i, p), v in zip(pending, v_np, strict=False):
                    values[i] = float(v)
                    k = self._position_key(p)
                    if k is not None:
                        self._val_cache[k] = float(v)
                        while len(self._val_cache) > self._val_cap:
                            self._val_cache.popitem(last=False)

        return np.asarray([(0.0 if v is None else float(v)) for v in values], dtype=np.float32)

    def get_metrics(self) -> dict[str, float]:
        with self._m_lock:
            return dict(self._m)

    # ---------------------------- internals
    def _encode_batch(self, positions: list[Any]) -> torch.Tensor:
        if not positions:
            return torch.zeros((0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32, device=self.device)

        encoded: list[np.ndarray | None] = [None] * len(positions)
        misses: list[tuple[int, Any]] = []
        keys = [self._position_key(p) for p in positions]

        with self.cache_lock:
            for i, k in enumerate(keys):
                if k is not None and k in self._enc_cache:
                    arr = self._enc_cache.pop(k)
                    self._enc_cache[k] = arr
                    encoded[i] = arr
                else:
                    misses.append((i, positions[i]))

        if misses:
            miss_np = encoder.encode_batch([p for _, p in misses])
            for j, (i, _) in enumerate(misses):
                encoded[i] = miss_np[j].astype(self._np_dtype, copy=False)

        with self.cache_lock:
            for i, k in enumerate(keys):
                if k is not None and encoded[i] is not None:
                    self._enc_cache[k] = cast(np.ndarray, encoded[i])
                    while len(self._enc_cache) > self._enc_cap:
                        self._enc_cache.popitem(last=False)

        x_np = np.stack([np.asarray(cast(np.ndarray, e), dtype=self._np_dtype, order="C") for e in encoded])
        x = torch.from_numpy(x_np)
        if torch.cuda.is_available() and x.device.type == "cpu":
            x = x.pin_memory()
        x = x.to(self.device, non_blocking=True)
        return x.contiguous(memory_format=torch.channels_last)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self.model_lock, torch.inference_mode():
            x = x.to(dtype=self._dtype)
            return cast(tuple[torch.Tensor, torch.Tensor], self.eval_model(x))

    def _infer_positions_legal_direct(self, positions: list[Any], moves_per_position: list[list[Any]]) -> tuple[list[np.ndarray], np.ndarray]:
        if not positions:
            return [], np.zeros((0,), dtype=np.float32)
        n = len(positions)
        with self._m_lock:
            self._m["requests_total"] += n

        idx_lists = [np.asarray(arr, dtype=np.int64).reshape(-1) for arr in encoder.encode_move_indices_batch(moves_per_position)]
        probs_out: list[np.ndarray] = [np.zeros((0,), dtype=np.float32) for _ in range(n)]
        values_out: list[float] = [0.0] * n

        hits: list[int] = []
        misses: list[int] = []
        keys = [self._position_key(p) for p in positions]

        with self.cache_lock:
            for i, k in enumerate(keys):
                if k is not None and k in self._out_cache:
                    logits_np, v = self._out_cache.pop(k)
                    self._out_cache[k] = (logits_np, v)
                    idx = idx_lists[i]
                    probs = np.zeros(idx.shape[0], dtype=np.float32)
                    valid = (idx >= 0) & (idx < POLICY_OUTPUT)
                    if np.any(valid):
                        sel = logits_np[idx[valid]].astype(np.float32, copy=False)
                        m = float(sel.max()) if sel.size > 0 else 0.0
                        ex = np.exp(sel - m)
                        s = float(ex.sum())
                        probs[valid] = (ex / (s if s > 0 else 1.0)).astype(np.float32, copy=False)
                    probs_out[i] = probs
                    values_out[i] = float(v)
                    hits.append(i)
                else:
                    misses.append(i)

        if misses:
            x = self._encode_batch([positions[i] for i in misses])
            pol_t, val_t = self._forward(x)
            logits = pol_t.float()
            logits = logits - logits.amax(dim=1, keepdim=True)

            Lmax = max(1, max(int(np.asarray(idx_lists[i], dtype=np.int64).size) for i in misses))
            idx_padded = torch.full((len(misses), Lmax), -1, dtype=torch.long, device=self.device)
            for j, i in enumerate(misses):
                arr = np.asarray(idx_lists[i], dtype=np.int64)
                if arr.size > 0:
                    t = torch.from_numpy(np.where((arr >= 0) & (arr < POLICY_OUTPUT), arr, -1)).to(self.device, non_blocking=True)
                    idx_padded[j, : t.numel()] = t

            mask = idx_padded >= 0
            gather_idx = idx_padded.clamp_min(0)
            gathered = logits.gather(1, gather_idx)
            neg_inf = torch.finfo(gathered.dtype).min
            masked = torch.where(mask, gathered, torch.full_like(gathered, neg_inf))
            maxv = masked.max(dim=1, keepdim=True).values
            ex = torch.exp(masked - maxv)
            ex = torch.where(mask, ex, torch.zeros_like(ex))
            denom = ex.sum(dim=1, keepdim=True).clamp_min(1e-9)
            probs = ex / denom

            probs_np = probs.detach().to(dtype=self._cache_dtype).cpu().numpy()
            vals_np = val_t.detach().to(dtype=self._cache_dtype).cpu().numpy()
            logits_np = logits.detach().to(dtype=self._cache_dtype).cpu().numpy()

            with self.cache_lock:
                for j, i in enumerate(misses):
                    L = int(np.asarray(idx_lists[i], dtype=np.int64).size)
                    probs_out[i] = probs_np[j, :L].astype(np.float32, copy=False)
                    values_out[i] = float(vals_np[j])
                    k = keys[i]
                    if k is not None:
                        self._out_cache[k] = (logits_np[j].reshape(-1), float(vals_np[j]))
                        while len(self._out_cache) > self._out_cap:
                            self._out_cache.popitem(last=False)

            with self._m_lock:
                self._m["batches_total"] += 1
                self._m["eval_positions_total"] += len(misses)
                self._m["cache_misses_total"] += len(misses)
                if len(misses) > self._m["batch_size_max"]:
                    self._m["batch_size_max"] = float(len(misses))

        if hits:
            with self._m_lock:
                self._m["cache_hits_total"] += len(hits)

        return list(probs_out), np.asarray(values_out, dtype=np.float32)

    def _position_key(self, pos: Any) -> int | None:
        try:
            h = getattr(pos, "hash", None)
            v = h() if callable(h) else h
            if v is None:
                return None
            if isinstance(v, np.generic):
                v = cast(Any, v).item()
            if isinstance(v, numbers.Integral):
                return int(v)
            if isinstance(v, str):
                try:
                    return int(v)
                except Exception:
                    return None
            return None
        except Exception:
            return None

    # ---------------------------- fusion
    @staticmethod
    def _fuse_conv_bn_(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
        if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
            return
        w = conv.weight.data
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(w.size(0), device=w.device, dtype=w.dtype))
        b = conv.bias.data
        gamma, beta = bn.weight.data, bn.bias.data
        rm = getattr(bn, "running_mean", None)
        rv = getattr(bn, "running_var", None)
        if rm is None or rv is None:
            return
        inv_std = torch.rsqrt(rv.data + bn.eps)
        w.mul_(gamma.reshape(-1, 1, 1, 1) * inv_std.reshape(-1, 1, 1, 1))
        b.copy_(b + (beta - gamma * rm.data * inv_std))

    def _fuse_eval_model(self) -> None:
        from network import ChessNet, ResidualBlock

        m = self.eval_model
        if not isinstance(m, ChessNet):
            return
        if isinstance(m.bn_in, nn.BatchNorm2d):
            self._fuse_conv_bn_(m.conv_in, m.bn_in)
        for blk in m.residual_stack:
            if isinstance(blk, ResidualBlock):
                if isinstance(blk.bn1, nn.BatchNorm2d):
                    self._fuse_conv_bn_(blk.conv1, blk.bn1)
                if isinstance(blk.bn2, nn.BatchNorm2d):
                    self._fuse_conv_bn_(blk.conv2, blk.bn2)
        if isinstance(m.policy_bn, nn.BatchNorm2d):
            self._fuse_conv_bn_(m.policy_conv, m.policy_bn)
        if isinstance(m.value_bn, nn.BatchNorm2d):
            self._fuse_conv_bn_(m.value_conv, m.value_bn)

    # ---------------------------- coalescing thread
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
                        r = self._queue.pop(0)
                        batch.append(r)
                        total += r.size
                    else:
                        break
            if not batch:
                continue
            try:
                flat_pos: list[Any] = []
                flat_moves: list[list[Any]] = []
                sizes: list[int] = []
                for r in batch:
                    flat_pos.extend(r.positions)
                    flat_moves.extend(r.moves)
                    sizes.append(r.size)
                pol_list, val_arr = self._infer_positions_legal_direct(flat_pos, flat_moves)
                off = 0
                for r, sz in zip(batch, sizes, strict=False):
                    r.out_pol = pol_list[off : off + sz]
                    r.out_val = val_arr[off : off + sz]
                    r.ev.set()
                    off += sz
            except Exception:
                for r in batch:
                    r.out_pol = [np.zeros((0,), dtype=np.float32) for _ in range(r.size)]
                    r.out_val = np.zeros((r.size,), dtype=np.float32)
                    r.ev.set()


class _EvalRequest:
    __slots__ = ("ev", "moves", "out_pol", "out_val", "positions", "size")

    def __init__(self, positions: list[Any], moves: list[list[Any]]) -> None:
        self.positions = positions
        self.moves = moves
        self.size = len(positions)
        self.out_pol: list[np.ndarray] | None = None
        self.out_val: np.ndarray | None = None
        self.ev = threading.Event()