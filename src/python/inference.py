"""Batched inference helpers for evaluating neural network positions."""

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
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model_lock = threading.Lock()
        self._inference_dtype: torch.dtype = select_inference_dtype(self.device)
        self.eval_model: nn.Module = prepare_model(
            ChessNet(),
            self.device,
            channels_last=C.TORCH.EVAL_MODEL_CHANNELS_LAST,
            dtype=self._inference_dtype,
            eval_mode=True,
            freeze=True,
        )
        self._np_inference_dtype: np.dtype[Any]
        self._cache_out_dtype: torch.dtype
        if self._inference_dtype == torch.bfloat16:
            self._np_inference_dtype = np.dtype(np.float32)
            self._cache_out_dtype = torch.float32
        else:
            self._np_inference_dtype = (
                np.dtype(np.float16)
                if self._inference_dtype == torch.float16
                else np.dtype(np.float32)
            )
            self._cache_out_dtype = (
                torch.float16
                if (C.EVAL.CACHE_USE_FP16 and self._inference_dtype == torch.float16)
                else torch.float32
            )
        self.cache_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, float] = {
            "requests_total": 0,
            "cache_hits_total": 0,
            "cache_misses_total": 0,
            "batches_total": 0,
            "eval_positions_total": 0,
            "batch_size_max": 0,
        }
        self._shutdown = threading.Event()
        self._batch_size_cap = int(C.EVAL.BATCH_SIZE_MAX)
        self._coalesce_ms = int(C.EVAL.COALESCE_MS)

        self._val_cache_cap = 200_000
        self._val_cache: OrderedDict[int, float] = OrderedDict()
        self._enc_cache_cap = 64_000
        self._enc_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._out_cache_cap = int(C.EVAL.CACHE_CAPACITY)
        self._out_cache: OrderedDict[int, tuple[np.ndarray, float]] = OrderedDict()

        self._req_cv = threading.Condition()
        self._req_queue: list[_EvalRequest] = []
        self._coalesce_thread = threading.Thread(
            target=self._coalesce_loop, name="EvalCoalesce", daemon=True
        )
        self._coalesce_thread.start()

    def set_batching_params(
        self, batch_size_max: int | None = None, coalesce_ms: int | None = None
    ) -> None:
        if batch_size_max is not None:
            self._batch_size_cap = int(max(1, batch_size_max))
        if coalesce_ms is not None:
            self._coalesce_ms = int(max(0, coalesce_ms))

    def close(self) -> None:
        self._shutdown.set()
        with suppress(Exception), self._req_cv:
            self._req_cv.notify_all()
        with suppress(Exception):
            if (
                getattr(self, "_coalesce_thread", None) is not None
                and self._coalesce_thread.is_alive()
            ):
                self._coalesce_thread.join(timeout=1.0)
        with suppress(Exception):
            self._enc_cache.clear()
        with suppress(Exception):
            self._val_cache.clear()
        with suppress(Exception):
            self._out_cache.clear()

    def set_cache_capacity(self, capacity: int) -> None:
        cap = int(max(0, capacity))
        self._enc_cache_cap = max(1, cap)
        self._val_cache_cap = max(1, cap)
        self._out_cache_cap = max(1, cap)

    @property
    def cache_capacity(self) -> int:
        return int(self._enc_cache_cap)

    @cache_capacity.setter
    def cache_capacity(self, value: int) -> None:
        self.set_cache_capacity(int(value))

    def __enter__(self) -> BatchedEvaluator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self):
        with suppress(Exception):
            self.close()

    def refresh_from(self, src_model: torch.nn.Module) -> None:
        with self.model_lock:
            base_obj = getattr(src_model, "_orig_mod", src_model)
            if hasattr(base_obj, "module"):
                base_obj = base_obj.module
            base = cast(nn.Module, base_obj)
            model = prepare_model(
                ChessNet(),
                self.device,
                channels_last=C.TORCH.EVAL_MODEL_CHANNELS_LAST,
            )
            model.load_state_dict(base.state_dict(), strict=True)
            self.eval_model = model
            with suppress(Exception):
                self._fuse_eval_model()
                model = self.eval_model
            self.eval_model = prepare_model(
                model,
                self.device,
                channels_last=C.TORCH.EVAL_MODEL_CHANNELS_LAST,
                dtype=self._inference_dtype,
                eval_mode=True,
                freeze=True,
            )
        with self.cache_lock:
            self._enc_cache.clear()
            self._val_cache.clear()
            self._out_cache.clear()

    def _encode_batch(self, positions: list[Any]) -> torch.Tensor:
        if not positions:
            x = torch.zeros(
                (0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32
            )
            return x.to(self.device)
        encoded: list[np.ndarray | None] = [None] * len(positions)
        misses: list[tuple[int, Any]] = []
        keys: list[int | None] = [self._position_key(p) for p in positions]
        with self.cache_lock:
            for i, k in enumerate(keys):
                if k is not None and k in self._enc_cache:
                    arr = self._enc_cache.pop(k)
                    self._enc_cache[k] = arr
                    encoded[i] = arr
                else:
                    misses.append((i, positions[i]))
        if misses:
            miss_pos = [p for _, p in misses]
            miss_np = encoder.encode_batch(miss_pos)
            for j, (i, _) in enumerate(misses):
                arr = miss_np[j].astype(self._np_inference_dtype, copy=False)
                encoded[i] = arr

        with self.cache_lock:
            for i, k in enumerate(keys):
                if k is not None:
                    maybe_arr = encoded[i]
                    if maybe_arr is not None:
                        self._enc_cache[k] = maybe_arr
                        while len(self._enc_cache) > self._enc_cache_cap:
                            self._enc_cache.popitem(last=False)
        x_np = np.stack(
            [
                np.asarray(
                    cast(np.ndarray, e), dtype=self._np_inference_dtype, order="C"
                )
                for e in encoded
            ]
        )
        x = torch.from_numpy(x_np)
        if torch.cuda.is_available() and x.device.type == "cpu":
            x = x.pin_memory()
        x = x.to(self.device, non_blocking=True)
        return (
            x.contiguous(memory_format=torch.channels_last)
            if x.dim() == 4
            else x.contiguous()
        )

    @staticmethod
    def _fuse_conv_bn_(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
        if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
            return
        w = conv.weight.data
        bias = conv.bias
        if bias is None:
            conv.bias = nn.Parameter(
                torch.zeros(w.size(0), device=w.device, dtype=w.dtype)
            )
            bias = conv.bias
        b = bias.data
        gamma = bn.weight.data
        beta = bn.bias.data
        running_mean = getattr(bn, "running_mean", None)
        running_var = getattr(bn, "running_var", None)
        if running_mean is None or running_var is None:
            return
        mean = running_mean.data
        var = running_var.data
        eps = bn.eps
        inv_std = torch.rsqrt(var + eps)
        w.mul_(gamma.reshape(-1, 1, 1, 1) * inv_std.reshape(-1, 1, 1, 1))
        b.mul_(1)
        b.copy_(b + (beta - gamma * mean * inv_std))

    def _fuse_eval_model(self) -> None:
        m = self.eval_model
        from network import ChessNet, ResidualBlock

        if not isinstance(m, ChessNet):
            return
        if isinstance(m.bn_in, nn.BatchNorm2d):
            self._fuse_conv_bn_(m.conv_in, m.bn_in)
            m.bn_in = nn.Identity()
        for blk in m.residual_stack:
            if isinstance(blk, ResidualBlock):
                if isinstance(blk.bn1, nn.BatchNorm2d):
                    self._fuse_conv_bn_(blk.conv1, blk.bn1)
                    blk.bn1 = nn.Identity()
                if isinstance(blk.bn2, nn.BatchNorm2d):
                    self._fuse_conv_bn_(blk.conv2, blk.bn2)
                    blk.bn2 = nn.Identity()
        if isinstance(m.policy_bn, nn.BatchNorm2d):
            self._fuse_conv_bn_(m.policy_conv, m.policy_bn)
            m.policy_bn = nn.Identity()
        if isinstance(m.value_bn, nn.BatchNorm2d):
            self._fuse_conv_bn_(m.value_conv, m.value_bn)
            m.value_bn = nn.Identity()

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self.model_lock, torch.inference_mode():
            x = x.to(dtype=self._inference_dtype)
            return cast(tuple[torch.Tensor, torch.Tensor], self.eval_model(x))

    def _infer_positions_legal_direct(
        self, positions: list[Any], moves_per_position: list[list[Any]]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        if not positions:
            return [], np.zeros((0,), dtype=np.float32)
        n = len(positions)
        with self._metrics_lock:
            self._metrics["requests_total"] += n

        idx_lists = [
            np.asarray(encoded, dtype=np.int64).reshape(-1)
            for encoded in encoder.encode_move_indices_batch(moves_per_position)
        ]

        probs_out: list[np.ndarray] = [
            np.zeros((0,), dtype=np.float32) for _ in range(n)
        ]
        values_out: list[float] = [0.0] * n
        hits: list[int] = []
        misses: list[int] = []
        keys: list[int | None] = [self._position_key(p) for p in positions]

        with self.cache_lock:
            for i, k in enumerate(keys):
                if k is not None and k in self._out_cache:
                    pol_logits_np, v = self._out_cache.pop(k)
                    self._out_cache[k] = (pol_logits_np, v)
                    idx_arr = np.asarray(idx_lists[i], dtype=np.int64)
                    valid = (idx_arr >= 0) & (idx_arr < POLICY_OUTPUT)
                    idx_arr = idx_arr[valid]
                    if idx_arr.size > 0:
                        sel = pol_logits_np[idx_arr].astype(np.float32, copy=False)
                        m = float(sel.max()) if sel.size > 0 else 0.0
                        ex = np.exp(sel - m)
                        s = float(ex.sum())
                        probs_out[i] = (ex / (s if s > 0 else 1.0)).astype(
                            np.float32, copy=False
                        )
                    values_out[i] = float(v)
                    hits.append(i)
                else:
                    misses.append(i)

        if misses:
            miss_positions = [positions[i] for i in misses]
            x = self._encode_batch(miss_positions)
            policy_out_t, value_out_t = self._forward(x)
            policy_logits = policy_out_t.float()
            policy_logits = policy_logits - policy_logits.amax(dim=1, keepdim=True)

            lengths_miss = [
                int(np.asarray(idx_lists[i], dtype=np.int64).shape[0]) for i in misses
            ]
            Lmax = max(1, max(lengths_miss))
            idx_padded = torch.full(
                (len(misses), Lmax), -1, dtype=torch.long, device=self.device
            )
            for j, i in enumerate(misses):
                arr = np.asarray(idx_lists[i], dtype=np.int64)
                if arr.size == 0:
                    continue
                arr = np.where((arr >= 0) & (arr < POLICY_OUTPUT), arr, -1)
                t = torch.from_numpy(arr).to(
                    self.device, non_blocking=True, dtype=torch.long
                )
                idx_padded[j, : t.numel()] = t
            mask = idx_padded >= 0
            gather_idx = idx_padded.clamp_min(0)
            gathered = policy_logits.gather(1, gather_idx)
            neg_inf = torch.finfo(gathered.dtype).min
            masked = torch.where(mask, gathered, torch.full_like(gathered, neg_inf))
            maxv = masked.max(dim=1, keepdim=True).values
            exp = torch.exp(masked - maxv)
            exp = torch.where(mask, exp, torch.zeros_like(exp))
            denom = exp.sum(dim=1, keepdim=True).clamp_min(1e-9)
            probs = exp / denom

            out_dtype = self._cache_out_dtype
            probs_np = probs.detach().to(dtype=out_dtype).cpu().numpy()
            values_np = value_out_t.detach().to(dtype=out_dtype).cpu().numpy()
            policy_logits_np_batch = (
                policy_logits.detach().to(dtype=out_dtype).cpu().numpy()
            )

            with self.cache_lock:
                for j, i in enumerate(misses):
                    L = int(np.asarray(idx_lists[i], dtype=np.int64).shape[0])
                    probs_out[i] = probs_np[j, :L].astype(np.float32, copy=False)
                    values_out[i] = float(values_np[j])
                    k = keys[i]
                    if k is not None:
                        pol_logits_np = policy_logits_np_batch[j].reshape(-1)
                        self._out_cache[k] = (pol_logits_np, float(values_np[j]))
                        while len(self._out_cache) > self._out_cache_cap:
                            self._out_cache.popitem(last=False)

            with self._metrics_lock:
                self._metrics["batches_total"] += 1
                self._metrics["eval_positions_total"] += len(misses)
                self._metrics["cache_misses_total"] += len(misses)
                if len(misses) > int(self._metrics["batch_size_max"]):
                    self._metrics["batch_size_max"] = float(len(misses))

        if hits:
            with self._metrics_lock:
                self._metrics["cache_hits_total"] += len(hits)

        pol_list: list[np.ndarray] = list(probs_out)
        values_arr = np.asarray(values_out, dtype=np.float32)
        return pol_list, values_arr

    def infer_positions_legal(
        self, positions: list[Any], moves_per_position: list[list[Any]]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        if self._batch_size_cap <= 1 or self._coalesce_ms <= 0:
            return self._infer_positions_legal_direct(positions, moves_per_position)
        req = _EvalRequest(positions, moves_per_position)
        with self._req_cv:
            self._req_queue.append(req)
            self._req_cv.notify()
        req.ev.wait()
        return cast(tuple[list[np.ndarray], np.ndarray], (req.out_pol, req.out_val))

    @property
    def batch_size_cap(self) -> int:
        return int(self._batch_size_cap)

    def _coalesce_loop(self) -> None:
        while not self._shutdown.is_set():
            batch: list[_EvalRequest] = []
            with self._req_cv:
                while (not self._req_queue) and (not self._shutdown.is_set()):
                    self._req_cv.wait(timeout=0.01)
                if self._shutdown.is_set():
                    break
                if self._req_queue:
                    batch.append(self._req_queue.pop(0))
                else:
                    continue
                total = batch[0].size
                deadline = time.monotonic() + max(
                    0.0, float(self._coalesce_ms) / 1000.0
                )
                while (total < self._batch_size_cap) and (time.monotonic() < deadline):
                    if self._req_queue:
                        if self._req_queue[0].size <= (self._batch_size_cap - total):
                            r = self._req_queue.pop(0)
                            batch.append(r)
                            total += r.size
                            continue
                        else:
                            break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._req_cv.wait(timeout=min(remaining, 0.002))
                    if not self._req_queue:
                        break
            try:
                flat_pos: list[Any] = []
                flat_moves: list[list[Any]] = []
                sizes: list[int] = []
                for r in batch:
                    flat_pos.extend(r.positions)
                    flat_moves.extend(r.moves)
                    sizes.append(r.size)
                pol_list, val_arr = self._infer_positions_legal_direct(
                    flat_pos, flat_moves
                )
                off = 0
                for r, sz in zip(batch, sizes, strict=False):
                    r.out_pol = pol_list[off : off + sz]
                    r.out_val = val_arr[off : off + sz]
                    r.ev.set()
                    off += sz
            except Exception:
                for r in batch:
                    r.out_pol = [
                        np.zeros((0,), dtype=np.float32) for _ in range(r.size)
                    ]
                    r.out_val = np.zeros((r.size,), dtype=np.float32)
                    r.ev.set()

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

    def get_metrics(self) -> dict[str, float]:
        with self._metrics_lock:
            return dict(self._metrics)

    def infer_values(self, positions: list[Any]) -> np.ndarray:
        if not positions:
            return np.zeros((0,), dtype=np.float32)
        pending: list[tuple[int, Any]] = []
        values: list[float | None] = [None] * len(positions)
        with self.cache_lock:
            for i, pos in enumerate(positions):
                key = self._position_key(pos)
                if key is not None and key in self._val_cache:
                    val = self._val_cache.pop(key)
                    self._val_cache[key] = val
                    values[i] = float(val)
                else:
                    pending.append((i, pos))
        if pending:
            x = self._encode_batch([p for _, p in pending])
            _, val_t = self._forward(x)
            val_np = val_t.detach().to(dtype=torch.float32).cpu().numpy()
            with self.cache_lock:
                for (i, pos), v in zip(pending, val_np, strict=False):
                    values[i] = float(v)
                    key = self._position_key(pos)
                    if key is not None:
                        self._val_cache[key] = float(v)
                        while len(self._val_cache) > self._val_cache_cap:
                            self._val_cache.popitem(last=False)
        return np.asarray(
            [(0.0 if v is None else float(v)) for v in values], dtype=np.float32
        )


class _EvalRequest:
    __slots__ = ("ev", "moves", "out_pol", "out_val", "positions", "size")

    def __init__(self, positions: list[Any], moves: list[list[Any]]) -> None:
        self.positions = positions
        self.moves = moves
        self.size = len(positions)
        self.out_pol: list[np.ndarray] | None = None
        self.out_val: np.ndarray | None = None
        self.ev = threading.Event()
