from __future__ import annotations

import numbers
import threading
from collections import OrderedDict
from contextlib import suppress
from typing import Any, cast

import chesscore as ccore
import numpy as np
import torch
from torch import nn

import config as C
from network import BOARD_SIZE, INPUT_PLANES, POLICY_OUTPUT, ChessNet


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

    def set_batching_params(self, batch_size_max: int | None = None, coalesce_ms: int | None = None) -> None:
        if batch_size_max is not None:
            self._batch_size_cap = int(max(1, batch_size_max))
        if coalesce_ms is not None:
            self._coalesce_ms = int(max(0, coalesce_ms))

    def close(self) -> None:
        self._shutdown.set()
        with suppress(Exception):
            self._enc_cache.clear()
        with suppress(Exception):
            self._val_cache.clear()
    def set_cache_capacity(self, capacity: int) -> None:
        cap = int(max(0, capacity))
        self._enc_cache_cap = max(1, cap)
        self._val_cache_cap = max(1, cap)

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
            with suppress(Exception):
                self._fuse_eval_model()
            self.eval_model = cast(nn.Module, self.eval_model.to(dtype=torch.float16))
            self.eval_model.eval()
            for p in self.eval_model.parameters():
                p.requires_grad_(False)
        with self.cache_lock:
            self._enc_cache.clear()
        with suppress(Exception):
            self._val_cache.clear()

    def _encode_batch(self, positions: list[Any]) -> torch.Tensor:
        if not positions:
            x = torch.zeros((0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
            return x.to(self.device)
        encoded: list[np.ndarray] = [None] * len(positions)
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
            miss_np = ccore.encode_batch(miss_pos)
            for j, (i, _) in enumerate(misses):
                arr = miss_np[j].astype(np.float16, copy=False)
                encoded[i] = arr

        with self.cache_lock:
            for i, k in enumerate(keys):
                if k is not None:
                    arr = encoded[i]
                    if arr is not None:
                        self._enc_cache[k] = arr
                        while len(self._enc_cache) > self._enc_cache_cap:
                            self._enc_cache.popitem(last=False)
        x_np = np.stack([np.asarray(e, dtype=np.float16, order="C") for e in encoded])
        x = torch.from_numpy(x_np)
        if C.TORCH.EVAL_PIN_MEMORY:
            x = x.pin_memory()
        x = x.to(self.device, non_blocking=True)
        return x.contiguous(memory_format=torch.channels_last) if x.dim() == 4 else x.contiguous()

    @staticmethod
    def _fuse_conv_bn_(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
        if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
            return
        w = conv.weight.data
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(w.size(0), device=w.device, dtype=w.dtype))
        b = conv.bias.data
        gamma = bn.weight.data
        beta = bn.bias.data
        mean = bn.running_mean.data
        var = bn.running_var.data
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
            x = x.to(dtype=torch.float16)
            return cast(tuple[torch.Tensor, torch.Tensor], self.eval_model(x))
    def infer_positions_legal(
        self, positions: list[Any], moves_per_position: list[list[Any]]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        if not positions:
            return [], np.zeros((0,), dtype=np.float32)
        with self._metrics_lock:
            self._metrics["requests_total"] += len(positions)
        x = self._encode_batch(positions)
        policy_out_t, value_out_t = self._forward(x)
        policy_logits = policy_out_t.float()
        policy_logits = policy_logits - policy_logits.amax(dim=1, keepdim=True)
        try:
            idx_lists_py = ccore.encode_move_indices_batch(moves_per_position)
            idx_lists: list[np.ndarray] = [
                np.asarray(idx_lists_py[i], dtype=np.int64).reshape(-1) for i in range(len(positions))
            ]
        except Exception:
            idx_lists = [
                np.asarray([int(ccore.encode_move_index(m)) for m in moves], dtype=np.int64).reshape(-1)
                for moves in moves_per_position
            ]
        lengths = [int(arr.shape[0]) for arr in idx_lists]
        Lmax = max(1, max(lengths))
        idx_padded = torch.full((len(positions), Lmax), -1, dtype=torch.long, device=self.device)
        for i, arr in enumerate(idx_lists):
            if arr.size == 0:
                continue
            arr_clamped = np.where((arr >= 0) & (arr < POLICY_OUTPUT), arr, -1)
            t = torch.from_numpy(arr_clamped)
            if C.TORCH.EVAL_PIN_MEMORY:
                with suppress(Exception):
                    t = t.pin_memory()
            t = t.to(self.device, non_blocking=True, dtype=torch.long)
            idx_padded[i, : t.numel()] = t
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
        out_dtype = torch.float16 if C.EVAL.CACHE_USE_FP16 else torch.float32
        probs_np = probs.detach().to(dtype=out_dtype).cpu().numpy()
        values_np = value_out_t.detach().to(dtype=out_dtype).cpu().numpy()
        pol_list: list[np.ndarray] = []
        for i, L in enumerate(lengths):
            pol_list.append(probs_np[i, :L].astype(np.float32, copy=False))
        with self._metrics_lock:
            self._metrics["batches_total"] += 1
            self._metrics["eval_positions_total"] += len(positions)
            if len(positions) > int(self._metrics["batch_size_max"]):
                self._metrics["batch_size_max"] = float(len(positions))
        return pol_list, values_np.astype(np.float32, copy=False)

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
        raise NotImplementedError("infer_positions dense path is removed; use infer_positions_legal")

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
        return np.asarray([(0.0 if v is None else float(v)) for v in values], dtype=np.float32)
