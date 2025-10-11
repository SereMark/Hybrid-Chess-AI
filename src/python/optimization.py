"""Optimization primitives for Hybrid Chess AI training."""

from __future__ import annotations

from typing import Any, cast

import config as C
import torch

__all__ = ["build_optimizer", "WarmupCosine", "EMA"]


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    nodecay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "bn" in name.lower() or "batchnorm" in name.lower():
            nodecay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": decay, "weight_decay": C.TRAIN.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=C.TRAIN.learning_rate_init,
        momentum=C.TRAIN.momentum,
        nesterov=True,
        foreach=True,
    )
    return optimizer


class WarmupCosine:
    """Cosine learning-rate scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: Any,
        base_lr: float,
        warmup_steps: int,
        final_lr: float,
        total_steps: int,
        restart_interval: int | None = None,
        restart_decay: float = 1.0,
    ) -> None:
        self.opt = optimizer
        self.base = float(base_lr)
        self.warm = max(1, int(warmup_steps))
        self.final = float(final_lr)
        self.total = max(1, int(total_steps))
        self.restart_steps = max(0, int(restart_interval or 0))
        self.restart_decay = max(0.0, float(restart_decay))
        self.t = 0

    def _lr_at(self, step: int) -> float:
        import math as _m

        if step <= self.warm:
            return self.base * (step / max(1, self.warm))

        if self.restart_steps <= 0:
            progress = min(1.0, (step - self.warm) / max(1, self.total - self.warm))
            return self.final + (self.base - self.final) * 0.5 * (1.0 + _m.cos(_m.pi * progress))

        cycle_len = max(1, self.restart_steps)
        cycle_step = max(0, step - self.warm)
        cycle_index = cycle_step // cycle_len
        within_cycle = cycle_step % cycle_len
        cycle_progress = within_cycle / cycle_len
        decay_scale = self.restart_decay**cycle_index if self.restart_decay > 0.0 else 0.0
        peak = self.base * decay_scale
        trough = self.final * decay_scale
        return trough + (peak - trough) * 0.5 * (1.0 + _m.cos(_m.pi * cycle_progress))

    def step(self) -> None:
        self.t += 1
        lr = self._lr_at(self.t)
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def peek_next_lr(self) -> float:
        t_next = self.t + 1
        return self._lr_at(t_next)

    def set_total_steps(self, total_steps: int) -> None:
        self.total = max(self.t + 1, int(total_steps))
        if self.warm >= self.total:
            self.warm = max(1, self.total - 1)


class EMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float | None = None) -> None:
        self.decay = float(C.TRAIN.ema_decay if decay is None else decay)
        base = _unwrap_module(model)
        self.shadow = {k: v.detach().clone() for k, v in base.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        base = _unwrap_module(model)
        for k, v in base.state_dict().items():
            if not torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
                continue
            if self.shadow[k].dtype != v.dtype:
                self.shadow[k] = self.shadow[k].to(dtype=v.dtype)
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        base = _unwrap_module(model)
        base.load_state_dict(self.shadow, strict=True)


def _unwrap_module(model: torch.nn.Module) -> torch.nn.Module:
    base = getattr(model, "_orig_mod", model)
    if hasattr(base, "module"):
        base = base.module
    return cast(torch.nn.Module, base)
