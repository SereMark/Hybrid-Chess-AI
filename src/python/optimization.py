from __future__ import annotations

from typing import Any, cast

import torch
import config as C

def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """SGD with per-parameter weight decay (exclude biases and norms)."""
    decay: list[torch.nn.Parameter] = []
    nodecay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "bn" in name.lower() or "batchnorm" in name.lower():
            nodecay.append(p)
        else:
            decay.append(p)
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": C.TRAIN.weight_decay}, {"params": nodecay, "weight_decay": 0.0}],
        lr=C.TRAIN.learning_rate_init,
        momentum=C.TRAIN.momentum,
        nesterov=True,
        foreach=True,
    )


class WarmupCosine:
    """Cosine LR schedule with linear warmup and optional restarts."""

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
        import math as m

        if step <= self.warm:
            return self.base * (step / max(1, self.warm))

        if self.restart_steps <= 0:
            prog = min(1.0, (step - self.warm) / max(1, self.total - self.warm))
            return self.final + (self.base - self.final) * 0.5 * (1.0 + m.cos(m.pi * prog))

        cycle = max(1, self.restart_steps)
        idx = max(0, step - self.warm)
        cyc_i, within = divmod(idx, cycle)
        prog = within / cycle
        decay = self.restart_decay**cyc_i if self.restart_decay > 0.0 else 0.0
        peak = self.base * decay
        trough = self.final * decay
        return trough + (peak - trough) * 0.5 * (1.0 + m.cos(m.pi * prog))

    def step(self) -> None:
        self.t += 1
        lr = self._lr_at(self.t)
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def peek_next_lr(self) -> float:
        return self._lr_at(self.t + 1)

    def set_total_steps(self, total_steps: int) -> None:
        self.total = max(self.t + 1, int(total_steps))
        if self.warm >= self.total:
            self.warm = max(1, self.total - 1)


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float | None = None) -> None:
        self.decay = float(C.TRAIN.ema_decay if decay is None else decay)
        base = _unwrap(model)
        self.shadow = {k: v.detach().clone() for k, v in base.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        base = _unwrap(model)
        for k, v in base.state_dict().items():
            if not torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
                continue
            if self.shadow[k].dtype != v.dtype:
                self.shadow[k] = self.shadow[k].to(dtype=v.dtype)
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        _unwrap(model).load_state_dict(self.shadow, strict=True)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    base = getattr(model, "_orig_mod", model)
    if hasattr(base, "module"):
        base = base.module
    return cast(torch.nn.Module, base)