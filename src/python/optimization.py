from __future__ import annotations

from typing import Any, cast

import config as C
import torch

__all__ = ["build_optimizer", "WarmupCosine", "EMA"]


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "bn" in name.lower() or "batchnorm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups = [
        {"params": decay, "weight_decay": C.TRAIN.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    try:
        return torch.optim.SGD(
            param_groups,
            lr=C.TRAIN.learning_rate_init,
            momentum=C.TRAIN.momentum,
            nesterov=True,
            foreach=True,
        )
    except TypeError:
        return torch.optim.SGD(
            param_groups,
            lr=C.TRAIN.learning_rate_init,
            momentum=C.TRAIN.momentum,
            nesterov=True,
        )


class WarmupCosine:
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

    def step(self) -> None:
        self.t += 1
        lr = self._lr_at(self.t)
        for group in self.opt.param_groups:
            group["lr"] = lr

    def set_total_steps(self, total_steps: int) -> None:
        self.total = max(self.t + 1, int(total_steps))
        if self.warm >= self.total:
            self.warm = max(1, self.total - 1)

    def _lr_at(self, step: int) -> float:
        import math as m

        if step <= self.warm:
            return self.base * (step / max(1, self.warm))

        if self.restart_steps <= 0:
            progress = min(1.0, (step - self.warm) / max(1, self.total - self.warm))
            return self.final + (self.base - self.final) * 0.5 * (1.0 + m.cos(m.pi * progress))

        cycle = max(1, self.restart_steps)
        offset = max(0, step - self.warm)
        cycle_index, within = divmod(offset, cycle)
        progress = within / cycle
        if self.restart_decay == 0.0:
            decay = 1.0 if cycle_index == 0 else 0.0
        else:
            decay = self.restart_decay**cycle_index
        peak = self.base * decay
        trough = self.final * decay
        return trough + (peak - trough) * 0.5 * (1.0 + m.cos(m.pi * progress))


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float | None = None) -> None:
        self.decay = float(C.TRAIN.ema_decay if decay is None else decay)
        base = _unwrap(model)
        self.shadow = {name: tensor.detach().clone() for name, tensor in base.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        base = _unwrap(model)
        for name, tensor in base.state_dict().items():
            if not torch.is_floating_point(tensor):
                self.shadow[name] = tensor.detach().clone()
                continue
            if self.shadow[name].dtype != tensor.dtype:
                self.shadow[name] = self.shadow[name].to(dtype=tensor.dtype)
            self.shadow[name].mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        _unwrap(model).load_state_dict(self.shadow, strict=True)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    base = getattr(model, "_orig_mod", model)
    if hasattr(base, "module"):
        base = base.module
    return cast(torch.nn.Module, base)
