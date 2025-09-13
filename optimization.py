from __future__ import annotations

from typing import Any

import torch

import config as C


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
            {"params": decay, "weight_decay": C.TRAIN.WEIGHT_DECAY},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=C.TRAIN.LR_INIT,
        momentum=C.TRAIN.MOMENTUM,
        nesterov=True,
    )
    return optimizer


class WarmupCosine:

    def __init__(
        self,
        optimizer: Any,
        base_lr: float,
        warmup_steps: int,
        final_lr: float,
        total_steps: int,
    ) -> None:
        self.opt = optimizer
        self.base = float(base_lr)
        self.warm = max(1, int(warmup_steps))
        self.final = float(final_lr)
        self.total = max(1, int(total_steps))
        self.t = 0

    def step(self) -> None:
        self.t += 1
        if self.t <= self.warm:
            lr = self.base * (self.t / self.warm)
        else:
            import math as _m

            progress = min(1.0, (self.t - self.warm) / max(1, self.total - self.warm))
            lr = self.final + (self.base - self.final) * 0.5 * (1.0 + _m.cos(_m.pi * progress))
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def peek_next_lr(self) -> float:
        t_next = self.t + 1
        if t_next <= self.warm:
            return self.base * (t_next / self.warm)
        import math as _m

        progress = min(1.0, (t_next - self.warm) / max(1, self.total - self.warm))
        return self.final + (self.base - self.final) * 0.5 * (1.0 + _m.cos(_m.pi * progress))

    def set_total_steps(self, total_steps: int) -> None:
        self.total = max(self.t + 1, int(total_steps))
        if self.warm >= self.total:
            self.warm = max(1, self.total - 1)


class EMA:

    def __init__(self, model: torch.nn.Module, decay: float | None = None) -> None:
        self.decay = float(C.TRAIN.EMA_DECAY if decay is None else decay)
        base = getattr(model, "_orig_mod", model)
        if hasattr(base, "module"):
            base = base.module
        self.shadow = {k: v.detach().clone() for k, v in base.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        base = getattr(model, "_orig_mod", model)
        if hasattr(base, "module"):
            base = base.module
        for k, v in base.state_dict().items():
            if not torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
                continue
            if self.shadow[k].dtype != v.dtype:
                self.shadow[k] = self.shadow[k].to(dtype=v.dtype)
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        base = getattr(model, "_orig_mod", model)
        if hasattr(base, "module"):
            base = base.module
        base.load_state_dict(self.shadow, strict=True)
