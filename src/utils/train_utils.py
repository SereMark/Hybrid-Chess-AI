from torch.amp import autocast
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

def initialize_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_optimizer(model: torch.nn.Module, optimizer_type: str, lr: float, wd: float, momentum: float) -> optim.Optimizer:
    optim_dict = {
        'adamw': optim.AdamW(model.parameters(), lr=lr, weight_decay=wd),
        'sgd':   optim.SGD(model.parameters(),  lr=lr, weight_decay=wd, momentum=momentum),
        'adam':  optim.Adam(model.parameters(), lr=lr, weight_decay=wd),
        'rmsprop': optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    }
    if optimizer_type.lower() not in optim_dict:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    return optim_dict[optimizer_type.lower()]

def initialize_scheduler(optimizer: optim.Optimizer, scheduler_type: str, total_steps: int) -> optim.lr_scheduler._LRScheduler:
    sched_dict = {
        'cosineannealingwarmrestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
        'step':   optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
        'linear': optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(total_steps // 10, 1)),
        'onecycle': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], total_steps=total_steps),
    }
    if scheduler_type.lower() not in sched_dict:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    return sched_dict[scheduler_type.lower()]

def train_epoch(
    model,
    loader,
    device,
    scaler,
    optimizer,
    scheduler,
    epoch,
    max_epoch,
    accum_steps,
    compute_acc,
    p_w,
    v_w,
    max_grad,
    prog_cb,
    status_cb,
    use_wandb
):
    model.train()
    total_p_loss = 0.0
    total_v_loss = 0.0
    correct       = 0.0
    total         = 0.0
    accum         = 0

    if use_wandb:
        import wandb

    for idx, (inp, pol, val) in enumerate(loader, start=1):
        inp = inp.to(device, non_blocking=True)
        pol = pol.to(device, non_blocking=True)
        if pol.dim() == 1: pol = pol.long()
        else: pol = pol.float()
        val = val.to(device, non_blocking=True).float()

        with autocast(device_type=device.type):
            p_pred, v_pred = model(inp)

            if pol.dim() == 1:
                p_loss = F.cross_entropy(p_pred, pol, label_smoothing=0.1)
            else:
                p_loss = -(pol * F.log_softmax(p_pred, dim=1)).sum(dim=1).mean()

            v_loss = F.mse_loss(v_pred.view(-1), val)

            loss = (p_w * p_loss + v_w * v_loss) / accum_steps

        scaler.scale(loss).backward()
        accum += 1

        if (accum % accum_steps == 0) or (idx == len(loader)):
            if max_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum = 0

        bs = inp.size(0)
        total_p_loss += p_loss.item() * bs
        total_v_loss += v_loss.item() * bs
        total        += bs

        batch_correct = 0.0
        if compute_acc and pol.dim() == 1:
            batch_correct = (p_pred.argmax(dim=1) == pol).float().sum().item()

        if use_wandb:
            wandb.log({
                "train_policy_loss": p_loss.item(),
                "train_value_loss":  v_loss.item(),
                "train_step_accuracy": (batch_correct / bs) if (compute_acc and pol.dim() == 1) else float('nan'),
                "learning_rate": scheduler.get_last_lr()[0],
            })

        correct += batch_correct

        if prog_cb:
            prog_cb(idx / len(loader) * 100)
        if status_cb and idx % 10 == 0:
            status_cb(
                f"Epoch {epoch}/{max_epoch} | Batch {idx}/{len(loader)} "
                f"| Policy Loss: {total_p_loss / total:.4f} "
                f"| Value Loss: {total_v_loss / total:.4f}"
            )

    return {
        "policy_loss": total_p_loss / total if total else 0.0,
        "value_loss":  total_v_loss / total if total else 0.0,
        "accuracy":    (correct / total) if (compute_acc and total) else 0.0
    }

def validate_epoch(
    model,
    loader,
    device,
    epoch,
    max_epoch,
    prog_cb,
    status_cb,
    use_wandb
):
    model.eval()
    val_p_loss = 0.0
    val_v_loss = 0.0
    correct    = 0.0
    total      = 0.0

    if use_wandb:
        import wandb

    with torch.no_grad():
        for idx, (inp, pol, val) in enumerate(loader, start=1):
            inp = inp.to(device, non_blocking=True)
            pol = pol.to(device, non_blocking=True).long()
            val = val.to(device, non_blocking=True).float()

            with autocast(device_type=device.type):
                p_pred, v_pred = model(inp)
                p_loss = F.cross_entropy(p_pred, pol, label_smoothing=0.0)
                v_loss = F.mse_loss(v_pred.view(-1), val)

            bs = inp.size(0)
            val_p_loss += p_loss.item() * bs
            val_v_loss += v_loss.item() * bs
            total      += bs

            batch_correct = (p_pred.argmax(dim=1) == pol).float().sum().item()
            correct      += batch_correct

            if use_wandb:
                wandb.log({
                    "val_policy_loss": p_loss.item(),
                    "val_value_loss":  v_loss.item(),
                    "val_step_accuracy": batch_correct / bs,
                })

            if prog_cb:
                prog_cb(idx / len(loader) * 100)
            if status_cb and idx % 10 == 0:
                status_cb(
                    f"Validation Epoch {epoch}/{max_epoch} | Batch {idx}/{len(loader)} "
                    f"| Policy Loss: {val_p_loss / total:.4f} "
                    f"| Value Loss: {val_v_loss / total:.4f}"
                )

    return {
        "policy_loss": val_p_loss / total if total else 0.0,
        "value_loss":  val_v_loss / total if total else 0.0,
        "accuracy":    (correct / total) if total else 0.0
    }