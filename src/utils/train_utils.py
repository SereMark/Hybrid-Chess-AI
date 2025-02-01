import torch
import random
import numpy as np
import torch.optim as optim
from torch.amp import autocast
import torch.nn.functional as F

def initialize_random_seeds(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def initialize_optimizer(model, optimizer_type, lr, wd, momentum):
    optim_dict = {
        'adamw': optim.AdamW(model.parameters(), lr=lr, weight_decay=wd),
        'sgd': torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum),
        'adam': torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd),
        'rmsprop': torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    }
    opt = optim_dict.get(optimizer_type.lower())
    if opt is None:
        raise ValueError("Invalid optimizer type")
    return opt

def initialize_scheduler(optimizer, scheduler_type, total_steps):
    import torch
    sched_dict = {
        'cosineannealingwarmrestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
        'step': torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
        'linear': torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(total_steps // 10, 1)),
        'onecycle': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], total_steps=total_steps)
    }
    sched = sched_dict.get(scheduler_type.lower())
    if sched is None:
        raise ValueError("Invalid scheduler type")
    return sched

def train_epoch(model, loader, device, scaler, optimizer, scheduler, epoch, max_epoch, accum_steps, compute_acc, p_w, v_w, max_grad, prog_cb, status_cb, use_wandb):
    model.train()
    total_p_loss, total_v_loss, correct, total, accum = 0, 0, 0, 0, 0
    for idx, (inp, pol, val) in enumerate(loader, start=1):
        inp = inp.to(device, non_blocking=True)
        pol = pol.to(device, non_blocking=True)
        pol = pol.long() if pol.dim() == 1 else pol.float()
        val = val.to(device, non_blocking=True).float()
        with autocast(device_type=device.type):
            p_pred, v_pred = model(inp)
            if pol.dim() == 1:
                p_loss = F.cross_entropy(p_pred, pol, label_smoothing=0.1)
            else:
                p_loss = F.kl_div(F.log_softmax(p_pred, dim=1), pol, reduction="batchmean", log_target=False)
            v_loss = F.mse_loss(v_pred.view(-1), val)
            loss = (p_w * p_loss + v_w * v_loss) / accum_steps
        scaler.scale(loss).backward()
        accum += 1
        grad_norm = float('nan')
        if (accum % accum_steps == 0) or (idx == len(loader)):
            if max_grad > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum = 0
        bs = inp.size(0)
        total_p_loss += p_loss.item() * bs
        total_v_loss += v_loss.item() * bs
        total += bs
        batch_correct = (p_pred.argmax(dim=1) == pol).float().sum().item() if compute_acc and pol.dim() == 1 else 0
        correct += batch_correct
        if use_wandb and idx % 10 == 0:
            from src.utils.common import wandb_log
            wandb_log({
                "train/policy_loss": p_loss.item(),
                "train/value_loss": v_loss.item(),
                "train/accuracy": batch_correct / bs if compute_acc and pol.dim() == 1 else float('nan'),
                "train/grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })
        if prog_cb:
            prog_cb(idx / len(loader) * 100)
        if status_cb and idx % 10 == 0:
            status_cb(f"📊 Epoch {epoch}/{max_epoch} | Batch {idx}/{len(loader)} | Policy Loss: {total_p_loss/total:.4f} | Value Loss: {total_v_loss/total:.4f} | Accuracy: {correct/total:.4f}")
    return {
        "policy_loss": total_p_loss / total if total else 0,
        "value_loss": total_v_loss / total if total else 0,
        "accuracy": correct / total if (compute_acc and total) else 0
    }

def validate_epoch(model, loader, device, epoch, max_epoch, prog_cb, status_cb):
    model.eval()
    val_p_loss, val_v_loss, correct, total = 0, 0, 0, 0
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
            total += bs
            batch_correct = (p_pred.argmax(dim=1) == pol).float().sum().item()
            correct += batch_correct
            if prog_cb:
                prog_cb(idx / len(loader) * 100)
            if status_cb and idx % 10 == 0:
                status_cb(f"🔍 Validation Epoch {epoch}/{max_epoch} | Batch {idx}/{len(loader)} | Policy Loss: {val_p_loss/total:.4f} | Value Loss: {val_v_loss/total:.4f} | Accuracy: {correct/total:.4f}")
    return {
        "policy_loss": val_p_loss / total if total else 0,
        "value_loss": val_v_loss / total if total else 0,
        "accuracy": correct / total if total else 0
    }