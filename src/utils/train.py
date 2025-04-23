import os
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

def get_optimizer(model, type_name, lr, weight_decay, momentum=0.0):
    optimizers = {
        'adamw': lambda: torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay),
        'sgd': lambda: torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum),
        'adam': lambda: torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        'rmsprop': lambda: torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    }
    
    if type_name.lower() not in optimizers:
        raise ValueError(f"Invalid optimizer: {type_name}")
        
    return optimizers[type_name.lower()]()

def get_scheduler(optimizer, type_name, total_steps):
    schedulers = {
        'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2),
        'step': lambda: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1),
        'linear': lambda: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=max(total_steps // 10, 1)),
        'onecycle': lambda: torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=optimizer.param_groups[0]['lr'], total_steps=total_steps)
    }
    
    if type_name.lower() not in schedulers:
        raise ValueError(f"Invalid scheduler: {type_name}")
        
    return schedulers[type_name.lower()]()

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {"type": "gpu" if torch.cuda.is_available() else "cpu", "device": device}

def calc_loss(output, targets, value_targets, policy_weight, value_weight):
    p_pred, v_pred = output
    
    if targets.dim() == 1:
        p_loss = F.cross_entropy(p_pred, targets, label_smoothing=0.1)
    else:
        p_loss = F.kl_div(F.log_softmax(p_pred, dim=1), targets, reduction="batchmean", log_target=False)
        
    v_loss = F.mse_loss(v_pred.view(-1), value_targets)
    combined_loss = policy_weight * p_loss + value_weight * v_loss
    
    return combined_loss, p_loss, v_loss

def train_step(model, inputs, targets, value_targets, optimizer, device_info, 
              policy_weight, value_weight, accum_steps, current_step, grad_clip, 
              scheduler=None, scaler=None):
    
    device = device_info["device"]
    device_type = device_info["type"]
    use_amp = device_type == "gpu" and scaler is not None
    
    if inputs.device != device:
        inputs = inputs.to(device, non_blocking=True)
    if targets.device != device:
        targets = targets.to(device, non_blocking=True)
    if value_targets.device != device:
        value_targets = value_targets.to(device, non_blocking=True).float()
    
    with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        output = model(inputs)
        combined_loss, p_loss, v_loss = calc_loss(
            output, targets, value_targets, policy_weight, value_weight)
        combined_loss = combined_loss / accum_steps
    
    if use_amp:
        scaler.scale(combined_loss).backward()
    else:
        combined_loss.backward()
    
    if (current_step + 1) % accum_steps == 0:
        if grad_clip > 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            
        optimizer.zero_grad(set_to_none=True)
        
        if scheduler is not None:
            scheduler.step()
    
    return {
        "policy_loss": p_loss.item(),
        "value_loss": v_loss.item(),
        "combined_loss": combined_loss.item() * accum_steps
    }

def train_epoch(model, dataloader, device_info, optimizer, policy_weight, value_weight,
               accum_steps, grad_clip, scheduler=None, compute_accuracy=True, log_interval=10):
    
    model.train()
    device = device_info["device"]
    device_type = device_info["type"]
    
    total_p_loss = 0.0
    total_v_loss = 0.0
    correct = 0
    total = 0
    
    scaler = GradScaler(enabled=(device_type == "gpu"))
    
    for idx, (inputs, targets, value_targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        value_targets = value_targets.to(device, non_blocking=True).float()
        
        step_result = train_step(
            model, inputs, targets, value_targets, optimizer, device_info,
            policy_weight, value_weight, accum_steps, idx, grad_clip, 
            scheduler, scaler
        )
        
        batch_size = inputs.size(0)
        total_p_loss += step_result["policy_loss"] * batch_size
        total_v_loss += step_result["value_loss"] * batch_size
        total += batch_size
        
        if compute_accuracy and targets.dim() == 1:
            with torch.no_grad():
                p_pred = model(inputs)[0]
                pred = p_pred.argmax(dim=1)
                batch_correct = (pred == targets).float().sum().item()
                correct += batch_correct
        
        if (idx + 1) % log_interval == 0:
            print(f"Batch {idx + 1}/{len(dataloader)}: "
                  f"P_Loss = {step_result['policy_loss']:.4f}, "
                  f"V_Loss = {step_result['value_loss']:.4f}")
    
    avg_p_loss = total_p_loss / total if total > 0 else 0
    avg_v_loss = total_v_loss / total if total > 0 else 0
    accuracy = correct / total if compute_accuracy and total > 0 else 0
    
    return {
        "policy_loss": avg_p_loss,
        "value_loss": avg_v_loss,
        "accuracy": accuracy
    }

def validate(model, dataloader, device_info):
    model.eval()
    device = device_info["device"]
    device_type = device_info["type"]
    
    total_p_loss = 0.0
    total_v_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets, value_targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True).float()
            
            with autocast(device_type='cuda' if device_type == "gpu" else 'cpu', 
                         dtype=torch.float16, enabled=(device_type == "gpu")):
                p_pred, v_pred = model(inputs)
                p_loss = F.cross_entropy(p_pred, targets)
                v_loss = F.mse_loss(v_pred.view(-1), value_targets)
            
            batch_size = inputs.size(0)
            total_p_loss += p_loss.item() * batch_size
            total_v_loss += v_loss.item() * batch_size
            total += batch_size
            
            pred = p_pred.argmax(dim=1)
            batch_correct = (pred == targets).float().sum().item()
            correct += batch_correct
    
    return {
        "policy_loss": total_p_loss / total if total > 0 else 0,
        "value_loss": total_v_loss / total if total > 0 else 0,
        "accuracy": correct / total if total > 0 else 0
    }