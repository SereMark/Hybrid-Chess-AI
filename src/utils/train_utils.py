from torch.cuda.amp import autocast
from torch.nn import functional as F
import torch, random, numpy as np, torch.optim as optim

def initialize_random_seeds(random_seed:int)->None:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def initialize_optimizer(model:torch.nn.Module, optimizer_type:str, learning_rate:float, weight_decay:float)->optim.Optimizer:
    optimizer_type=optimizer_type.lower()
    optimizers={
        'adamw': optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9),
        'adam': optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'rmsprop': optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
    }
    return optimizers.get(optimizer_type, optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay))

def initialize_scheduler(optimizer:optim.Optimizer, scheduler_type:str, total_steps:int=None):
    scheduler_type=scheduler_type.lower()
    schedulers={
        'cosineannealingwarmrestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
        'cosineannealing': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
        'steplr': optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
        'exponentiallr': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
        'onelr': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], total_steps=total_steps)
    }
    return schedulers.get(scheduler_type) if scheduler_type != 'none' else None

def compute_policy_loss(predicted_policies:torch.Tensor, target_policies:torch.Tensor, apply_smoothing:bool=True)->torch.Tensor:
    if apply_smoothing:
        one_hot=torch.zeros_like(predicted_policies)
        one_hot_targets=one_hot.scatter(1, target_policies.unsqueeze(1), 1)
        smooth_factor=0.1 / (predicted_policies.size(1)-1)
        target_policies=one_hot_targets * 0.9 + (1 - one_hot_targets) * smooth_factor
    return -(target_policies * F.log_softmax(predicted_policies, dim=1)).sum(dim=1).mean()

def compute_value_loss(value_preds:torch.Tensor, value_targets:torch.Tensor)->torch.Tensor:
    return F.mse_loss(value_preds.view(-1), value_targets)

def compute_total_loss(policy_loss:torch.Tensor, value_loss:torch.Tensor, batch_size:int)->torch.Tensor:
    accumulation_steps=max(256 // batch_size, 1)
    return (policy_loss + value_loss) / accumulation_steps

def compute_accuracy(predictions:torch.Tensor, targets:torch.Tensor)->float:
    _, predicted_classes = torch.max(predictions.data, 1)
    correct = (predicted_classes == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0

def train_epoch(model, data_loader, device, scaler, optimizer, scheduler=None, epoch=1, skip_batches:int=0, accumulation_steps:int=1, batch_size:int=1, smooth_policy_targets:bool=False, compute_accuracy_flag:bool=False, total_batches_processed:int=0, progress_callback=None, status_callback=None):
    model.train()
    total_policy_loss, total_value_loss, correct_predictions, total_samples = 0.0, 0.0, 0, 0
    data_iter = iter(data_loader)
    for _ in range(skip_batches):
        try:
            next(data_iter)
        except StopIteration:
            break
    accumulate_count = 0
    for batch_idx, (inputs, policy_targets, value_targets) in enumerate(data_iter, start=1):
        inputs, policy_targets, value_targets = inputs.to(device, non_blocking=True), policy_targets.to(device, non_blocking=True), value_targets.to(device, non_blocking=True)
        with autocast():
            policy_preds, value_preds = model(inputs)
            policy_loss = compute_policy_loss(policy_preds, policy_targets, smooth_policy_targets)
            value_loss = compute_value_loss(value_preds, value_targets)
            loss = compute_total_loss(policy_loss, value_loss, batch_size)
        scaler.scale(loss).backward()
        accumulate_count +=1
        if accumulate_count % accumulation_steps == 0 or batch_idx == len(data_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            accumulate_count = 0
        if scheduler and isinstance(scheduler, optim.lr_scheduler._LRScheduler):
            scheduler.step()
        batch_sz = inputs.size(0)
        total_policy_loss += policy_loss.item() * batch_sz
        total_value_loss += value_loss.item() * batch_sz
        total_samples += batch_sz
        if compute_accuracy_flag:
            correct_predictions += compute_accuracy(policy_preds, policy_targets) * batch_sz
        total_batches_processed +=1
        progress_callback(total_batches_processed / len(data_loader) * 100)
        status_callback(f"Epoch {epoch} - Batch {batch_idx}/{len(data_loader)} - Policy Loss: {total_policy_loss / total_samples:.4f} - Value Loss: {total_value_loss / total_samples:.4f}")
        del inputs, policy_targets, value_targets, policy_preds, value_preds, loss
        torch.cuda.empty_cache()
    progress_callback(100)

def validate_epoch(model, val_loader, device, epoch:int, smooth_policy_targets:bool=True, progress_callback=None, status_callback=None):
    model.eval()
    val_policy_loss, val_value_loss, val_correct_predictions, val_total_predictions = 0.0, 0.0, 0, 0
    total_batches = len(val_loader)
    with torch.no_grad():
        for batch_idx, (inputs, policy_targets, value_targets) in enumerate(val_loader, start=1):
            inputs, policy_targets, value_targets = inputs.to(device, non_blocking=True), policy_targets.to(device, non_blocking=True), value_targets.to(device, non_blocking=True)
            policy_preds, value_preds = model(inputs)
            policy_loss = compute_policy_loss(policy_preds, policy_targets, smooth_policy_targets)
            value_loss = compute_value_loss(value_preds, value_targets)
            batch_sz = inputs.size(0)
            val_policy_loss += policy_loss.item() * batch_sz
            val_value_loss += value_loss.item() * batch_sz
            val_correct_predictions += compute_accuracy(policy_preds, policy_targets) * batch_sz
            val_total_predictions += batch_sz
            progress_callback(batch_idx / total_batches * 100)
            status_callback(f"Validation - Epoch {epoch} - Batch {batch_idx}/{total_batches} - Policy Loss: {val_policy_loss / val_total_predictions:.4f} - Value Loss: {val_value_loss / val_total_predictions:.4f}")
            del inputs, policy_targets, value_targets, policy_preds, value_preds, policy_loss, value_loss
        torch.cuda.empty_cache()
    progress_callback(100)