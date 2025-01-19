import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.amp import autocast
from src.utils.common_utils import format_time_left, wait_if_paused

def initialize_random_seeds(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_optimizer(model: torch.nn.Module, optimizer_type: str, learning_rate: float, weight_decay: float, logger=None) -> optim.Optimizer:
    optimizer_type = optimizer_type.lower()

    optimizers = {
        'adamw': optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9),
        'adam': optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
    }

    optimizer = optimizers.get(optimizer_type)
    if optimizer is None and logger:
        logger.error(f"Unsupported optimizer type '{optimizer_type}'.")
    return optimizer

def initialize_scheduler(optimizer: optim.Optimizer, scheduler_type: str, total_steps: int = None, logger=None):
    scheduler_type = scheduler_type.lower()

    schedulers = {
        'cosineannealingwarmrestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
        'cosineannealing': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
        'none': None,
    }

    if scheduler_type == 'onecyclelr':
        if total_steps is None and logger:
            logger.error("total_steps must be provided for OneCycleLR.")
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], total_steps=total_steps)

    scheduler = schedulers.get(scheduler_type)
    if scheduler is None and logger:
        logger.error(f"Unsupported scheduler type '{scheduler_type}'.")
    return scheduler

def compute_policy_loss(predicted_policies: torch.Tensor, target_policies: torch.Tensor, apply_smoothing: bool = True) -> torch.Tensor:
    if apply_smoothing:
        one_hot = torch.zeros_like(predicted_policies)
        one_hot_targets = one_hot.scatter(1, target_policies.unsqueeze(1), 1)
        smooth_factor = 0.1 / (predicted_policies.size(1) - 1)
        smoothed_targets = one_hot_targets * 0.9 + (1 - one_hot_targets) * smooth_factor
        target_policies = smoothed_targets

    log_probabilities = F.log_softmax(predicted_policies, dim=1)
    policy_loss = -(target_policies * log_probabilities).sum(dim=1).mean()
    return policy_loss

def compute_value_loss(value_preds: torch.Tensor, value_targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(value_preds.view(-1), value_targets)

def compute_total_loss(policy_loss: torch.Tensor, value_loss: torch.Tensor, batch_size: int) -> torch.Tensor:
    accumulation_steps = max(256 // batch_size, 1)
    return (policy_loss + value_loss) / accumulation_steps

def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted_classes = torch.max(predictions.data, 1)
    correct = (predicted_classes == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0

def train_epoch( model, data_loader,  device, scaler, optimizer, scheduler=None, epoch: int = 1, total_epochs: int = 1, skip_batches: int = 0, accumulation_steps: int = 1, batch_size: int = 1,
                 smooth_policy_targets: bool = False, compute_accuracy_flag: bool = False, total_batches_processed: int = 0, batch_loss_update_signal=None,
                   batch_accuracy_update_signal=None, progress_update_signal=None, time_left_update_signal=None, checkpoint_manager=None,
                     checkpoint_type: str = None, logger=None, is_stopped_event=None, is_paused_event=None, start_time=None, total_steps=None):
    model.train()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    data_iter = iter(data_loader)

    # Skip batches if resuming from a checkpoint
    if skip_batches > 0 and logger:
        logger.info(f"Skipping {skip_batches} batch(es) from checkpoint.")
        for _ in range(skip_batches):
            try:
                next(data_iter)
            except StopIteration:
                break

    accumulate_count = 0
    start_epoch_time = time.time()

    for batch_idx, (inputs, policy_targets, value_targets) in enumerate(data_iter, start=1):
        # External stop
        if is_stopped_event and is_stopped_event.is_set():
            break

        # External pause
        if is_paused_event:
            wait_if_paused(is_paused_event)

        # Move data to device
        inputs = inputs.to(device, non_blocking=True)
        policy_targets = policy_targets.to(device, non_blocking=True)
        value_targets = value_targets.to(device, non_blocking=True)

        # Forward pass (with AMP)
        with autocast("cuda", enabled=(device.type == 'cuda')):
            policy_preds, value_preds = model(inputs)
            policy_loss = compute_policy_loss(policy_preds, policy_targets, smooth_policy_targets)
            value_loss = compute_value_loss(value_preds, value_targets)
            loss = compute_total_loss(policy_loss, value_loss, batch_size)

        # Backprop (with gradient scaling)
        scaler.scale(loss).backward()
        accumulate_count += 1

        # Gradient accumulation step
        if (accumulate_count % accumulation_steps == 0) or (batch_idx == len(data_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            accumulate_count = 0

        # Scheduler step
        if scheduler and isinstance(scheduler, optim.lr_scheduler._LRScheduler):
            scheduler.step()

        # Track losses
        batch_sz = inputs.size(0)
        total_policy_loss += policy_loss.item() * batch_sz
        total_value_loss += value_loss.item() * batch_sz
        total_samples += batch_sz

        # Accuracy (optional)
        if compute_accuracy_flag:
            batch_accuracy = compute_accuracy(policy_preds, policy_targets)
            correct_predictions += batch_accuracy * batch_sz
        else:
            batch_accuracy = 0.0

        total_batches_processed += 1

        # Periodic UI/log updates
        if (total_batches_processed % 100) == 0:
            if batch_loss_update_signal:
                batch_loss_update_signal.emit(total_batches_processed, {'policy': policy_loss.item(), 'value': value_loss.item()})
            if compute_accuracy_flag and batch_accuracy_update_signal:
                batch_accuracy_update_signal.emit(total_batches_processed, batch_accuracy)
            if progress_update_signal and total_steps:
                current_progress = min(int((total_batches_processed / total_steps) * 100), 100)
                progress_update_signal.emit(current_progress)
            if time_left_update_signal and start_time and total_steps:
                elapsed_time = time.time() - start_time
                remaining_steps = total_steps - total_batches_processed
                if total_batches_processed > 0:
                    est_total_time = (elapsed_time / total_batches_processed) * remaining_steps
                    time_left_str = format_time_left(est_total_time)
                    time_left_update_signal.emit(time_left_str)
                else:
                    time_left_update_signal.emit("Calculating...")

        # Checkpoint saving if needed
        if checkpoint_manager and checkpoint_type in ("batch", "iteration"):
            if checkpoint_manager.should_save(batch_idx=total_batches_processed, iteration=total_batches_processed):
                checkpoint_data = {
                    'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'batch_idx': total_batches_processed,
                    'training_stats': {}
                }
                checkpoint_manager.save(checkpoint_data)
                if logger:
                    logger.info(f"Checkpoint saved at step {total_batches_processed}.")

        # Cleanup
        del inputs, policy_targets, value_targets, policy_preds, value_preds, loss
        torch.cuda.empty_cache()

    # Final metrics
    avg_policy_loss = (total_policy_loss / total_samples) if total_samples > 0 else float('inf')
    avg_value_loss = (total_value_loss / total_samples) if total_samples > 0 else float('inf')
    accuracy = (correct_predictions / total_samples if (total_samples > 0 and compute_accuracy_flag) else 0.0)

    epoch_duration = time.time() - start_epoch_time
    if logger:
        msg = (
            f"Finished epoch {epoch}/{total_epochs} in {format_time_left(epoch_duration)}. "
            f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, "
            f"Accuracy: {accuracy * 100:.2f}%."
        )
        logger.info(msg)

    return {
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "accuracy": accuracy,
        "total_batches_processed": total_batches_processed
    }

def validate_epoch(model, val_loader, device, epoch: int, training_accuracy: float, val_loss_update_signal=None, validation_accuracy_update_signal=None, 
                   logger=None, is_stopped_event=None, is_paused_event=None, smooth_policy_targets: bool = True):
    val_policy_loss = 0.0
    val_value_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0

    model.eval()
    with torch.no_grad():
        try:
            for inputs, policy_targets, value_targets in val_loader:
                if is_stopped_event and is_stopped_event.is_set():
                    break
                if is_paused_event:
                    wait_if_paused(is_paused_event)

                inputs = inputs.to(device, non_blocking=True)
                policy_targets = policy_targets.to(device, non_blocking=True)
                value_targets = value_targets.to(device, non_blocking=True)

                policy_preds, value_preds = model(inputs)
                policy_loss = compute_policy_loss(policy_preds, policy_targets, smooth_policy_targets)
                value_loss = compute_value_loss(value_preds, value_targets)

                batch_sz = inputs.size(0)
                val_policy_loss += policy_loss.item() * batch_sz
                val_value_loss += value_loss.item() * batch_sz

                batch_accuracy = compute_accuracy(policy_preds, policy_targets)
                val_correct_predictions += batch_accuracy * batch_sz
                val_total_predictions += batch_sz

        except Exception as e:
            if logger:
                logger.error(f"Error during validation for epoch {epoch}: {str(e)}")
            return {
                'policy_loss': float('inf'),
                'value_loss': float('inf'),
                'accuracy': 0.0
            }

    if val_total_predictions > 0:
        avg_policy_loss = val_policy_loss / val_total_predictions
        avg_value_loss = val_value_loss / val_total_predictions
        accuracy = val_correct_predictions / val_total_predictions
    else:
        avg_policy_loss, avg_value_loss, accuracy = float('inf'), float('inf'), 0.0

    if val_loss_update_signal:
        val_loss_update_signal.emit(epoch, {'policy': avg_policy_loss, 'value': avg_value_loss})
    if validation_accuracy_update_signal:
        validation_accuracy_update_signal.emit(epoch, training_accuracy, accuracy)

    if logger:
        logger.info(
            f"Epoch {epoch}: Validation Policy Loss {avg_policy_loss:.4f}, "
            f"Value Loss {avg_value_loss:.4f}, Accuracy {accuracy * 100:.2f}%."
        )

    return {
        'policy_loss': avg_policy_loss,
        'value_loss': avg_value_loss,
        'accuracy': accuracy
    }