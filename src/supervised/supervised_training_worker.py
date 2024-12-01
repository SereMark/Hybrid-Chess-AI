import os, time, numpy as np, torch, torch.optim as optim, torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.base.base_trainer import TrainerBase
from src.utils.datasets import H5Dataset
from src.utils.common_utils import format_time_left, log_message, should_stop, wait_if_paused
from src.utils.chess_utils import get_total_moves
from src.models.model import ChessModel


class SupervisedWorker(BaseWorker, TrainerBase):
    batch_loss_update = pyqtSignal(int, dict)
    batch_accuracy_update = pyqtSignal(int, float)
    epoch_loss_update = pyqtSignal(int, dict)
    epoch_accuracy_update = pyqtSignal(int, float, float)
    val_loss_update = pyqtSignal(int, dict)
    initial_batches_processed = pyqtSignal(int)
    lr_update = pyqtSignal(int, float)

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        save_checkpoints: bool,
        checkpoint_interval: int,
        dataset_path: str,
        train_indices_path: str,
        val_indices_path: str,
        checkpoint_path: str = None,
        automatic_batch_size: bool = False,
        checkpoint_type: str = 'epoch',
        checkpoint_interval_minutes: int = 60,
        checkpoint_batch_interval: int = 1000,
        optimizer_type: str = 'adamw',
        scheduler_type: str = 'cosineannealingwarmrestarts',
        output_model_path: str = 'models/saved_models/pre_trained_model.pth',
        num_workers: int = 4,
        random_seed: int = 42
    ):
        super().__init__()
        TrainerBase.__init__(
            self,
            save_checkpoints=save_checkpoints,
            checkpoint_interval=checkpoint_interval,
            checkpoint_type=checkpoint_type,
            checkpoint_interval_minutes=checkpoint_interval_minutes,
            checkpoint_batch_interval=checkpoint_batch_interval,
            checkpoint_dir=os.path.join('models', 'checkpoints', 'supervised'),
            log_fn=self.log_update.emit,
            progress_fn=self.progress_update.emit,
            time_left_fn=self.time_left_update.emit,
            stop_event=self._is_stopped,
            pause_event=self._is_paused,
            random_seed=random_seed,
            automatic_batch_size=automatic_batch_size,
            batch_size=batch_size,
            model_class=ChessModel,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            num_workers=num_workers,
            device=None
        )
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.train_indices_path = train_indices_path
        self.val_indices_path = val_indices_path
        self.checkpoint_path = checkpoint_path
        self.output_model_path = output_model_path
        self.loss_fn = self.epoch_loss_update.emit
        self.val_loss_fn = self.val_loss_update.emit
        self.accuracy_fn = self.epoch_accuracy_update.emit
        self.batch_loss_fn = self.batch_loss_update.emit
        self.batch_accuracy_fn = self.batch_accuracy_update.emit
        self.lr_fn = self.lr_update.emit
        self.initial_batches_processed_callback = self.initial_batches_processed.emit

    def run_task(self):
        self.log_update.emit("Initializing supervised training...")
        self.train_model()
        if self._is_stopped.is_set():
            self.log_update.emit("Supervised training stopped by user request.")

    def train_model(self):
        device = self.device
        log_message(f"Using device: {device}", self.log_fn)
        required_files = [
            (self.dataset_path, "Dataset file"),
            (self.train_indices_path, "Training indices"),
            (self.val_indices_path, "Validation indices"),
        ]
        for file_path, description in required_files:
            if not os.path.exists(file_path):
                log_message(f"{description} not found at {file_path}.", self.log_fn)
                return

        try:
            log_message("Preparing dataset...", self.log_fn)
            train_indices = np.load(self.train_indices_path)
            val_indices = np.load(self.val_indices_path)
            train_dataset = H5Dataset(self.dataset_path, train_indices)
            val_dataset = H5Dataset(self.dataset_path, val_indices)
            log_message(
                f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}",
                self.log_fn,
            )

            num_moves = get_total_moves()
            self.initialize_model(num_moves=num_moves)
            model = self.model

            num_workers = self.num_workers
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            self.initialize_optimizer()
            optimizer = self.optimizer

            total_steps = self.epochs * len(train_loader)
            self.initialize_scheduler(total_steps)
            scheduler = self.scheduler

            start_epoch = 1
            skip_batches = 0
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                checkpoint = self.load_checkpoint(self.checkpoint_path, map_location=device)
                if checkpoint:
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1
                        skip_batches = checkpoint.get('batch_idx', 0) or 0
                        log_message(
                            f"Resumed training from epoch {start_epoch - 1}, batch {skip_batches}",
                            self.log_fn,
                        )
                    else:
                        log_message(
                            "No epoch information found in checkpoint. Starting from epoch 1.",
                            self.log_fn,
                        )
            else:
                log_message("No checkpoint found. Starting training from scratch.", self.log_fn)

            remaining_epochs = self.epochs - (start_epoch - 1)
            total_steps = remaining_epochs * len(train_loader)
            if self.initial_batches_processed_callback:
                self.initial_batches_processed_callback(self.total_batches_processed)
            best_val_loss = float('inf')
            early_stopping_patience = 5
            no_improvement_epochs = 0

            scaler = GradScaler(device='cuda') if self.device.type == 'cuda' else GradScaler()
            desired_effective_batch_size = 256
            accumulation_steps = max(desired_effective_batch_size // self.batch_size, 1)

            for epoch in range(start_epoch, self.epochs + 1):
                epoch_start_time = time.time()
                if should_stop(self.stop_event):
                    break
                log_message(f"Epoch {epoch}/{self.epochs} started.", self.log_fn)
                model.train()
                optimizer.zero_grad()
                train_iterator = iter(train_loader)
                if epoch == start_epoch and skip_batches > 0:
                    if skip_batches >= len(train_loader):
                        log_message(
                            f"Skip batches ({skip_batches}) exceed total batches ({len(train_loader)}). Skipping entire epoch.",
                            self.log_fn,
                        )
                        continue
                    for _ in range(skip_batches):
                        try:
                            next(train_iterator)
                        except StopIteration:
                            break
                train_metrics = self._train_epoch(
                    model,
                    train_iterator,
                    optimizer,
                    scheduler,
                    epoch,
                    device,
                    total_steps,
                    train_loader,
                    scaler,
                    accumulation_steps,
                )
                if should_stop(self.stop_event):
                    break
                model.eval()
                val_metrics = self._validate_epoch(model, val_loader, epoch, device, train_metrics['accuracy'])
                total_train_loss = train_metrics['policy_loss'] + train_metrics['value_loss']
                total_val_loss = val_metrics['policy_loss'] + val_metrics['value_loss']
                epoch_duration = time.time() - epoch_start_time
                log_message(
                    f"Epoch {epoch}/{self.epochs} completed in {format_time_left(epoch_duration)} - "
                    f"Training Loss: {total_train_loss:.4f}, "
                    f"Validation Loss: {total_val_loss:.4f}, "
                    f"Training Accuracy: {train_metrics['accuracy']*100:.2f}%, "
                    f"Validation Accuracy: {val_metrics['accuracy']*100:.2f}%",
                    self.log_fn,
                )
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    no_improvement_epochs = 0
                    best_model_checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                        if hasattr(self, 'scheduler') and self.scheduler
                        else None,
                        'epoch': epoch,
                        'total_batches_processed': self.total_batches_processed,
                    }
                    torch.save(best_model_checkpoint, self.output_model_path)
                    log_message(
                        f"Best model updated at epoch {epoch} - "
                        f"Validation Loss: {total_val_loss:.4f}, "
                        f"Training Loss: {total_train_loss:.4f}",
                        self.log_fn,
                    )
                else:
                    no_improvement_epochs += 1
                    if no_improvement_epochs >= early_stopping_patience:
                        log_message("Early stopping triggered.", self.log_fn)
                        break
                if self.should_save_checkpoint(epoch=epoch, batch_idx=None):
                    self.save_checkpoint(epoch)
                if isinstance(scheduler, optim.lr_scheduler.StepLR):
                    scheduler.step()
                elif isinstance(
                    scheduler,
                    (
                        optim.lr_scheduler.CosineAnnealingWarmRestarts,
                        optim.lr_scheduler.OneCycleLR,
                    ),
                ):
                    pass
            if not should_stop(self.stop_event):
                log_message("Training completed successfully.", self.log_fn)
            else:
                log_message("Training stopped by user.", self.log_fn)
        except Exception as e:
            log_message(f"Error during training: {str(e)}", self.log_fn)

    def _train_epoch(
        self,
        model,
        train_iterator,
        optimizer,
        scheduler,
        epoch,
        device,
        total_steps,
        train_loader,
        scaler,
        accumulation_steps,
    ):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        local_steps = 0
        try:
            start_time = time.time()
            optimizer.zero_grad()
            for batch_idx, (inputs, policy_targets, value_targets) in enumerate(train_iterator, 1):
                if should_stop(self.stop_event):
                    break
                wait_if_paused(self.pause_event)
                inputs = inputs.to(device, non_blocking=True)
                policy_targets = policy_targets.to(device, non_blocking=True)
                value_targets = value_targets.to(device, non_blocking=True)
                with autocast(device_type=self.device.type):
                    policy_preds, value_preds = model(inputs)
                    smoothing = 0.1
                    confidence = 1.0 - smoothing
                    n_classes = policy_preds.size(1)
                    one_hot = torch.zeros_like(policy_preds).scatter(1, policy_targets.unsqueeze(1), 1)
                    smoothed_labels = one_hot * confidence + (1 - one_hot) * (smoothing / (n_classes - 1))

                    log_probs = F.log_softmax(policy_preds, dim=1)
                    policy_loss = -(smoothed_labels * log_probs).sum(dim=1).mean()
                    value_loss = F.mse_loss(value_preds.view(-1), value_targets)
                    loss = (policy_loss + value_loss) / accumulation_steps

                scaler.scale(loss).backward()
                if (batch_idx % accumulation_steps == 0) or (batch_idx == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                if isinstance(scheduler, (optim.lr_scheduler.CosineAnnealingWarmRestarts, optim.lr_scheduler.OneCycleLR)):
                    scheduler.step(epoch - 1 + batch_idx / len(train_loader))
                elif isinstance(scheduler, optim.lr_scheduler.StepLR):
                    pass

                total_policy_loss += policy_loss.item() * inputs.size(0)
                total_value_loss += value_loss.item() * inputs.size(0)
                _, predicted = torch.max(policy_preds.data, 1)
                total_predictions += policy_targets.size(0)
                correct_predictions += (predicted == policy_targets).sum().item()
                batch_accuracy = (predicted == policy_targets).sum().item() / policy_targets.size(0)

                with self._lock:
                    self.total_batches_processed += 1
                    local_steps += 1
                    current_progress = min(int((self.total_batches_processed / total_steps) * 100), 100)
                if self.batch_loss_fn:
                    self.batch_loss_fn(
                        self.total_batches_processed,
                        {'policy': policy_loss.item(), 'value': value_loss.item()},
                    )
                if self.batch_accuracy_fn:
                    self.batch_accuracy_fn(self.total_batches_processed, batch_accuracy)
                if self.lr_fn:
                    current_lr = optimizer.param_groups[0]['lr']
                    self.lr_fn(self.total_batches_processed, current_lr)
                if self.progress_fn:
                    self.progress_fn(current_progress)
                if self.time_left_fn:
                    elapsed_time = time.time() - start_time
                    if local_steps > 0:
                        estimated_total_time = (elapsed_time / local_steps) * (total_steps - self.total_batches_processed)
                        time_left = estimated_total_time
                        self.time_left_fn(format_time_left(time_left))
                    else:
                        self.time_left_fn("Calculating...")
                if self.should_save_checkpoint(epoch=epoch, batch_idx=self.total_batches_processed):
                    self.save_checkpoint(epoch=epoch, batch_idx=self.total_batches_processed)
                del inputs, policy_targets, value_targets, policy_preds, value_preds, loss
            metrics = {
                'policy_loss': total_policy_loss / total_predictions,
                'value_loss': total_value_loss / total_predictions,
                'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0,
            }
            log_message(f"Epoch {epoch}/{self.epochs}, Training Accuracy: {metrics['accuracy'] * 100:.2f}%", self.log_fn)
            if self.loss_fn:
                self.loss_fn(epoch, {'policy': metrics['policy_loss'], 'value': metrics['value_loss']})
            return metrics
        except Exception as e:
            log_message(f"Error during training epoch: {str(e)}", self.log_fn)
            return {'policy_loss': float('inf'), 'value_loss': float('inf'), 'accuracy': 0.0}
        finally:
            with self._lock:
                progress = min(int((self.total_batches_processed / total_steps) * 100), 100)
                if self.progress_fn:
                    self.progress_fn(progress)

    def _validate_epoch(self, model, val_loader, epoch, device, training_accuracy):
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        val_batches = 0
        try:
            with torch.no_grad():
                for inputs, policy_targets, value_targets in val_loader:
                    if should_stop(self.stop_event):
                        break
                    wait_if_paused(self.pause_event)
                    inputs = inputs.to(device)
                    policy_targets = policy_targets.to(device)
                    value_targets = value_targets.to(device)
                    policy_preds, value_preds = model(inputs)
                    policy_loss = F.cross_entropy(policy_preds, policy_targets, reduction='sum')
                    value_loss = F.mse_loss(value_preds.view(-1), value_targets, reduction='sum')
                    val_policy_loss += policy_loss.item()
                    val_value_loss += value_loss.item()
                    _, predicted = torch.max(policy_preds.data, 1)
                    val_total_predictions += policy_targets.size(0)
                    val_correct_predictions += (predicted == policy_targets).sum().item()
                    val_batches += 1
            if val_total_predictions > 0:
                metrics = {
                    'policy_loss': val_policy_loss / val_total_predictions,
                    'value_loss': val_value_loss / val_total_predictions,
                    'accuracy': val_correct_predictions / val_total_predictions,
                }
            else:
                metrics = {'policy_loss': float('inf'), 'value_loss': float('inf'), 'accuracy': 0.0}
            if self.val_loss_fn:
                self.val_loss_fn(epoch, {'policy': metrics['policy_loss'], 'value': metrics['value_loss']})
            if self.accuracy_fn:
                self.accuracy_fn(epoch, training_accuracy, metrics['accuracy'])
            log_message(
                f"Epoch {epoch}/{self.epochs}, "
                f"Validation Policy Loss: {metrics['policy_loss']:.4f}, "
                f"Validation Value Loss: {metrics['value_loss']:.4f}, "
                f"Validation Accuracy: {metrics['accuracy'] * 100:.2f}%",
                self.log_fn,
            )
            return metrics
        except Exception as e:
            log_message(f"Error during validation: {str(e)}", self.log_fn)
            return {'policy_loss': float('inf'), 'value_loss': float('inf'), 'accuracy': 0.0}