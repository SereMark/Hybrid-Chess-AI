import os, time, numpy as np, torch, torch.optim as optim, torch.nn.functional as F, threading
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.models.model import ChessModel
from src.utils.datasets import H5Dataset
from src.utils.common_utils import (
    format_time_left,
    wait_if_paused,
    initialize_model,
    initialize_optimizer,
    initialize_scheduler,
    initialize_random_seeds
)
from src.utils.chess_utils import get_total_moves
from src.utils.checkpoint_manager import CheckpointManager

class SupervisedWorker(BaseWorker):
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
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_batch_interval = checkpoint_batch_interval
        self.checkpoint_dir = os.path.join('models', 'checkpoints', 'supervised')
        self.random_seed = random_seed
        self.automatic_batch_size = automatic_batch_size
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.num_workers = num_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        initialize_random_seeds(self.random_seed)

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
        self.log_fn = self.log_update.emit

        self.lock = threading.Lock()
        self.total_batches_processed = 0

        self.model, inferred_batch_size = initialize_model(
            ChessModel, num_moves=get_total_moves(), device=self.device, automatic_batch_size=self.automatic_batch_size, log_fn=self.log_fn
        )
        if inferred_batch_size:
            self.batch_size = inferred_batch_size

        self.optimizer = initialize_optimizer(
            self.model,
            self.optimizer_type,
            self.learning_rate,
            self.weight_decay,
            log_fn=self.log_fn
        )

        self.scheduler = None

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_type=self.checkpoint_type,
            checkpoint_interval=self.checkpoint_interval,
            log_fn=self.log_fn
        )

    def run_task(self):
        self.log_update.emit("Starting training process...")
        device = self.device
        self.log_update.emit(f"Using device: {device}")

        required_files = [
            (self.dataset_path, "Dataset file"),
            (self.train_indices_path, "Training indices"),
            (self.val_indices_path, "Validation indices"),
        ]
        for file_path, description in required_files:
            if not os.path.exists(file_path):
                self.log_update.emit(f"{description} not found at {file_path}.")
                return

        try:
            self.log_update.emit("Preparing dataset...")
            train_indices = np.load(self.train_indices_path)
            val_indices = np.load(self.val_indices_path)
            train_dataset = H5Dataset(self.dataset_path, train_indices)
            val_dataset = H5Dataset(self.dataset_path, val_indices)
            self.log_update.emit(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            total_steps = self.epochs * len(train_loader)
            try:
                self.scheduler = initialize_scheduler(
                    self.optimizer,
                    self.scheduler_type,
                    total_steps=total_steps,
                    log_fn=self.log_fn
                )
            except ValueError as ve:
                self.log_update.emit(str(ve))
                self.scheduler = None

            start_epoch = 1
            skip_batches = 0
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                checkpoint = self.checkpoint_manager.load(
                    self.checkpoint_path, self.device, self.model, self.optimizer, self.scheduler
                )
                if checkpoint:
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1
                        skip_batches = checkpoint.get('batch_idx', 0) or 0
                        self.log_update.emit(f"Resumed training from epoch {start_epoch - 1}, batch {skip_batches}")
                    else:
                        self.log_update.emit("No epoch information found in checkpoint. Starting from epoch 1.")
            else:
                self.log_update.emit("No checkpoint found. Starting training from scratch.")

            if self.initial_batches_processed_callback:
                self.initial_batches_processed_callback(self.total_batches_processed)

            scaler = GradScaler(device='cuda') if self.device.type == 'cuda' else GradScaler()
            desired_effective_batch_size = 256
            accumulation_steps = max(desired_effective_batch_size // self.batch_size, 1)

            for epoch in range(start_epoch, self.epochs + 1):
                if self._is_stopped.is_set():
                    break
                epoch_start_time = time.time()
                self.log_update.emit(f"Epoch {epoch}/{self.epochs} started.")
                self.model.train()
                train_metrics = self._train_epoch(
                    train_loader,
                    epoch,
                    device,
                    total_steps,
                    scaler,
                    accumulation_steps,
                    skip_batches if epoch == start_epoch else 0
                )
                if self._is_stopped.is_set():
                    break
                self.model.eval()
                val_metrics = self._validate_epoch(val_loader, epoch, device, train_metrics['accuracy'])
                total_train_loss = train_metrics['policy_loss'] + train_metrics['value_loss']
                total_val_loss = val_metrics['policy_loss'] + val_metrics['value_loss']
                epoch_duration = time.time() - epoch_start_time
                self.log_update.emit(
                    f"Epoch {epoch}/{self.epochs} completed in {format_time_left(epoch_duration)} - "
                    f"Training Loss: {total_train_loss:.4f}, "
                    f"Validation Loss: {total_val_loss:.4f}, "
                    f"Training Accuracy: {train_metrics['accuracy']*100:.2f}%, "
                    f"Validation Accuracy: {val_metrics['accuracy']*100:.2f}%"
                )

                if self.save_checkpoints and self.checkpoint_type == 'epoch' and self.checkpoint_manager.should_save(epoch=epoch):
                    checkpoint_data = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'epoch': epoch,
                        'batch_idx': None,
                        'iteration': None,
                        'training_stats': {},
                    }
                    self.checkpoint_manager.save(checkpoint_data)

                if isinstance(self.scheduler, optim.lr_scheduler.StepLR):
                    self.scheduler.step()
                elif isinstance(self.scheduler, (optim.lr_scheduler.CosineAnnealingWarmRestarts, optim.lr_scheduler.OneCycleLR)):
                    pass

            if not self._is_stopped.is_set():
                self.log_update.emit("Training completed successfully.")
                try:
                    checkpoint = {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                        "epoch": epoch,
                        "batch_idx": self.total_batches_processed,
                    }
                    torch.save(checkpoint, self.output_model_path)
                    self.log_update.emit(f"Final model saved to {self.output_model_path}")
                except Exception as e:
                    self.log_update.emit(f"Error saving final model: {str(e)}")
            else:
                self.log_update.emit("Training stopped by user.")
        except Exception as e:
            self.log_update.emit(f"Error during training: {str(e)}")

        self.task_finished.emit()
        self.finished.emit()

    def _train_epoch(self, train_loader, epoch, device, total_steps, scaler, accumulation_steps, skip_batches):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        local_steps = 0
        try:
            start_time = time.time()
            if skip_batches:
                self.log_update.emit(f"Skipping {skip_batches} batches for resuming.")
            for batch_idx, (inputs, policy_targets, value_targets) in enumerate(train_loader, 1):
                if self._is_stopped.is_set():
                    break
                if skip_batches > 0:
                    skip_batches -= 1
                    continue
                wait_if_paused(self._is_paused)
                inputs = inputs.to(device, non_blocking=True)
                policy_targets = policy_targets.to(device, non_blocking=True)
                value_targets = value_targets.to(device, non_blocking=True)
                with autocast(device_type=self.device.type):
                    policy_preds, value_preds = self.model(inputs)
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
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                if isinstance(self.scheduler, (optim.lr_scheduler.CosineAnnealingWarmRestarts, optim.lr_scheduler.OneCycleLR)):
                    self.scheduler.step(epoch - 1 + batch_idx / len(train_loader))

                total_policy_loss += policy_loss.item() * inputs.size(0)
                total_value_loss += value_loss.item() * inputs.size(0)
                _, predicted = torch.max(policy_preds.data, 1)
                total_predictions += policy_targets.size(0)
                correct_predictions += (predicted == policy_targets).sum().item()
                batch_accuracy = (predicted == policy_targets).sum().item() / policy_targets.size(0)

                with self.lock:
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
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.lr_fn(self.total_batches_processed, current_lr)
                if self.progress_update:
                    self.progress_update.emit(current_progress)
                if self.time_left_update:
                    elapsed_time = time.time() - start_time
                    if local_steps > 0:
                        estimated_total_time = (elapsed_time / local_steps) * (total_steps - self.total_batches_processed)
                        time_left = estimated_total_time
                        self.time_left_update.emit(format_time_left(time_left))
                    else:
                        self.time_left_update.emit("Calculating...")

                if self.save_checkpoints and self.checkpoint_type in ['batch', 'iteration'] and self.checkpoint_manager.should_save(batch_idx=self.total_batches_processed):
                    checkpoint_data = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'epoch': epoch,
                        'batch_idx': self.total_batches_processed,
                        'iteration': None,
                        'training_stats': {},
                    }
                    self.checkpoint_manager.save(checkpoint_data)

                del inputs, policy_targets, value_targets, policy_preds, value_preds, loss
                torch.cuda.empty_cache()

            metrics = {
                'policy_loss': total_policy_loss / total_predictions if total_predictions > 0 else float('inf'),
                'value_loss': total_value_loss / total_predictions if total_predictions > 0 else float('inf'),
                'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0,
            }
            self.log_update.emit(f"Epoch {epoch}/{self.epochs}, Training Accuracy: {metrics['accuracy'] * 100:.2f}%")
            if self.loss_fn:
                self.loss_fn(epoch, {'policy': metrics['policy_loss'], 'value': metrics['value_loss']})
            return metrics
        except Exception as e:
            self.log_update.emit(f"Error during training epoch: {str(e)}")
            return {'policy_loss': float('inf'), 'value_loss': float('inf'), 'accuracy': 0.0}
        finally:
            with self.lock:
                progress = min(int((self.total_batches_processed / total_steps) * 100), 100)
                if self.progress_update:
                    self.progress_update.emit(progress)

    def _validate_epoch(self, val_loader, epoch, device, training_accuracy):
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        try:
            with torch.no_grad():
                for inputs, policy_targets, value_targets in val_loader:
                    if self._is_stopped.is_set():
                        break
                    wait_if_paused(self._is_paused)
                    inputs = inputs.to(device, non_blocking=True)
                    policy_targets = policy_targets.to(device, non_blocking=True)
                    value_targets = value_targets.to(device, non_blocking=True)
                    policy_preds, value_preds = self.model(inputs)
                    policy_loss = F.cross_entropy(policy_preds, policy_targets, reduction='sum')
                    value_loss = F.mse_loss(value_preds.view(-1), value_targets, reduction='sum')
                    val_policy_loss += policy_loss.item()
                    val_value_loss += value_loss.item()
                    _, predicted = torch.max(policy_preds.data, 1)
                    val_total_predictions += policy_targets.size(0)
                    val_correct_predictions += (predicted == policy_targets).sum().item()

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
            self.log_update.emit(
                f"Epoch {epoch}/{self.epochs}, "
                f"Validation Policy Loss: {metrics['policy_loss']:.4f}, "
                f"Validation Value Loss: {metrics['value_loss']:.4f}, "
                f"Validation Accuracy: {metrics['accuracy'] * 100:.2f}%"
            )
            return metrics
        except Exception as e:
            self.log_update.emit(f"Error during validation: {str(e)}")
            return {'policy_loss': float('inf'), 'value_loss': float('inf'), 'accuracy': 0.0}