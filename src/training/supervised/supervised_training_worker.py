import os
import time
import threading
from typing import Optional, Dict
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.models.model import ChessModel
from src.utils.datasets import H5Dataset
from src.utils.common_utils import format_time_left, initialize_optimizer, initialize_scheduler, initialize_random_seeds, wait_if_paused, compute_policy_loss, compute_value_loss, compute_total_loss, compute_accuracy
from src.utils.chess_utils import get_total_moves
from src.utils.checkpoint_manager import CheckpointManager

class SupervisedWorker(BaseWorker):
    batch_loss_update = pyqtSignal(int, dict)
    batch_accuracy_update = pyqtSignal(int, float)
    val_loss_update = pyqtSignal(int, dict)
    validation_accuracy_update = pyqtSignal(int, float, float)

    def __init__(self, epochs: int, batch_size: int, learning_rate: float, weight_decay: float, save_checkpoints: bool, checkpoint_interval: int, dataset_path: str, train_indices_path: str, val_indices_path: str, model_path: Optional[str] = None, checkpoint_type: str = 'epoch', checkpoint_interval_minutes: int = 60, checkpoint_batch_interval: int = 1000, optimizer_type: str = 'adamw', scheduler_type: str = 'cosineannealingwarmrestarts', num_workers: int = 4, random_seed: int = 42):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_batch_interval = checkpoint_batch_interval
        self.dataset_path = dataset_path
        self.train_indices_path = train_indices_path
        self.val_indices_path = val_indices_path
        self.model_path = model_path
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.num_workers = num_workers
        self.random_seed = random_seed

        # Initialize random seeds for reproducibility
        initialize_random_seeds(self.random_seed)

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = ChessModel(get_total_moves()).to(self.device)

        # Initialize optimizer
        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay, logger=self.logger)

        # Initialize scheduler
        self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, total_steps=self.epochs*self._get_total_steps(), logger=self.logger)

        # Initialize checkpoint manager
        self.checkpoint_dir = os.path.join('models', 'checkpoints', 'supervised')
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir, checkpoint_type=self.checkpoint_type, checkpoint_interval=self.checkpoint_interval, logger=self.logger)

        # Initialize GradScaler for mixed precision
        self.scaler = GradScaler(device='cuda') if self.device.type == 'cuda' else GradScaler()

        # Tracking variables
        self.total_batches_processed = 0

    def _get_total_steps(self) -> int:
        try:
            train_indices = np.load(self.train_indices_path)
            train_dataset = H5Dataset(self.dataset_path, train_indices)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
            return len(train_loader)
        except Exception as e:
            self.logger.error(f"Error calculating total steps: {str(e)}")
            return 1

    def run_task(self):
        self.logger.info("Starting supervised training worker.")
        self.logger.info(f"Using device: {self.device}")

        # Validate required files
        required_files = [(self.dataset_path, "dataset file"), (self.train_indices_path, "training indices"), (self.val_indices_path, "validation indices")]
        for file_path, description in required_files:
            if not os.path.exists(file_path):
                self.logger.error(f"Required {description} was not found at {file_path}. Aborting.")
                return

        try:
            # Load datasets
            self.logger.info("Loading training and validation datasets.")
            train_indices = np.load(self.train_indices_path)
            val_indices = np.load(self.val_indices_path)

            train_dataset = H5Dataset(self.dataset_path, train_indices)
            val_dataset = H5Dataset(self.dataset_path, val_indices)

            self.logger.info(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}.")

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

            # Load checkpoint if available
            start_epoch = 1
            skip_batches = 0
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = self.checkpoint_manager.load(self.model_path, self.device, self.model, self.optimizer, self.scheduler)
                if checkpoint:
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    skip_batches = checkpoint.get('batch_idx', 0) or 0
                    self.total_batches_processed = skip_batches
                    self.logger.info(f"Resuming from epoch {start_epoch - 1}, batch {skip_batches}.")
                else:
                    self.logger.warning("No valid checkpoint found. Starting from epoch 1.")
            else:
                self.logger.info("No checkpoint path provided or checkpoint does not exist. Training from scratch.")

            # Training loop
            for epoch in range(start_epoch, self.epochs + 1):
                if self._is_stopped.is_set():
                    break

                epoch_start_time = time.time()
                self.logger.info(f"Beginning epoch {epoch}/{self.epochs}.")

                # Train the current epoch
                self.model.train()
                train_metrics = self._train_epoch(train_loader, epoch, self.device, self.scaler, skip_batches if epoch == start_epoch else 0)
                skip_batches = 0

                if self._is_stopped.is_set():
                    break

                # Validate the current epoch
                self.model.eval()
                val_metrics = self._validate_epoch(val_loader, epoch, self.device, train_metrics['accuracy'])

                # Calculate total losses
                total_train_loss = train_metrics['policy_loss'] + train_metrics['value_loss']
                total_val_loss = val_metrics['policy_loss'] + val_metrics['value_loss']

                # Calculate epoch duration
                epoch_duration = time.time() - epoch_start_time
                self.logger.info(
                    f"Epoch {epoch}/{self.epochs} completed in {format_time_left(epoch_duration)}. "
                    f"Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']*100:.2f}%, Val Acc: {val_metrics['accuracy']*100:.2f}%."
                )

                # Emit epoch accuracy update
                self.validation_accuracy_update.emit(epoch, train_metrics['accuracy'], val_metrics['accuracy'])

                # Save checkpoint if required
                if self.save_checkpoints and self.checkpoint_type == 'epoch' and self.checkpoint_manager.should_save(epoch=epoch):
                    self._save_checkpoint(epoch)

            # Save the final model if training completed without interruption
            if not self._is_stopped.is_set():
                try:
                    final_dir = os.path.join("models", "saved_models")
                    final_path = os.path.join(final_dir, "pre_trained_model.pth")
                    os.makedirs(final_dir, exist_ok=True)
                    checkpoint = {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                        "epoch": epoch,
                        "batch_idx": self.total_batches_processed,
                        "training_stats": {}
                    }
                    torch.save(checkpoint, final_path)
                    self.logger.info(f"Final model saved at {final_path}")
                except Exception as e:
                    self.logger.error(f"Error saving final model: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error in SupervisedWorker: {str(e)}")
        finally:
            self.task_finished.emit()
            self.finished.emit()

    def _save_checkpoint(self, epoch: int):
        checkpoint_data = {
            'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'batch_idx': self.total_batches_processed,
            'training_stats': {}
        }
        self.checkpoint_manager.save(checkpoint_data)
        self.logger.info(f"Checkpoint saved at epoch {epoch}.")

    def _train_epoch(self, train_loader: DataLoader, epoch: int, device: torch.device, scaler: GradScaler, skip_batches: int) -> Dict[str, float]:
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        accumulate_count = 0

        try:
            start_time = time.time()
            if skip_batches > 0:
                self.logger.info(f"Skipping {skip_batches} batch(es) from checkpoint.")

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

                    # Calculate losses
                    policy_loss = compute_policy_loss(policy_preds, policy_targets)
                    value_loss = compute_value_loss(value_preds, value_targets)
                    loss = compute_total_loss(policy_loss, value_loss, self.batch_size)

                scaler.scale(loss).backward()
                accumulate_count += 1

                if (accumulate_count % max(256 // self.batch_size, 1) == 0) or (batch_idx == len(train_loader)):
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    accumulate_count = 0

                # Update scheduler if applicable
                if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler._LRScheduler):
                    self.scheduler.step()

                # Accumulate losses
                total_policy_loss += policy_loss.item() * inputs.size(0)
                total_value_loss += value_loss.item() * inputs.size(0)

                # Calculate accuracy
                batch_accuracy = compute_accuracy(policy_preds, policy_targets)
                correct_predictions += batch_accuracy * inputs.size(0)
                total_predictions += inputs.size(0)

                # Update tracking variables
                with threading.Lock():
                    self.total_batches_processed += 1

                # Emit batch metrics and progress updates every 100 batches
                if self.total_batches_processed % 100 == 0:
                    # Emit batch metrics
                    self.batch_loss_update.emit(self.total_batches_processed, {'policy': policy_loss.item(), 'value': value_loss.item()})
                    self.batch_accuracy_update.emit(self.total_batches_processed, batch_accuracy)

                    # Emit progress
                    current_progress = min(int((self.total_batches_processed / (self.epochs * len(train_loader))) * 100), 100)
                    self.progress_update.emit(current_progress)

                    # Estimate and emit time left
                    elapsed_time = time.time() - start_time
                    if self.total_batches_processed > 0:
                        estimated_total_time = (elapsed_time / self.total_batches_processed) * (self.epochs * len(train_loader) - self.total_batches_processed)
                        time_left_str = format_time_left(estimated_total_time)
                        self.time_left_update.emit(time_left_str)
                    else:
                        self.time_left_update.emit("Calculating...")

                # Save checkpoint based on batch interval or iteration
                if self.checkpoint_type == 'batch' and self.checkpoint_manager.should_save(batch_idx=self.total_batches_processed):
                    self._save_checkpoint(epoch)
                elif self.checkpoint_type == 'iteration' and self.checkpoint_manager.should_save(iteration=self.total_batches_processed):
                    self._save_checkpoint(epoch)

                # Clean up to free memory
                del inputs, policy_targets, value_targets, policy_preds, value_preds, loss
                torch.cuda.empty_cache()

            # Calculate metrics
            metrics = {'policy_loss': total_policy_loss / total_predictions if total_predictions > 0 else float('inf'), 'value_loss': total_value_loss / total_predictions if total_predictions > 0 else float('inf'), 'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0}

            self.logger.info(f"Epoch {epoch}: Training Accuracy {metrics['accuracy']*100:.2f}%.")

            return metrics
        except Exception as e:
            self.logger.error(f"Error during training for epoch {epoch}: {str(e)}")
            return {'policy_loss': float('inf'), 'value_loss': float('inf'), 'accuracy': 0.0}

    def _validate_epoch(self, val_loader: DataLoader, epoch: int, device: torch.device, training_accuracy: float) -> Dict[str, float]:
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

                    # Calculate losses
                    policy_loss = compute_policy_loss(policy_preds, policy_targets)
                    value_loss = compute_value_loss(value_preds, value_targets)

                    # Accumulate losses
                    val_policy_loss += policy_loss.item() * inputs.size(0)
                    val_value_loss += value_loss.item() * inputs.size(0)

                    # Calculate accuracy
                    batch_accuracy = compute_accuracy(policy_preds, policy_targets)
                    val_correct_predictions += batch_accuracy * inputs.size(0)
                    val_total_predictions += inputs.size(0)

            # Calculate validation metrics
            if val_total_predictions > 0:
                metrics = {'policy_loss': val_policy_loss / val_total_predictions, 'value_loss': val_value_loss / val_total_predictions, 'accuracy': val_correct_predictions / val_total_predictions}
            else:
                metrics = {'policy_loss': float('inf'), 'value_loss': float('inf'), 'accuracy': 0.0}

            # Emit validation metrics
            self.val_loss_update.emit(epoch, {'policy': metrics['policy_loss'], 'value': metrics['value_loss']})
            self.validation_accuracy_update.emit(epoch, training_accuracy, metrics['accuracy'])

            self.logger.info(f"Epoch {epoch}: Validation Policy Loss {metrics['policy_loss']:.4f}, "f"Value Loss {metrics['value_loss']:.4f}, Accuracy {metrics['accuracy']*100:.2f}%.")

            return metrics

        except Exception as e:
            self.logger.error(f"Error during validation for epoch {epoch}: {str(e)}")
            return {'policy_loss': float('inf'), 'value_loss': float('inf'), 'accuracy': 0.0}