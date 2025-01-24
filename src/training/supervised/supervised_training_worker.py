import os
import time
from typing import Optional
import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from models.tempname import TransformerChessModel
from src.utils.datasets import H5Dataset
from src.utils.common_utils import format_time_left
from src.utils.train_utils import initialize_optimizer, initialize_scheduler, initialize_random_seeds, validate_epoch, train_epoch
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
        self.model = TransformerChessModel(get_total_moves()).to(self.device)

        # Initialize optimizer
        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay, logger=self.logger)

        # Initialize scheduler
        total_steps = self.epochs * len(DataLoader(H5Dataset(self.dataset_path, np.load(self.train_indices_path)), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True))
        self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, total_steps=total_steps, logger=self.logger)

        # Mixed Precision Training
        self.scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize checkpoint manager
        self.checkpoint_dir = os.path.join('models', 'checkpoints', 'supervised')
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir, checkpoint_type=self.checkpoint_type, checkpoint_interval=self.checkpoint_interval, logger=self.logger)

        # Tracking variables
        self.total_batches_processed = 0

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

            # For time-left estimation
            total_steps = self.epochs * len(train_loader)
            start_time = time.time()

            # Training loop
            for epoch in range(start_epoch, self.epochs + 1):
                if self._is_stopped.is_set():
                    break

                epoch_start_time = time.time()
                self.logger.info(f"Beginning epoch {epoch}/{self.epochs}.")

                # Train for one epoch
                train_metrics = train_epoch(model=self.model, data_loader=train_loader, device=self.device, scaler=self.scaler, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, 
                                            total_epochs=self.epochs, skip_batches=skip_batches if epoch == start_epoch else 0, accumulation_steps=max(256 // self.batch_size, 1), batch_size=self.batch_size, 
                                            smooth_policy_targets=True, compute_accuracy_flag=True, total_batches_processed=self.total_batches_processed, batch_loss_update_signal=self.batch_loss_update, 
                                            batch_accuracy_update_signal=self.batch_accuracy_update, progress_update_signal=self.progress_update, time_left_update_signal=self.time_left_update, 
                                            checkpoint_manager=self.checkpoint_manager, logger=self.logger, is_stopped_event=self._is_stopped, is_paused_event=self._is_paused, start_time=start_time, total_steps=total_steps)

                # Update total_batches_processed after the epoch
                self.total_batches_processed = train_metrics["total_batches_processed"]
                skip_batches = 0

                if self._is_stopped.is_set():
                    break

                # Validate the current epoch
                self.model.eval()
                val_metrics = validate_epoch(model=self.model, val_loader=val_loader, device=self.device, epoch=epoch, training_accuracy=train_metrics['accuracy'], 
                                             val_loss_update_signal=self.val_loss_update, validation_accuracy_update_signal=self.validation_accuracy_update, logger=self.logger, 
                                             is_stopped_event=self._is_stopped, is_paused_event=self._is_paused, smooth_policy_targets=True)

                # Calculate total losses
                total_train_loss = train_metrics['policy_loss'] + train_metrics['value_loss']
                total_val_loss = val_metrics['policy_loss'] + val_metrics['value_loss']

                # Calculate epoch duration
                epoch_duration = time.time() - epoch_start_time
                self.logger.info(
                    f"Epoch {epoch}/{self.epochs} completed in {format_time_left(epoch_duration)}. "
                    f"Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']*100:.2f}%, "
                    f"Val Acc: {val_metrics['accuracy']*100:.2f}%."
                )

                # Emit epoch accuracy update
                self.validation_accuracy_update.emit(epoch, train_metrics['accuracy'], val_metrics['accuracy'])

                # Let the checkpoint manager handle saving if required (epoch-based)
                if self.save_checkpoints:
                    self.checkpoint_manager.save(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, batch_idx=self.total_batches_processed, iteration=None, training_stats={})

            # Save the final model if training completed without interruption
            if not self._is_stopped.is_set():
                self.checkpoint_manager.save_final_model(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, batch_idx=self.total_batches_processed, training_stats={},
                                                         final_path=os.path.join("models", "saved_models", "pre_trained_model.pth"))

        except Exception as e:
            self.logger.error(f"Error in SupervisedWorker: {str(e)}")
        finally:
            self.task_finished.emit()
            self.finished.emit()