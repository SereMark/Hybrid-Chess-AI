import os, threading, time, numpy as np, torch, torch.optim as optim, torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from src.models.model import ChessModel
from src.utils.datasets import H5Dataset
from src.utils.chess_utils import TOTAL_MOVES, estimate_batch_size
from src.utils.common_utils import format_time_left, log_message, should_stop, wait_if_paused, initialize_random_seeds

class SupervisedTrainer:
    def __init__(
        self,
        epochs=3,
        batch_size=256,
        lr=0.001,
        weight_decay=1e-4,
        log_fn=None,
        progress_fn=None,
        loss_fn=None,
        val_loss_fn=None,
        accuracy_fn=None,
        stop_event=None,
        pause_event=None,
        time_left_fn=None,
        save_checkpoints=True,
        checkpoint_interval=1,
        checkpoint_type='epoch',
        checkpoint_interval_minutes=60,
        checkpoint_batch_interval=1000,
        dataset_path='data/processed/dataset.h5',
        train_indices_path='data/processed/train_indices.npy',
        val_indices_path='data/processed/val_indices.npy',
        checkpoint_path=None,
        automatic_batch_size=False,
        batch_loss_fn=None,
        batch_accuracy_fn=None,
        lr_fn=None,
        initial_batches_processed_callback=None,
        optimizer_type='adamw',
        scheduler_type='cosineannealingwarmrestarts',
        output_model_path='models/saved_models/pre_trained_model.pth',
        num_workers=4,
        random_seed=42
    ):
        self._lock = threading.Lock()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.log_fn = log_fn
        self.progress_fn = progress_fn
        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn
        self.accuracy_fn = accuracy_fn
        self.stop_event = stop_event or threading.Event()
        self.pause_event = pause_event or threading.Event()
        self.pause_event.set()
        self.time_left_fn = time_left_fn
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.dataset_path = dataset_path
        self.train_indices_path = train_indices_path
        self.val_indices_path = val_indices_path
        self.checkpoint_path = checkpoint_path
        self.automatic_batch_size = automatic_batch_size
        self.batch_loss_fn = batch_loss_fn
        self.batch_accuracy_fn = batch_accuracy_fn
        self.lr_fn = lr_fn
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_batch_interval = checkpoint_batch_interval
        self.last_checkpoint_time = time.time()
        self.initial_batches_processed_callback = initial_batches_processed_callback
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.output_model_path = output_model_path
        self.num_workers = num_workers
        self.random_seed = random_seed
        with self._lock:
            self.total_batches_processed = 0

    def should_save_checkpoint(self, epoch, batch_idx, total_batches):
        with self._lock:
            if not self.save_checkpoints:
                return False
            if self.checkpoint_type == 'epoch':
                if batch_idx is None:
                    return epoch % self.checkpoint_interval == 0
                return False
            elif self.checkpoint_type == 'time':
                current_time = time.time()
                elapsed_minutes = (current_time - self.last_checkpoint_time) / 60
                if elapsed_minutes >= self.checkpoint_interval_minutes:
                    self.last_checkpoint_time = current_time
                    return True
                return False
            elif self.checkpoint_type == 'batch':
                if batch_idx is not None:
                    return self.total_batches_processed % self.checkpoint_batch_interval == 0
                return False
            return False

    def save_checkpoint(self, model, optimizer, scheduler, epoch, batch_idx=None):
        checkpoint_dir = os.path.join('models', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if self.checkpoint_type == 'epoch':
            checkpoint_name = f'checkpoint_epoch_{epoch}_{timestamp}.pth'
        elif self.checkpoint_type == 'time':
            checkpoint_name = f'checkpoint_time_{timestamp}.pth'
        else:
            checkpoint_name = f'checkpoint_epoch_{epoch}_batch_{batch_idx}_{timestamp}.pth'
        temp_path = os.path.join(checkpoint_dir, f'.temp_{checkpoint_name}')
        final_path = os.path.join(checkpoint_dir, checkpoint_name)
        try:
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'batch_idx': batch_idx,
                'total_batches_processed': self.total_batches_processed
            }
            torch.save(checkpoint_data, temp_path)
            os.replace(temp_path, final_path)
            log_message(f"Checkpoint saved: {checkpoint_name}", self.log_fn)
        except Exception as e:
            log_message(f"Error saving checkpoint: {str(e)}", self.log_fn)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise

    def train_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_message(f"Using device: {device}", self.log_fn)
        required_files = [
            (self.dataset_path, "Dataset file"),
            (self.train_indices_path, "Training indices"),
            (self.val_indices_path, "Validation indices")
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
            log_message(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}", self.log_fn)
            model = ChessModel(num_moves=TOTAL_MOVES)
            model.to(device)
            if self.automatic_batch_size:
                self.batch_size = estimate_batch_size(model, device)
                log_message(f"Automatic batch size estimation: Using batch size {self.batch_size}", self.log_fn)

            initialize_random_seeds(self.random_seed)

            num_workers = self.num_workers
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

            if self.optimizer_type == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            elif self.optimizer_type == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
            else:
                log_message(f"Unsupported optimizer type: {self.optimizer_type}. Using AdamW by default.", self.log_fn)
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

            total_steps = self.epochs * len(train_loader)
            if self.scheduler_type == 'cosineannealingwarmrestarts':
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            elif self.scheduler_type == 'steplr':
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            elif self.scheduler_type == 'onecyclelr':
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=total_steps)
            else:
                log_message(f"Unsupported scheduler type: {self.scheduler_type}. Using CosineAnnealingWarmRestarts by default.", self.log_fn)
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

            start_epoch = 1
            skip_batches = 0
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'total_batches_processed' in checkpoint:
                    self.total_batches_processed = checkpoint['total_batches_processed']
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                    skip_batches = checkpoint.get('batch_idx', 0)
                    log_message(f"Resumed training from epoch {start_epoch}, batch {skip_batches}", self.log_fn)
                else:
                    log_message("No epoch information found in checkpoint. Starting from epoch 1.", self.log_fn)
            else:
                log_message("No checkpoint found. Starting training from scratch.", self.log_fn)

            remaining_epochs = self.epochs - (start_epoch - 1)
            total_steps = remaining_epochs * len(train_loader)
            if self.initial_batches_processed_callback:
                self.initial_batches_processed_callback(self.total_batches_processed)
            best_val_loss = float('inf')
            early_stopping_patience = 5
            no_improvement_epochs = 0

            scaler = GradScaler()
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
                        log_message(f"Skip batches ({skip_batches}) exceed total batches ({len(train_loader)}). Skipping entire epoch.", self.log_fn)
                        continue
                    for _ in range(skip_batches):
                        try:
                            next(train_iterator)
                        except StopIteration:
                            break
                train_metrics = self._train_epoch(model, train_iterator, optimizer, scheduler, epoch, device, total_steps, train_loader, scaler, accumulation_steps)
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
                    self.log_fn
                )
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    no_improvement_epochs = 0
                    best_model_checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                    }
                    torch.save(best_model_checkpoint, self.output_model_path)
                    log_message(
                        f"Best model updated at epoch {epoch} - "
                        f"Validation Loss: {total_val_loss:.4f}, "
                        f"Training Loss: {total_train_loss:.4f}",
                        self.log_fn
                    )
                else:
                    no_improvement_epochs += 1
                    if no_improvement_epochs >= early_stopping_patience:
                        log_message("Early stopping triggered.", self.log_fn)
                        break
                if self.should_save_checkpoint(epoch, None, len(train_loader)):
                    self.save_checkpoint(model, optimizer, scheduler, epoch)
                if isinstance(scheduler, optim.lr_scheduler.StepLR):
                    scheduler.step()
            if not should_stop(self.stop_event):
                log_message("Training completed successfully.", self.log_fn)
            else:
                log_message("Training stopped by user.", self.log_fn)
        except Exception as e:
            log_message(f"Error during training: {str(e)}", self.log_fn)
            raise e

    def _train_epoch(self, model, train_iterator, optimizer, scheduler, epoch, device, total_steps, train_loader, scaler, accumulation_steps):
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
                with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
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
                    self.batch_loss_fn(self.total_batches_processed, {
                        'policy': policy_loss.item(),
                        'value': value_loss.item()
                    })
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
                if self.should_save_checkpoint(epoch, self.total_batches_processed, len(train_loader)):
                    self.save_checkpoint(model, optimizer, scheduler, epoch, batch_idx=self.total_batches_processed)
                del inputs, policy_targets, value_targets, policy_preds, value_preds, loss
            metrics = {
                'policy_loss': total_policy_loss / total_predictions,
                'value_loss': total_value_loss / total_predictions,
                'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0
            }
            log_message(f"Epoch {epoch}/{self.epochs}, Training Accuracy: {metrics['accuracy'] * 100:.2f}%", self.log_fn)
            if self.loss_fn:
                self.loss_fn(epoch, {
                    'policy': metrics['policy_loss'],
                    'value': metrics['value_loss']
                })
            return metrics
        except Exception as e:
            log_message(f"Error during training epoch: {str(e)}", self.log_fn)
            return {
                'policy_loss': float('inf'),
                'value_loss': float('inf'),
                'accuracy': 0.0
            }
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
            if val_batches > 0:
                metrics = {
                    'policy_loss': val_policy_loss / val_total_predictions,
                    'value_loss': val_value_loss / val_total_predictions,
                    'accuracy': val_correct_predictions / val_total_predictions
                }
            else:
                metrics = {
                    'policy_loss': float('inf'),
                    'value_loss': float('inf'),
                    'accuracy': 0.0
                }
            if self.val_loss_fn:
                self.val_loss_fn(epoch, {
                    'policy': metrics['policy_loss'],
                    'value': metrics['value_loss']
                })
            if self.accuracy_fn:
                self.accuracy_fn(epoch, training_accuracy, metrics['accuracy'])
            log_message(
                f"Epoch {epoch}/{self.epochs}, "
                f"Validation Policy Loss: {metrics['policy_loss']:.4f}, "
                f"Validation Value Loss: {metrics['value_loss']:.4f}, "
                f"Validation Accuracy: {metrics['accuracy'] * 100:.2f}%",
                self.log_fn
            )
            return metrics
        except Exception as e:
            log_message(f"Error during validation: {str(e)}", self.log_fn)
            return {
                'policy_loss': float('inf'),
                'value_loss': float('inf'),
                'accuracy': 0.0
            }