import os, time, h5py, numpy as np, chess, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, threading
from torch.utils.data import DataLoader, Dataset

MOVE_MAPPING = {}
INDEX_MAPPING = {}

def initialize_move_mappings():
    index = 0
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            move = chess.Move(from_sq, to_sq)
            MOVE_MAPPING[index] = move
            INDEX_MAPPING[move] = index
            index += 1
            if chess.square_rank(to_sq) in [0, 7]:
                for promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    move = chess.Move(from_sq, to_sq, promotion=promo)
                    MOVE_MAPPING[index] = move
                    INDEX_MAPPING[move] = index
                    index += 1
    return index

TOTAL_MOVES = initialize_move_mappings()

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 1), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.training:
            return x.mul_(y.expand_as(x))
        else:
            return x * y.expand_as(x)

class SEResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels, reduction)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ChessModel(nn.Module):
    def __init__(self, filters=128, res_blocks=20, num_moves=TOTAL_MOVES):
        super(ChessModel, self).__init__()
        self.num_moves = num_moves
        self.initial_conv = nn.Conv2d(20, filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        layers = []
        for _ in range(res_blocks):
            layers.append(SEResidualUnit(filters, filters))
        self.residual_layers = nn.Sequential(*layers)
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(2 * 8 * 8, self.num_moves)
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc1_bn = nn.BatchNorm1d(256)
        self.value_fc1_relu = nn.ReLU(inplace=True)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.initial_bn(self.initial_conv(x)))
        x = self.residual_layers(x)
        p = self.policy_relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        v = self.value_relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = self.value_fc1_relu(self.value_fc1_bn(self.value_fc1(v)))
        v = torch.tanh(self.value_fc2(v))
        return p, v

class H5Dataset(Dataset):
    def __init__(self, h5_file, indices):
        self.h5_file = h5_file
        self.indices = indices
        self.input_shape = h5_file['inputs'].shape[1:]
        self.policy_shape = h5_file['policy_targets'].shape[1:]
        self.value_shape = h5_file['value_targets'].shape[1:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        try:
            actual_idx = self.indices[idx]
            input_tensor = self.h5_file['inputs'][actual_idx]
            policy_target = self.h5_file['policy_targets'][actual_idx]
            value_target = self.h5_file['value_targets'][actual_idx]
            if input_tensor.shape != self.input_shape:
                raise ValueError(f"Input shape mismatch at index {actual_idx}")
            if policy_target.shape != self.policy_shape:
                raise ValueError(f"Policy target shape mismatch at index {actual_idx}")
            if value_target.shape != self.value_shape:
                raise ValueError(f"Value target shape mismatch at index {actual_idx}")
            input_tensor = torch.from_numpy(input_tensor).float()
            policy_target = torch.tensor(policy_target).long()
            value_target = torch.tensor(value_target).float()
            return input_tensor, policy_target, value_target
        except Exception as e:
            raise RuntimeError(f"Error loading data at index {idx}: {str(e)}")

class ModelTrainer:
    def __init__(self, epochs=3, batch_size=256, lr=0.001, weight_decay=1e-4,
                log_fn=None, progress_fn=None, loss_fn=None, val_loss_fn=None, 
                accuracy_fn=None, stop_event=None, pause_event=None, 
                time_left_fn=None, save_checkpoints=True, checkpoint_interval=1,
                checkpoint_type='epoch', checkpoint_interval_minutes=60,
                checkpoint_batch_interval=1000, dataset_path='data/processed/dataset.h5',
                train_indices_path='data/processed/train_indices.npy',
                val_indices_path='data/processed/val_indices.npy',
                checkpoint_path=None, automatic_batch_size=False,
                batch_loss_fn=None, batch_accuracy_fn=None, lr_fn=None,
                initial_batches_processed_callback=None):
        super().__init__()
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
            if self.log_fn:
                checkpoint_info = f"Checkpoint saved: {checkpoint_name}"
                self.log_fn(checkpoint_info)
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Error saving checkpoint: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise

    def format_time_left(self, seconds):
        days = seconds // 86400
        remainder = seconds % 86400
        hours = remainder // 3600
        minutes = (remainder % 3600) // 60
        secs = remainder % 60
        if days >= 1:
            day_str = f"{int(days)}d " if days > 1 else "1d "
            return f"{day_str}{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
        else:
            return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

    def estimate_max_batch_size(self, model, device):
        if not torch.cuda.is_available():
            if self.log_fn:
                self.log_fn("CUDA not available. Using default batch size 32.")
            return 32
        sample_input = torch.randn(1, 20, 8, 8).to(device)
        model = model.to(device)
        try:
            model.eval()
            with torch.no_grad():
                model(sample_input)
            del sample_input
            torch.cuda.empty_cache()
            max_mem = torch.cuda.get_device_properties(device).total_memory
            allocated_mem = torch.cuda.memory_allocated(device)
            per_sample_mem = allocated_mem
            free_mem = max_mem - allocated_mem
            if per_sample_mem == 0:
                estimated_batch_size = 32
            else:
                estimated_batch_size = int((free_mem // per_sample_mem) * 0.25)
                estimated_batch_size = max(min(estimated_batch_size, 512), 1)
            return estimated_batch_size
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Error estimating batch size: {e}")
            return 32
        finally:
            model.train()
            torch.cuda.empty_cache()

    def train_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.log_fn:
            self.log_fn(f"Using device: {device}")
        required_files = [
            (self.dataset_path, "Dataset file"),
            (self.train_indices_path, "Training indices"),
            (self.val_indices_path, "Validation indices")
        ]
        for file_path, description in required_files:
            if not os.path.exists(file_path):
                if self.log_fn:
                    self.log_fn(f"{description} not found at {file_path}.")
                return
        h5_file = None
        try:
            if self.log_fn:
                self.log_fn("Opening dataset file...")
            h5_file = h5py.File(self.dataset_path, 'r')
            train_indices = np.load(self.train_indices_path)
            val_indices = np.load(self.val_indices_path)
            train_dataset = H5Dataset(h5_file, train_indices)
            val_dataset = H5Dataset(h5_file, val_indices)
            if self.log_fn:
                self.log_fn(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
            model = ChessModel(num_moves=TOTAL_MOVES)
            model.to(device)
            if self.automatic_batch_size:
                self.batch_size = self.estimate_max_batch_size(model, device)
                if self.log_fn:
                    self.log_fn(f"Automatic batch size estimation: Using batch size {self.batch_size}")
            torch.manual_seed(42)
            np.random.seed(42)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'total_batches_processed' in checkpoint:
                    self.total_batches_processed = checkpoint['total_batches_processed']
                if 'epoch' in checkpoint:
                    if 'batch_idx' in checkpoint and checkpoint['batch_idx'] is not None:
                        start_epoch = checkpoint['epoch']
                        self.log_fn(f"Resumed training from epoch {checkpoint['epoch']}, batch {checkpoint['batch_idx']}")
                        skip_batches = checkpoint['batch_idx']
                    else:
                        start_epoch = checkpoint['epoch'] + 1
                        self.log_fn(f"Resumed training from epoch {checkpoint['epoch']}")
                        skip_batches = 0
                else:
                    start_epoch = 1
                    skip_batches = 0
                remaining_epochs = self.epochs - (start_epoch - 1)
                total_steps = remaining_epochs * len(train_loader)
            else:
                start_epoch = 1
                remaining_epochs = self.epochs
                total_steps = self.epochs * len(train_loader)
                skip_batches = 0
            if self.initial_batches_processed_callback:
                self.initial_batches_processed_callback(self.total_batches_processed)
            best_val_loss = float('inf')
            for epoch in range(start_epoch, self.epochs + 1):
                epoch_start_time = time.time()
                if self.stop_event.is_set():
                    break
                if self.log_fn:
                    self.log_fn(f"Epoch {epoch}/{self.epochs} started.")
                model.train()
                train_iterator = iter(train_loader)
                if epoch == start_epoch and skip_batches > 0:
                    if skip_batches >= len(train_loader):
                        self.log_fn(f"Skip batches ({skip_batches}) exceed total batches ({len(train_loader)}). Skipping entire epoch.")
                        skip_batches = 0
                    for _ in range(skip_batches):
                        try:
                            next(train_iterator)
                        except StopIteration:
                            break
                train_metrics = self._train_epoch(model, train_iterator, optimizer, scheduler, epoch, device, total_steps)
                if self.stop_event.is_set():
                    break
                model.eval()
                val_metrics = self._validate_epoch(model, val_loader, epoch, device, train_metrics['accuracy'])
                total_train_loss = train_metrics['policy_loss'] + train_metrics['value_loss']
                total_val_loss = val_metrics['policy_loss'] + val_metrics['value_loss']
                epoch_duration = time.time() - epoch_start_time
                if self.log_fn:
                    self.log_fn(f"Epoch {epoch}/{self.epochs} completed in {self.format_time_left(epoch_duration)} - "
                            f"Training Loss: {total_train_loss:.4f}, "
                            f"Validation Loss: {total_val_loss:.4f}, "
                            f"Training Accuracy: {train_metrics['accuracy']*100:.2f}%, "
                            f"Validation Accuracy: {val_metrics['accuracy']*100:.2f}%")
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    best_model_path = os.path.join('models', 'saved_models', 'best_model.pth')
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict()
                    }, best_model_path)
                    if self.log_fn:
                        self.log_fn(f"Best model updated at epoch {epoch} - "
                                f"Validation Loss: {total_val_loss:.4f}, "
                                f"Training Loss: {total_train_loss:.4f}")
                if self.should_save_checkpoint(epoch, None, len(train_loader)):
                    self.save_checkpoint(model, optimizer, scheduler, epoch)
            if not self.stop_event.is_set():
                model_dir = os.path.join('models', 'saved_models')
                os.makedirs(model_dir, exist_ok=True)
                final_model_path = os.path.join(model_dir, 'final_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': self.epochs,
                    'timestamp': time.strftime('%Y%m%d_%H%M%S'),
                    'total_batches_processed': self.total_batches_processed
                }, final_model_path)
                if self.log_fn:
                    self.log_fn("Training completed and final model saved.")
            else:
                if self.log_fn:
                    self.log_fn("Training stopped by user.")
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Error during training: {str(e)}")
            return
        finally:
            if h5_file:
                h5_file.close()

    def _train_epoch(self, model, train_iterator, optimizer, scheduler, epoch, device, total_steps):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        local_steps = 0
        try:
            start_time = time.time()
            for batch_idx, (inputs, policy_targets, value_targets) in enumerate(train_iterator, 1):
                if self.stop_event.is_set():
                    break
                self.pause_event.wait()
                inputs = inputs.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device)
                optimizer.zero_grad()
                policy_preds, value_preds = model(inputs)
                smoothing = 0.1
                confidence = 1.0 - smoothing
                n_classes = policy_preds.size(1)
                one_hot = torch.zeros_like(policy_preds).scatter(1, policy_targets.unsqueeze(1), 1)
                smoothed_labels = one_hot * confidence + (1 - one_hot) * (smoothing / (n_classes - 1))
                log_probs = F.log_softmax(policy_preds, dim=1)
                policy_loss = -(smoothed_labels * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value_preds.view(-1), value_targets)
                loss = policy_loss + value_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step(epoch + batch_idx / len(train_iterator))
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
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
                    current_lr = scheduler.get_last_lr()[0]
                    self.lr_fn(self.total_batches_processed, current_lr)
                if self.progress_fn:
                    self.progress_fn(current_progress)
                if self.time_left_fn:
                    elapsed_time = time.time() - start_time
                    if local_steps > 0:
                        estimated_total_time = (elapsed_time / local_steps) * (total_steps - self.total_batches_processed)
                        time_left = estimated_total_time
                        self.time_left_fn(self.format_time_left(time_left))
                    else:
                        self.time_left_fn("Calculating...")
                if self.should_save_checkpoint(epoch, self.total_batches_processed, len(train_iterator)):
                    self.save_checkpoint(model, optimizer, scheduler, epoch, batch_idx=self.total_batches_processed)
                del inputs, policy_targets, value_targets, policy_preds, value_preds, loss
                torch.cuda.empty_cache()
            metrics = {
                'policy_loss': total_policy_loss / len(train_iterator),
                'value_loss': total_value_loss / len(train_iterator),
                'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0
            }
            if self.log_fn:
                self.log_fn(f"Epoch {epoch}/{self.epochs}, Training Accuracy: {metrics['accuracy'] * 100:.2f}%")
            if self.loss_fn:
                self.loss_fn(epoch, {'policy': metrics['policy_loss'], 
                                'value': metrics['value_loss']})
            return metrics
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Error during training epoch: {str(e)}")
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
                    if self.stop_event.is_set():
                        break
                    self.pause_event.wait()
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
                self.val_loss_fn(epoch, {'policy': metrics['policy_loss'], 
                                        'value': metrics['value_loss']})
            if self.accuracy_fn:
                self.accuracy_fn(epoch, training_accuracy, metrics['accuracy'])
            if self.log_fn:
                self.log_fn(f"Epoch {epoch}/{self.epochs}, "
                        f"Validation Policy Loss: {metrics['policy_loss']:.4f}, "
                        f"Validation Value Loss: {metrics['value_loss']:.4f}, "
                        f"Validation Accuracy: {metrics['accuracy'] * 100:.2f}%")
            return metrics
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Error during validation: {str(e)}")
            return {
                'policy_loss': float('inf'),
                'value_loss': float('inf'),
                'accuracy': 0.0
            }