import os
import time
import h5py
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import threading

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
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        input_tensor = torch.from_numpy(self.h5_file['inputs'][actual_idx]).float()
        policy_target = torch.tensor(self.h5_file['policy_targets'][actual_idx]).long()
        value_target = torch.tensor(self.h5_file['value_targets'][actual_idx]).float()
        return input_tensor, policy_target, value_target

class ModelTrainer:
    def __init__(self, epochs=3, batch_size=256, lr=0.001, weight_decay=1e-4,
                 log_fn=None, progress_fn=None, loss_fn=None, val_loss_fn=None, accuracy_fn=None,
                 stop_event=None, pause_event=None, time_left_fn=None,
                 save_checkpoints=True, checkpoint_interval=1,
                 dataset_path='data/processed/dataset.h5',
                 train_indices_path='data/processed/train_indices.npy',
                 val_indices_path='data/processed/val_indices.npy',
                 checkpoint_path=None,
                 automatic_batch_size=False,
                 batch_loss_fn=None, batch_accuracy_fn=None, lr_fn=None):
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
        self.time_left_fn = time_left_fn or (lambda time_left_str: None)
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

    def estimate_max_batch_size(self, model, device):
        sample_input = torch.randn(1, 20, 8, 8).to(device)
        try:
            model.to(device)
            model.eval()
            with torch.no_grad():
                model(sample_input)
            torch.cuda.empty_cache()
            max_mem = torch.cuda.get_device_properties(device).total_memory
            allocated_mem = torch.cuda.memory_allocated(device)
            per_sample_mem = allocated_mem
            free_mem = max_mem - allocated_mem
            estimated_batch_size = int(free_mem // per_sample_mem)
            estimated_batch_size = max(estimated_batch_size // 2, 1)
            return estimated_batch_size
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Error estimating batch size: {e}")
            return 1
        finally:
            model.train()

    def train_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.log_fn:
            self.log_fn(f"Using device: {device}")

        h5_path = self.dataset_path
        train_idx_path = self.train_indices_path
        val_idx_path = self.val_indices_path

        if not os.path.exists(h5_path):
            if self.log_fn:
                self.log_fn(f"Dataset file not found at {h5_path}. Please run data preparation first.")
            return
        if not os.path.exists(train_idx_path) or not os.path.exists(val_idx_path):
            if self.log_fn:
                self.log_fn("Training or validation indices not found. Please run data preparation first.")
            return

        if self.log_fn:
            self.log_fn("Opening dataset file...")

        h5_file = h5py.File(h5_path, 'r')

        try:
            train_indices = np.load(train_idx_path)
            val_indices = np.load(val_idx_path)

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

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            total_steps = self.epochs * len(train_loader)
            steps_done = 0
            start_time = time.time()

            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            best_val_loss = float('inf')
            start_epoch = 1

            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                if self.log_fn:
                    self.log_fn(f"Resumed training from epoch {start_epoch}")

            for epoch in range(start_epoch, self.epochs + 1):
                if self.stop_event.is_set():
                    break
                self.pause_event.wait()
                if self.log_fn:
                    self.log_fn(f"Epoch {epoch}/{self.epochs} started.")
                model.train()
                total_policy_loss = 0.0
                total_value_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                for batch_idx, (inputs, policy_targets, value_targets) in enumerate(train_loader, 1):
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
                    scheduler.step(epoch + batch_idx / len(train_loader))

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    steps_done += 1
                    _, predicted = torch.max(policy_preds.data, 1)
                    total_predictions += policy_targets.size(0)
                    correct_predictions += (predicted == policy_targets).sum().item()

                    batch_accuracy = (predicted == policy_targets).sum().item() / policy_targets.size(0)
                    if self.batch_loss_fn:
                        self.batch_loss_fn(steps_done, {'policy': policy_loss.item(), 'value': value_loss.item()})
                    if self.batch_accuracy_fn:
                        self.batch_accuracy_fn(steps_done, batch_accuracy)
                    if self.lr_fn:
                        current_lr = scheduler.get_last_lr()[0]
                        self.lr_fn(steps_done, current_lr)

                    progress = min(int((steps_done / total_steps) * 100), 100)
                    elapsed_time = time.time() - start_time
                    if self.progress_fn and batch_idx % 10 == 0:
                        self.progress_fn(progress)
                    if self.time_left_fn and batch_idx % 10 == 0:
                        estimated_total_time = (elapsed_time / steps_done) * total_steps
                        time_left = estimated_total_time - elapsed_time
                        time_left_str = time.strftime('%H:%M:%S', time.gmtime(time_left))
                        self.time_left_fn(time_left_str)
                    if batch_idx % 100 == 0 and self.log_fn:
                        avg_policy_loss = total_policy_loss / batch_idx
                        avg_value_loss = total_value_loss / batch_idx
                        self.log_fn(f"Epoch {epoch}/{self.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                                    f"Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}")
                if total_predictions > 0:
                    training_accuracy = correct_predictions / total_predictions
                else:
                    training_accuracy = 0.0
                if self.log_fn:
                    self.log_fn(f"Epoch {epoch}/{self.epochs}, Training Accuracy: {training_accuracy * 100:.2f}%")
                if self.loss_fn:
                    avg_policy_loss = total_policy_loss / len(train_loader)
                    avg_value_loss = total_value_loss / len(train_loader)
                    self.loss_fn(epoch, {'policy': avg_policy_loss, 'value': avg_value_loss})
                if self.stop_event.is_set():
                    break
                model.eval()
                val_policy_loss = 0.0
                val_value_loss = 0.0
                val_correct_predictions = 0
                val_total_predictions = 0
                val_batches = 0
                with torch.no_grad():
                    for inputs, policy_targets, value_targets in val_loader:
                        if self.stop_event.is_set():
                            break
                        self.pause_event.wait()
                        inputs = inputs.to(device)
                        policy_targets = policy_targets.to(device)
                        value_targets = value_targets.to(device)
                        policy_preds, value_preds = model(inputs)
                        policy_loss = F.cross_entropy(policy_preds, policy_targets)
                        value_loss = F.mse_loss(value_preds.view(-1), value_targets)
                        val_policy_loss += policy_loss.item()
                        val_value_loss += value_loss.item()
                        _, predicted = torch.max(policy_preds.data, 1)
                        val_total_predictions += policy_targets.size(0)
                        val_correct_predictions += (predicted == policy_targets).sum().item()
                        val_batches += 1
                if val_batches > 0:
                    avg_val_policy = val_policy_loss / val_batches
                    avg_val_value = val_value_loss / val_batches
                    total_val_loss = avg_val_policy + avg_val_value
                else:
                    avg_val_policy = avg_val_value = total_val_loss = 0.0
                if val_total_predictions > 0:
                    val_accuracy = val_correct_predictions / val_total_predictions
                else:
                    val_accuracy = 0.0
                if self.val_loss_fn:
                    self.val_loss_fn(epoch, {'policy': avg_val_policy, 'value': avg_val_value})
                if self.accuracy_fn:
                    self.accuracy_fn(epoch, training_accuracy, val_accuracy)
                if self.log_fn:
                    self.log_fn(f"Epoch {epoch}/{self.epochs}, Validation Policy Loss: {avg_val_policy:.4f}, "
                                f"Validation Value Loss: {avg_val_value:.4f}")
                    self.log_fn(f"Epoch {epoch}/{self.epochs}, Validation Accuracy: {val_accuracy * 100:.2f}%")
                if self.save_checkpoints and epoch % self.checkpoint_interval == 0:
                    checkpoint_dir = os.path.join('models', 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, checkpoint_path)
                    if self.log_fn:
                        self.log_fn(f"Model checkpoint saved at epoch {epoch}.")
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    best_model_path = os.path.join('models', 'saved_models', 'best_model.pth')
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_path)
                    if self.log_fn:
                        self.log_fn(f"Best model updated at epoch {epoch} with validation loss {best_val_loss:.4f}")
            model_dir = os.path.join('models', 'saved_models')
            os.makedirs(model_dir, exist_ok=True)
            final_model_path = os.path.join(model_dir, 'final_model.pth')
            torch.save(model.state_dict(), final_model_path)
            if self.log_fn:
                self.log_fn("Training completed and final model saved.")
            if self.stop_event.is_set() and self.log_fn:
                self.log_fn("Training stopped by user.")

        finally:
            h5_file.close()