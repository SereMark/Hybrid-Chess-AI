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

MOVE_MAPPING = {}
INDEX_MAPPING = {}

def initialize_move_mappings():
    index = 0
    for from_sq in range(64):
        for to_sq in range(64):
            for promo in [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                move = chess.Move(from_sq, to_sq, promotion=promo)
                if promo and chess.square_rank(to_sq) not in [0, 7]:
                    continue
                if chess.square_distance(from_sq, to_sq) == 0:
                    continue
                MOVE_MAPPING[index] = move
                INDEX_MAPPING[move] = index
                index += 1
    return index

TOTAL_MOVES = initialize_move_mappings()

class ResidualUnit(nn.Module):
    def __init__(self, filters):
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ChessModel(nn.Module):
    def __init__(self, filters=256, res_blocks=10, num_moves=4672):
        super(ChessModel, self).__init__()
        self.num_moves = num_moves
        self.initial_conv = nn.Conv2d(20, filters, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.residual_layers = nn.Sequential(*[ResidualUnit(filters) for _ in range(res_blocks)])
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(2 * 8 * 8, self.num_moves)
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1)
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

class ChessTrainingDataset(Dataset):
    def __init__(self, h5_path, indices):
        self.h5_path = h5_path
        self.indices = indices
        self.h5_file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.inputs = self.h5_file['inputs']
            self.policy_targets = self.h5_file['policy_targets']
            self.value_targets = self.h5_file['value_targets']
        real_idx = self.indices[idx]
        input_tensor = torch.tensor(self.inputs[real_idx], dtype=torch.float32)
        policy_target = torch.tensor(self.policy_targets[real_idx], dtype=torch.long)
        value_target = torch.tensor(self.value_targets[real_idx], dtype=torch.float32)
        return input_tensor, policy_target, value_target

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

class ModelTrainer:
    def __init__(self, epochs, batch_size, lr, momentum, weight_decay, log_fn=None, progress_fn=None, loss_fn=None, stop_fn=None, pause_fn=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.log_fn = log_fn
        self.progress_fn = progress_fn
        self.loss_fn = loss_fn
        self.stop_fn = stop_fn or (lambda: False)
        self.pause_fn = pause_fn or (lambda: False)

    def train_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.log_fn:
            self.log_fn(f"Using device: {device}")
        data_dir = os.path.join('data', 'processed')
        h5_path = os.path.join(data_dir, 'dataset.h5')
        train_idx_path = os.path.join(data_dir, 'train_indices.npy')
        val_idx_path = os.path.join(data_dir, 'val_indices.npy')
        if not os.path.exists(h5_path):
            if self.log_fn:
                self.log_fn(f"Dataset file not found at {h5_path}. Please run data preparation first.")
            return
        if not os.path.exists(train_idx_path) or not os.path.exists(val_idx_path):
            if self.log_fn:
                self.log_fn("Training or validation indices not found. Please run data preparation first.")
            return
        train_indices = np.load(train_idx_path)
        val_indices = np.load(val_idx_path)
        train_dataset = ChessTrainingDataset(h5_path, train_indices)
        val_dataset = ChessTrainingDataset(h5_path, val_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        total_steps = self.epochs * len(train_loader)
        steps_done = 0
        model = ChessModel(num_moves=TOTAL_MOVES)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        best_val_loss = float('inf')
        for epoch in range(1, self.epochs + 1):
            if self.stop_fn():
                if self.log_fn:
                    self.log_fn("Training stopped by user.")
                break
            while self.pause_fn():
                if self.log_fn:
                    self.log_fn("Training is paused. Waiting to resume...")
                time.sleep(0.5)
            if self.log_fn:
                self.log_fn(f"Epoch {epoch}/{self.epochs} started.")
            model.train()
            total_policy_loss = 0.0
            total_value_loss = 0.0
            for batch_idx, (inputs, policy_targets, value_targets) in enumerate(train_loader, 1):
                if self.stop_fn():
                    if self.log_fn:
                        self.log_fn("Training stopped by user.")
                    break
                while self.pause_fn():
                    if self.log_fn:
                        self.log_fn("Training is paused. Waiting to resume...")
                    time.sleep(0.5)
                inputs = inputs.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device)
                optimizer.zero_grad()
                policy_preds, value_preds = model(inputs)
                policy_loss = F.cross_entropy(policy_preds, policy_targets)
                value_loss = F.mse_loss(value_preds.view(-1), value_targets)
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                steps_done += 1
                progress = min(int((steps_done / total_steps) * 100), 100)
                if self.progress_fn:
                    self.progress_fn(progress)
                if self.loss_fn:
                    self.loss_fn(policy_loss.item(), value_loss.item())
                if batch_idx % 100 == 0:
                    if self.log_fn:
                        self.log_fn(f"Epoch {epoch}/{self.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                                    f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
            scheduler.step()
            model.eval()
            val_policy_loss = 0.0
            val_value_loss = 0.0
            with torch.no_grad():
                for inputs, policy_targets, value_targets in val_loader:
                    if self.stop_fn():
                        if self.log_fn:
                            self.log_fn("Training stopped by user during validation.")
                        break
                    inputs = inputs.to(device)
                    policy_targets = policy_targets.to(device)
                    value_targets = value_targets.to(device)
                    policy_preds, value_preds = model(inputs)
                    policy_loss = F.cross_entropy(policy_preds, policy_targets)
                    value_loss = F.mse_loss(value_preds.view(-1), value_targets)
                    val_policy_loss += policy_loss.item()
                    val_value_loss += value_loss.item()
            avg_val_policy = val_policy_loss / len(val_loader)
            avg_val_value = val_value_loss / len(val_loader)
            total_val = avg_val_policy + avg_val_value
            if self.log_fn:
                self.log_fn(f"Epoch {epoch}/{self.epochs}, Validation Policy Loss: {avg_val_policy:.4f}, "
                            f"Validation Value Loss: {avg_val_value:.4f}")
            if total_val < best_val_loss:
                best_val_loss = total_val
                checkpoint_dir = os.path.join('models', 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                if self.log_fn:
                    self.log_fn(f"Model saved at epoch {epoch} with validation loss {best_val_loss:.4f}")
        model_dir = os.path.join('models', 'saved_models')
        os.makedirs(model_dir, exist_ok=True)
        final_model_path = os.path.join(model_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        if self.log_fn:
            self.log_fn("Training completed and final model saved.")