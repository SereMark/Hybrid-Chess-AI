from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, QHBoxLayout, QLabel, QCheckBox, QComboBox, QProgressBar, QTextEdit, QFileDialog, QMessageBox, QSizePolicy
import threading
import torch
import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess
import time
from src.gui.visualizations.supervised_training_visualization import SupervisedTrainingVisualization

class SupervisedTrainingWorker(QObject):
    log_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    batch_loss_update = pyqtSignal(int, dict)
    batch_accuracy_update = pyqtSignal(int, float)
    learning_rate_update = pyqtSignal(int, float)
    epoch_loss_update = pyqtSignal(int, dict)
    epoch_accuracy_update = pyqtSignal(int, float, float)
    initial_batches_processed = pyqtSignal(int)
    val_loss_update = pyqtSignal(int, dict)
    training_finished = pyqtSignal()
    time_left_update = pyqtSignal(str)

    def __init__(self, epochs, batch_size, learning_rate, weight_decay, save_checkpoints,
                 checkpoint_interval, dataset_path, train_indices_path, val_indices_path, checkpoint_path=None,
                 automatic_batch_size=False, checkpoint_type='epoch',
                 checkpoint_interval_minutes=60,
                 checkpoint_batch_interval=1000):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.dataset_path = dataset_path
        self.train_indices_path = train_indices_path
        self.val_indices_path = val_indices_path
        self.checkpoint_path = checkpoint_path
        self.automatic_batch_size = automatic_batch_size
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_batch_interval = checkpoint_batch_interval
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()

    def run(self):
        try:
            self.log_update.emit("Initializing training...")
            trainer = ModelTrainer(
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                log_fn=self.log_update.emit,
                progress_fn=self.progress_update.emit,
                loss_fn=self.epoch_loss_update.emit,
                val_loss_fn=self.val_loss_update.emit,
                accuracy_fn=self.epoch_accuracy_update.emit,
                stop_event=self._stop_event,
                pause_event=self._pause_event,
                time_left_fn=self.time_left_update.emit,
                save_checkpoints=self.save_checkpoints,
                checkpoint_interval=self.checkpoint_interval,
                checkpoint_type=self.checkpoint_type,
                checkpoint_interval_minutes=self.checkpoint_interval_minutes,
                checkpoint_batch_interval=self.checkpoint_batch_interval,
                dataset_path=self.dataset_path,
                train_indices_path=self.train_indices_path,
                val_indices_path=self.val_indices_path,
                checkpoint_path=self.checkpoint_path,
                automatic_batch_size=self.automatic_batch_size,
                batch_loss_fn=self.batch_loss_update.emit,
                batch_accuracy_fn=self.batch_accuracy_update.emit,
                lr_fn=self.learning_rate_update.emit,
                initial_batches_processed_callback=self.initial_batches_processed.emit
            )
            self.log_update.emit("Starting training...")
            trainer.train_model()
            if not self._stop_event.is_set():
                self.log_update.emit("Training completed successfully.")
            else:
                self.log_update.emit("Training stopped by user request.")
        except Exception as e:
            import traceback
            error_msg = f"Error during training: {str(e)}\n{traceback.format_exc()}"
            self.log_update.emit(error_msg)
        finally:
            self.training_finished.emit()

    def pause(self):
        self._pause_event.clear()
        self.log_update.emit("Training paused by user.")

    def resume(self):
        self._pause_event.set()
        self.log_update.emit("Training resumed by user.")

    def stop(self):
        self._stop_event.set()
        self.log_update.emit("Training stopped by user.")


class SupervisedTrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.worker = None
        self.visualization = SupervisedTrainingVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        settings_group = QGroupBox("Training Settings")
        settings_layout = QFormLayout()
        self.dataset_input = QLineEdit("data/processed/dataset.h5")
        self.train_indices_input = QLineEdit("data/processed/train_indices.npy")
        self.val_indices_input = QLineEdit("data/processed/val_indices.npy")
        self.epochs_input = QLineEdit("25")
        self.batch_size_input = QLineEdit("128")
        self.learning_rate_input = QLineEdit("0.0005")
        self.weight_decay_input = QLineEdit("2e-4")
        dataset_browse_button = QPushButton("Browse")
        dataset_browse_button.clicked.connect(self.browse_dataset)
        train_indices_browse_button = QPushButton("Browse")
        train_indices_browse_button.clicked.connect(self.browse_train_indices)
        val_indices_browse_button = QPushButton("Browse")
        val_indices_browse_button.clicked.connect(self.browse_val_indices)
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.dataset_input)
        dataset_layout.addWidget(dataset_browse_button)
        train_indices_layout = QHBoxLayout()
        train_indices_layout.addWidget(self.train_indices_input)
        train_indices_layout.addWidget(train_indices_browse_button)
        val_indices_layout = QHBoxLayout()
        val_indices_layout.addWidget(self.val_indices_input)
        val_indices_layout.addWidget(val_indices_browse_button)
        settings_layout.addRow("Dataset Path:", dataset_layout)
        settings_layout.addRow("Train Indices Path:", train_indices_layout)
        settings_layout.addRow("Validation Indices Path:", val_indices_layout)
        settings_layout.addRow("Epochs:", self.epochs_input)
        settings_layout.addRow("Batch Size:", self.batch_size_input)
        settings_layout.addRow("Learning Rate:", self.learning_rate_input)
        settings_layout.addRow("Weight Decay:", self.weight_decay_input)
        settings_group.setLayout(settings_layout)
        checkpoint_group = QGroupBox("Checkpoint Settings")
        checkpoint_layout = QFormLayout()
        self.save_checkpoints_checkbox = QCheckBox("Enable Checkpoints")
        self.save_checkpoints_checkbox.setChecked(True)
        self.save_checkpoints_checkbox.stateChanged.connect(self.on_checkpoint_enabled_changed)
        checkpoint_layout.addRow(self.save_checkpoints_checkbox)
        checkpoint_type_layout = QHBoxLayout()
        self.checkpoint_type_combo = QComboBox()
        self.checkpoint_type_combo.addItems(['Epoch', 'Time', 'Batch'])
        self.checkpoint_type_combo.currentTextChanged.connect(self.on_checkpoint_type_changed)
        checkpoint_type_layout.addWidget(QLabel("Save checkpoint by:"))
        checkpoint_type_layout.addWidget(self.checkpoint_type_combo)
        checkpoint_type_layout.addStretch()
        checkpoint_layout.addRow(checkpoint_type_layout)
        interval_group = QGroupBox("Checkpoint Interval")
        interval_layout = QVBoxLayout()
        epoch_layout = QHBoxLayout()
        self.checkpoint_interval_input = QLineEdit("1")
        epoch_layout.addWidget(QLabel("Every"))
        epoch_layout.addWidget(self.checkpoint_interval_input)
        epoch_layout.addWidget(QLabel("epochs"))
        epoch_layout.addStretch()
        self.epoch_interval_widget = QWidget()
        self.epoch_interval_widget.setLayout(epoch_layout)
        interval_layout.addWidget(self.epoch_interval_widget)
        time_layout = QHBoxLayout()
        self.checkpoint_interval_minutes_input = QLineEdit("30")
        time_layout.addWidget(QLabel("Every"))
        time_layout.addWidget(self.checkpoint_interval_minutes_input)
        time_layout.addWidget(QLabel("minutes"))
        time_layout.addStretch()
        self.time_interval_widget = QWidget()
        self.time_interval_widget.setLayout(time_layout)
        interval_layout.addWidget(self.time_interval_widget)
        batch_layout = QHBoxLayout()
        self.checkpoint_batch_interval_input = QLineEdit("2000")
        batch_layout.addWidget(QLabel("Every"))
        batch_layout.addWidget(self.checkpoint_batch_interval_input)
        batch_layout.addWidget(QLabel("batches"))
        batch_layout.addStretch()
        self.batch_interval_widget = QWidget()
        self.batch_interval_widget.setLayout(batch_layout)
        interval_layout.addWidget(self.batch_interval_widget)
        interval_group.setLayout(interval_layout)
        checkpoint_layout.addRow(interval_group)
        self.checkpoint_path_input = QLineEdit("")
        checkpoint_browse_button = QPushButton("Browse")
        checkpoint_browse_button.clicked.connect(self.browse_checkpoint)
        checkpoint_path_layout = QHBoxLayout()
        checkpoint_path_layout.addWidget(self.checkpoint_path_input)
        checkpoint_path_layout.addWidget(checkpoint_browse_button)
        checkpoint_layout.addRow("Resume from checkpoint:", checkpoint_path_layout)
        checkpoint_group.setLayout(checkpoint_layout)
        self.on_checkpoint_type_changed(self.checkpoint_type_combo.currentText())
        self.automatic_batch_size_checkbox = QCheckBox("Automatic Batch Size")
        self.automatic_batch_size_checkbox.setChecked(True)
        self.automatic_batch_size_checkbox.toggled.connect(self.toggle_batch_size_input)
        control_buttons_layout = self.create_buttons_layout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label = QLabel("Time Left: Calculating...")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        main_layout.addWidget(settings_group)
        main_layout.addWidget(self.automatic_batch_size_checkbox)
        main_layout.addWidget(checkpoint_group)
        main_layout.addLayout(control_buttons_layout)
        main_layout.addWidget(self.progress_bar)
        self.remaining_time_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.remaining_time_label)
        main_layout.addWidget(self.log_text_edit)
        main_layout.addStretch(1)
        main_layout.addWidget(self.create_visualization_group())
        main_layout.addStretch(2)

    def on_checkpoint_enabled_changed(self, state):
        is_enabled = state == Qt.Checked
        self.checkpoint_type_combo.setEnabled(is_enabled)
        self.epoch_interval_widget.setEnabled(is_enabled and self.checkpoint_type_combo.currentText().lower() == 'epoch')
        self.time_interval_widget.setEnabled(is_enabled and self.checkpoint_type_combo.currentText().lower() == 'time')
        self.batch_interval_widget.setEnabled(is_enabled and self.checkpoint_type_combo.currentText().lower() == 'batch')
        self.checkpoint_path_input.setEnabled(is_enabled)

    def on_checkpoint_type_changed(self, text):
        text = text.lower()
        self.epoch_interval_widget.setVisible(text == 'epoch')
        self.time_interval_widget.setVisible(text == 'time')
        self.batch_interval_widget.setVisible(text == 'batch')
        self.epoch_interval_widget.setEnabled(text == 'epoch' and self.save_checkpoints_checkbox.isChecked())
        self.time_interval_widget.setEnabled(text == 'time' and self.save_checkpoints_checkbox.isChecked())
        self.batch_interval_widget.setEnabled(text == 'batch' and self.save_checkpoints_checkbox.isChecked())

    def create_buttons_layout(self):
        layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.pause_button = QPushButton("Pause Training")
        self.resume_button = QPushButton("Resume Training")
        self.stop_button = QPushButton("Stop Training")
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.resume_button)
        layout.addWidget(self.stop_button)
        layout.addStretch()
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_training)
        self.pause_button.clicked.connect(self.pause_training)
        self.resume_button.clicked.connect(self.resume_training)
        self.stop_button.clicked.connect(self.stop_training)
        return layout

    def toggle_batch_size_input(self, checked):
        self.batch_size_input.setEnabled(not checked)

    def browse_file(self, line_edit, title, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(self, title, line_edit.text(), file_filter)
        if file_path:
            line_edit.setText(file_path)

    def browse_dataset(self):
        self.browse_file(self.dataset_input, "Select Dataset File", "HDF5 Files (*.h5 *.hdf5)")

    def browse_train_indices(self):
        self.browse_file(self.train_indices_input, "Select Train Indices File", "NumPy Files (*.npy)")

    def browse_val_indices(self):
        self.browse_file(self.val_indices_input, "Select Validation Indices File", "NumPy Files (*.npy)")

    def browse_checkpoint(self):
        self.browse_file(self.checkpoint_path_input, "Select Checkpoint File", "PyTorch Files (*.pth *.pt)")

    def start_training(self):
        try:
            epochs = int(self.epochs_input.text())
            learning_rate = float(self.learning_rate_input.text())
            weight_decay = float(self.weight_decay_input.text())
            if any(v <= 0 for v in [epochs, learning_rate, weight_decay]):
                raise ValueError("Epochs, Learning Rate, and Weight Decay must be positive.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Please enter valid and positive parameters.\n{str(e)}")
            return
        automatic_batch_size = self.automatic_batch_size_checkbox.isChecked()
        if automatic_batch_size:
            batch_size = None
        else:
            try:
                batch_size = int(self.batch_size_input.text())
                if batch_size <= 0:
                    raise ValueError("Batch Size must be a positive integer.")
            except ValueError as e:
                QMessageBox.warning(self, "Input Error", f"Batch Size must be a positive integer.\n{str(e)}")
                return
        dataset_path = self.dataset_input.text()
        train_indices_path = self.train_indices_input.text()
        val_indices_path = self.val_indices_input.text()
        checkpoint_path = self.checkpoint_path_input.text() if self.checkpoint_path_input.text() else None
        if not all(os.path.exists(p) for p in [dataset_path, train_indices_path, val_indices_path]):
            QMessageBox.warning(self, "Error", "Dataset or indices files do not exist.")
            return
        save_checkpoints = self.save_checkpoints_checkbox.isChecked()
        checkpoint_type = self.checkpoint_type_combo.currentText().lower()
        checkpoint_interval = None
        checkpoint_interval_minutes = None
        checkpoint_batch_interval = None
        if checkpoint_type == 'epoch':
            try:
                checkpoint_interval = int(self.checkpoint_interval_input.text())
                if checkpoint_interval <= 0:
                    raise ValueError("Epoch interval must be positive.")
            except ValueError as e:
                QMessageBox.warning(self, "Input Error", str(e))
                return
        elif checkpoint_type == 'time':
            try:
                checkpoint_interval_minutes = int(self.checkpoint_interval_minutes_input.text())
                if checkpoint_interval_minutes <= 0:
                    raise ValueError("Time interval must be positive.")
            except ValueError as e:
                QMessageBox.warning(self, "Input Error", str(e))
                return
        elif checkpoint_type == 'batch':
            try:
                checkpoint_batch_interval = int(self.checkpoint_batch_interval_input.text())
                if checkpoint_batch_interval <= 0:
                    raise ValueError("Batch interval must be positive.")
            except ValueError as e:
                QMessageBox.warning(self, "Input Error", str(e))
                return
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()
        self.thread = QThread()
        self.worker = SupervisedTrainingWorker(
            epochs, batch_size, learning_rate, weight_decay, save_checkpoints,
            checkpoint_interval,
            dataset_path, train_indices_path, val_indices_path, checkpoint_path,
            automatic_batch_size,
            checkpoint_type=checkpoint_type,
            checkpoint_interval_minutes=checkpoint_interval_minutes,
            checkpoint_batch_interval=checkpoint_batch_interval
        )
        self.worker.moveToThread(self.thread)
        self.worker.log_update.connect(self.log_text_edit.append)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.batch_loss_update.connect(self.visualization.update_loss_plots)
        self.worker.batch_accuracy_update.connect(self.visualization.update_accuracy_plot)
        self.worker.learning_rate_update.connect(self.visualization.update_learning_rate)
        self.worker.epoch_loss_update.connect(self.handle_epoch_loss)
        self.worker.val_loss_update.connect(self.handle_val_loss)
        self.worker.epoch_accuracy_update.connect(self.handle_epoch_accuracy)
        self.worker.initial_batches_processed.connect(self.visualization.set_total_batches)
        self.worker.time_left_update.connect(self.update_time_left)
        self.worker.training_finished.connect(self.on_training_finished)
        self.worker.training_finished.connect(self.thread.quit)
        self.worker.training_finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()
        if not self.checkpoint_path_input.text():
            self.visualization.reset_visualization()

    def handle_epoch_loss(self, epoch, losses):
        self.log_text_edit.append(f"Epoch {epoch} Training Losses - Policy: {losses['policy']:.4f}, Value: {losses['value']:.4f}")

    def handle_val_loss(self, epoch, losses):
        self.log_text_edit.append(f"Epoch {epoch} Validation Losses - Policy: {losses['policy']:.4f}, Value: {losses['value']:.4f}")

    def handle_epoch_accuracy(self, epoch, training_accuracy, validation_accuracy):
        self.log_text_edit.append(f"Epoch {epoch} Accuracy - Training: {training_accuracy*100:.2f}%, Validation: {validation_accuracy*100:.2f}%")

    def on_thread_finished(self):
        self.worker = None
        self.thread = None

    def pause_training(self):
        if self.worker:
            self.worker.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)

    def resume_training(self):
        if self.worker:
            self.worker.resume()
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)

    def stop_training(self):
        if self.worker:
            self.worker.stop()

    def closeEvent(self, event):
        try:
            if self.worker and self.thread.isRunning():
                self.worker.stop()
                self.thread.quit()
                if not self.thread.wait(5000):
                    self.thread.terminate()
                    self.thread.wait()
            if self.visualization:
                self.visualization.close()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        finally:
            super().closeEvent(event)

    def on_training_finished(self):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Training Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_text_edit.append("Training process finished.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Progress: {value}%")

    def update_time_left(self, time_left_str):
        self.remaining_time_label.setText(f"Time Left: {time_left_str}")

    def create_visualization_group(self):
        visualization_group = QGroupBox("Training Visualization")
        vis_layout = QVBoxLayout()
        self.visualization.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vis_layout.addWidget(self.visualization)
        visualization_group.setLayout(vis_layout)
        return visualization_group
    
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