import threading
import queue
import os
from PyQt5.QtWidgets import (
    QMessageBox, QVBoxLayout, QProgressBar, QTextEdit, QWidget,
    QLabel, QLineEdit, QHBoxLayout, QPushButton, QGroupBox, QCheckBox, QFileDialog
)
from PyQt5 import QtCore
from src.neural_network.train import ModelTrainer
from src.gui.visualizations.training_visualization import TrainingVisualization

class TrainingWorker(threading.Thread):
    def __init__(self, epochs, batch_size, learning_rate, weight_decay, save_checkpoints,
                 checkpoint_interval, dataset_path, train_indices_path, val_indices_path, checkpoint_path=None,
                 automatic_batch_size=False, log_queue=None, progress_queue=None, stats_queue=None):
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
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.pause_event.set()
        self.log_queue = log_queue
        self.progress_queue = progress_queue
        self.stats_queue = stats_queue

    def run(self):
        try:
            trainer = ModelTrainer(
                epochs=self.epochs, batch_size=self.batch_size, lr=self.learning_rate,
                weight_decay=self.weight_decay,
                log_fn=self.log_update, progress_fn=self.update_progress,
                loss_fn=self.update_epoch_loss,
                accuracy_fn=self.update_accuracy, time_left_fn=self.update_time_left,
                pause_event=self.pause_event, stop_event=self.stop_event,
                save_checkpoints=self.save_checkpoints,
                checkpoint_interval=self.checkpoint_interval,
                dataset_path=self.dataset_path,
                train_indices_path=self.train_indices_path,
                val_indices_path=self.val_indices_path,
                checkpoint_path=self.checkpoint_path,
                automatic_batch_size=self.automatic_batch_size,
                batch_loss_fn=self.update_batch_loss,
                batch_accuracy_fn=self.update_batch_accuracy,
                lr_fn=self.update_learning_rate
            )
            self.log_update("Starting training...")
            trainer.train_model()
            if not self.stop_event.is_set():
                self.log_update("Training completed successfully.")
        except Exception as e:
            self.log_update(f"Error during training: {e}")

    def log_update(self, message):
        if self.log_queue:
            self.log_queue.put(message)

    def update_progress(self, value):
        if self.progress_queue:
            self.progress_queue.put(('progress', value))

    def update_epoch_loss(self, epoch, train_losses):
        if self.stats_queue:
            self.stats_queue.put(('epoch_loss', epoch, train_losses))

    def update_accuracy(self, epoch, training_accuracy, validation_accuracy):
        if self.stats_queue:
            self.stats_queue.put(('accuracy', epoch, training_accuracy, validation_accuracy))

    def update_time_left(self, time_left_str):
        if self.progress_queue:
            self.progress_queue.put(('time_left', time_left_str))

    def update_batch_loss(self, batch_idx, losses):
        if self.stats_queue:
            self.stats_queue.put(('batch_loss', batch_idx, losses))

    def update_batch_accuracy(self, batch_idx, accuracy):
        if self.stats_queue:
            self.stats_queue.put(('batch_accuracy', batch_idx, accuracy))

    def update_learning_rate(self, batch_idx, lr):
        if self.stats_queue:
            self.stats_queue.put(('learning_rate', batch_idx, lr))

    def pause(self):
        self.pause_event.clear()
        self.log_update("Training paused by user.")

    def resume(self):
        self.pause_event.set()
        self.log_update("Training resumed by user.")

    def stop(self):
        self.stop_event.set()
        self.pause_event.set()
        self.log_update("Training stopped by user.")

class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = TrainingVisualization()
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.stats_queue = queue.Queue()
        self.init_specific_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)

    def init_specific_ui(self):
        layout = QVBoxLayout(self)
        layout.addLayout(self.create_input_layout("Dataset Path:", "data/processed/dataset.h5", "dataset_input", "Browse", self.browse_dataset))
        layout.addLayout(self.create_input_layout("Train Indices Path:", "data/processed/train_indices.npy", "train_indices_input", "Browse", self.browse_train_indices))
        layout.addLayout(self.create_input_layout("Validation Indices Path:", "data/processed/val_indices.npy", "val_indices_input", "Browse", self.browse_val_indices))
        layout.addLayout(self.create_input_layout("Epochs:", "10", "epochs_input"))
        layout.addLayout(self.create_input_layout("Batch Size:", "256", "batch_size_input"))
        layout.addLayout(self.create_input_layout("Learning Rate:", "0.001", "learning_rate_input"))
        layout.addLayout(self.create_input_layout("Weight Decay:", "1e-4", "weight_decay_input"))
        layout.addLayout(self.create_input_layout("Checkpoint Interval (epochs):", "1", "checkpoint_interval_input"))
        layout.addLayout(self.create_input_layout("Checkpoint Path (Optional):", "", "checkpoint_path_input", "Browse", self.browse_checkpoint))
        self.save_checkpoints_checkbox = QCheckBox("Save Checkpoints")
        self.save_checkpoints_checkbox.setChecked(True)
        layout.addWidget(self.save_checkpoints_checkbox)
        self.automatic_batch_size_checkbox = QCheckBox("Automatic Batch Size")
        self.automatic_batch_size_checkbox.setChecked(False)
        self.automatic_batch_size_checkbox.toggled.connect(self.toggle_batch_size_input)
        layout.addWidget(self.automatic_batch_size_checkbox)
        buttons_layout = self.create_buttons_layout()
        layout.addLayout(buttons_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        layout.addWidget(self.progress_bar)
        self.remaining_time_label = QLabel("Time Left: Calculating...")
        layout.addWidget(self.remaining_time_label)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        layout.addWidget(self.log_text_edit)
        vis_group = QGroupBox("Training Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

    def create_input_layout(self, label_text, default_value, input_attr_name, button_text=None, button_callback=None):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(200)
        input_field = QLineEdit(default_value)
        input_field.setFixedWidth(200)
        setattr(self, input_attr_name, input_field)
        layout.addWidget(label)
        layout.addWidget(input_field)
        if button_text and button_callback:
            button = QPushButton(button_text)
            button.clicked.connect(button_callback)
            layout.addWidget(button)
        layout.addStretch()
        return layout

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
            checkpoint_interval = int(self.checkpoint_interval_input.text())
            if any(v <= 0 for v in [epochs, learning_rate, weight_decay, checkpoint_interval]):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid and positive hyperparameters.")
            return
        automatic_batch_size = self.automatic_batch_size_checkbox.isChecked()
        if automatic_batch_size:
            batch_size = None
        else:
            try:
                batch_size = int(self.batch_size_input.text())
                if batch_size <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Batch Size must be a positive integer.")
                return
        dataset_path = self.dataset_input.text()
        train_indices_path = self.train_indices_input.text()
        val_indices_path = self.val_indices_input.text()
        checkpoint_path = self.checkpoint_path_input.text() if self.checkpoint_path_input.text() else None
        if not all(os.path.exists(p) for p in [dataset_path, train_indices_path, val_indices_path]):
            QMessageBox.warning(self, "Error", "Dataset or indices files do not exist.")
            return
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.visualization.reset_visualization()
        save_checkpoints = self.save_checkpoints_checkbox.isChecked()
        self.worker = TrainingWorker(
            epochs, batch_size, learning_rate, weight_decay, save_checkpoints,
            checkpoint_interval, dataset_path, train_indices_path, val_indices_path, checkpoint_path,
            automatic_batch_size,
            log_queue=self.log_queue,
            progress_queue=self.progress_queue,
            stats_queue=self.stats_queue
        )
        self.worker.start()

    def pause_training(self):
        if self.worker and self.worker.is_alive():
            self.worker.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)

    def resume_training(self):
        if self.worker and self.worker.is_alive():
            self.worker.resume()
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)

    def stop_training(self):
        if self.worker and self.worker.is_alive():
            self.worker.stop()

    def update_ui(self):
        while not self.log_queue.empty():
            message = self.log_queue.get()
            self.log_text_edit.append(message)
            if "Training completed successfully." in message or "Training stopped by user." in message:
                self.on_training_finished()
        while not self.progress_queue.empty():
            item = self.progress_queue.get()
            if item[0] == 'progress':
                self.update_progress(item[1])
            elif item[0] == 'time_left':
                self.update_time_left(item[1])
        while not self.stats_queue.empty():
            item = self.stats_queue.get()
            if item[0] == 'batch_loss':
                batch_idx = item[1]
                losses = item[2]
                self.visualization.update_loss_plots(batch_idx, losses)
            elif item[0] == 'batch_accuracy':
                batch_idx = item[1]
                accuracy = item[2]
                self.visualization.update_accuracy_plot(batch_idx, accuracy)
            elif item[0] == 'learning_rate':
                batch_idx = item[1]
                lr = item[2]
                self.visualization.update_learning_rate(batch_idx, lr)
            elif item[0] == 'epoch_loss':
                pass
            elif item[0] == 'accuracy':
                pass

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