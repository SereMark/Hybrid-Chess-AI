from PyQt5.QtWidgets import (
    QMessageBox, QVBoxLayout, QProgressBar, QTextEdit, QWidget,
    QLabel, QLineEdit, QHBoxLayout, QPushButton, QGroupBox, QCheckBox, QFileDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, QWaitCondition, QMutex
from src.neural_network.train import ModelTrainer
from src.gui.visualizations.training_visualization import TrainingVisualization
import os

class TrainingWorker(QThread):
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    loss_update = pyqtSignal(float, float)
    val_loss_update = pyqtSignal(float, float)
    accuracy_update = pyqtSignal(float, float)
    time_left_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, epochs, batch_size, learning_rate, momentum, weight_decay, save_checkpoints,
                 checkpoint_interval, dataset_path, train_indices_path, val_indices_path, checkpoint_path=None,
                 automatic_batch_size=False):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.dataset_path = dataset_path
        self.train_indices_path = train_indices_path
        self.val_indices_path = val_indices_path
        self.checkpoint_path = checkpoint_path
        self.automatic_batch_size = automatic_batch_size
        self.stop_training = False
        self.pause_training = False
        self._pause_condition = QWaitCondition()
        self._mutex = QMutex()

    def run(self):
        try:
            trainer = ModelTrainer(
                epochs=self.epochs, batch_size=self.batch_size, lr=self.learning_rate,
                momentum=self.momentum, weight_decay=self.weight_decay,
                log_fn=self.log_update.emit, progress_fn=self.emit_progress,
                loss_fn=self.emit_loss, val_loss_fn=self.emit_val_loss,
                accuracy_fn=self.emit_accuracy, time_left_fn=self.emit_time_left,
                stop_fn=lambda: self.stop_training,
                pause_fn=lambda: self.pause_training,
                save_checkpoints=self.save_checkpoints,
                checkpoint_interval=self.checkpoint_interval,
                dataset_path=self.dataset_path,
                train_indices_path=self.train_indices_path,
                val_indices_path=self.val_indices_path,
                checkpoint_path=self.checkpoint_path,
                automatic_batch_size=self.automatic_batch_size
            )
            self.log_update.emit("Starting training...")
            trainer.train_model()
            if not self.stop_training:
                self.log_update.emit("Training completed successfully.")
        except Exception as e:
            self.log_update.emit(f"Error during training: {e}")
        finally:
            self.finished.emit()

    def emit_progress(self, value):
        self.progress_update.emit(value)

    def emit_loss(self, policy_loss, value_loss):
        self.loss_update.emit(policy_loss, value_loss)

    def emit_val_loss(self, val_policy_loss, val_value_loss):
        self.val_loss_update.emit(val_policy_loss, val_value_loss)

    def emit_accuracy(self, training_accuracy, validation_accuracy):
        self.accuracy_update.emit(training_accuracy, validation_accuracy)

    def emit_time_left(self, time_left_str):
        self.time_left_update.emit(time_left_str)

    def pause(self):
        self._mutex.lock()
        self.pause_training = True
        self._mutex.unlock()

    def resume(self):
        self._mutex.lock()
        self.pause_training = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()

    def stop(self):
        self._mutex.lock()
        self.stop_training = True
        if self.pause_training:
            self.pause_training = False
            self._pause_condition.wakeAll()
        self._mutex.unlock()

class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = TrainingVisualization()
        self.init_specific_ui()

    def init_specific_ui(self):
        layout = QVBoxLayout(self)
        layout.addLayout(self.create_input_layout("Dataset Path:", "data/processed/dataset.h5", "dataset_input", "Browse", self.browse_dataset))
        layout.addLayout(self.create_input_layout("Train Indices Path:", "data/processed/train_indices.npy", "train_indices_input", "Browse", self.browse_train_indices))
        layout.addLayout(self.create_input_layout("Validation Indices Path:", "data/processed/val_indices.npy", "val_indices_input", "Browse", self.browse_val_indices))
        layout.addLayout(self.create_input_layout("Epochs:", "10", "epochs_input"))
        layout.addLayout(self.create_input_layout("Batch Size:", "70", "batch_size_input"))
        layout.addLayout(self.create_input_layout("Learning Rate:", "0.1", "learning_rate_input"))
        layout.addLayout(self.create_input_layout("Momentum:", "0.9", "momentum_input"))
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
        input_field = QLineEdit(default_value)
        setattr(self, input_attr_name, input_field)
        layout.addWidget(label)
        layout.addWidget(input_field)
        if button_text and button_callback:
            button = QPushButton(button_text)
            button.clicked.connect(button_callback)
            layout.addWidget(button)
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

        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_training)
        self.pause_button.clicked.connect(self.pause_training)
        self.resume_button.clicked.connect(self.resume_training)
        self.stop_button.clicked.connect(self.stop_training)

        return layout

    def toggle_batch_size_input(self, checked):
        if checked:
            self.batch_size_input.setEnabled(False)
        else:
            self.batch_size_input.setEnabled(True)

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
            momentum = float(self.momentum_input.text())
            weight_decay = float(self.weight_decay_input.text())
            checkpoint_interval = int(self.checkpoint_interval_input.text())
            if any(v <= 0 for v in [epochs, learning_rate, momentum, weight_decay, checkpoint_interval]):
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

        save_checkpoints = self.save_checkpoints_checkbox.isChecked()

        self.worker = TrainingWorker(
            epochs, batch_size, learning_rate, momentum, weight_decay, save_checkpoints,
            checkpoint_interval, dataset_path, train_indices_path, val_indices_path, checkpoint_path,
            automatic_batch_size
        )
        self.worker.log_update.connect(self.log_text_edit.append)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.loss_update.connect(self.visualization.update_training_visualization)
        self.worker.val_loss_update.connect(self.visualization.update_validation_visualization)
        self.worker.accuracy_update.connect(self.visualization.update_accuracy_visualization)
        self.worker.time_left_update.connect(self.update_time_left)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.start()

    def pause_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)
            self.log_text_edit.append("Pausing training...")

    def resume_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.epochs = int(self.epochs_input.text())
            self.worker.learning_rate = float(self.learning_rate_input.text())
            self.worker.momentum = float(self.momentum_input.text())
            self.worker.weight_decay = float(self.weight_decay_input.text())
            self.worker.checkpoint_interval = int(self.checkpoint_interval_input.text())
            self.worker.resume()
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.log_text_edit.append("Resuming training...")

    def stop_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.log_text_edit.append("Stopping training...")

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