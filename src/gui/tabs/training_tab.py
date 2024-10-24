import os, threading
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import (
    QMessageBox, QVBoxLayout, QProgressBar, QTextEdit, QWidget,
    QLabel, QLineEdit, QHBoxLayout, QPushButton, QGroupBox, QCheckBox, QFileDialog, QFormLayout, QComboBox, QSizePolicy
)
from src.neural_network.train import ModelTrainer
from src.gui.visualizations.training_visualization import TrainingVisualization

class TrainingWorker(QObject):
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


class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.worker = None
        self.visualization = TrainingVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        settings_group = QGroupBox("Training Settings")
        settings_layout = QFormLayout()
        self.dataset_input = QLineEdit("data/processed/dataset.h5")
        self.train_indices_input = QLineEdit("data/processed/train_indices.npy")
        self.val_indices_input = QLineEdit("data/processed/val_indices.npy")
        self.epochs_input = QLineEdit("10")
        self.batch_size_input = QLineEdit("256")
        self.learning_rate_input = QLineEdit("0.001")
        self.weight_decay_input = QLineEdit("1e-4")
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
        self.checkpoint_interval_minutes_input = QLineEdit("60")
        time_layout.addWidget(QLabel("Every"))
        time_layout.addWidget(self.checkpoint_interval_minutes_input)
        time_layout.addWidget(QLabel("minutes"))
        time_layout.addStretch()
        self.time_interval_widget = QWidget()
        self.time_interval_widget.setLayout(time_layout)
        interval_layout.addWidget(self.time_interval_widget)
        batch_layout = QHBoxLayout()
        self.checkpoint_batch_interval_input = QLineEdit("1000")
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
        self.automatic_batch_size_checkbox.setChecked(False)
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
        self.worker = TrainingWorker(
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