from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton,
    QHBoxLayout, QLabel, QCheckBox, QComboBox, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QSizePolicy
)
import os
from src.gui.visualizations.supervised_training_visualization import SupervisedTrainingVisualization
from src.gui.workers.supervised_training_worker import SupervisedTrainingWorker


class SupervisedTrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.worker = None
        self.visualization = SupervisedTrainingVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        dataset_group = QGroupBox("Dataset Settings")
        dataset_layout = QFormLayout()

        self.dataset_input = QLineEdit("data/processed/dataset.h5")
        self.train_indices_input = QLineEdit("data/processed/train_indices.npy")
        self.val_indices_input = QLineEdit("data/processed/val_indices.npy")

        dataset_browse_button = QPushButton("Browse")
        dataset_browse_button.clicked.connect(self.browse_dataset)
        train_indices_browse_button = QPushButton("Browse")
        train_indices_browse_button.clicked.connect(self.browse_train_indices)
        val_indices_browse_button = QPushButton("Browse")
        val_indices_browse_button.clicked.connect(self.browse_val_indices)

        dataset_layout.addRow(
            "Dataset Path:", self.create_browse_layout(self.dataset_input, dataset_browse_button)
        )
        dataset_layout.addRow(
            "Train Indices Path:", self.create_browse_layout(self.train_indices_input, train_indices_browse_button)
        )
        dataset_layout.addRow(
            "Validation Indices Path:", self.create_browse_layout(self.val_indices_input, val_indices_browse_button)
        )

        dataset_group.setLayout(dataset_layout)

        training_group = QGroupBox("Training Hyperparameters")
        training_layout = QFormLayout()

        self.epochs_input = QLineEdit("25")
        self.batch_size_input = QLineEdit("128")
        self.learning_rate_input = QLineEdit("0.0005")
        self.weight_decay_input = QLineEdit("2e-4")
        self.optimizer_type_combo = QComboBox()
        self.optimizer_type_combo.addItems(['AdamW', 'SGD'])
        self.scheduler_type_combo = QComboBox()
        self.scheduler_type_combo.addItems(['CosineAnnealingWarmRestarts', 'StepLR'])

        self.automatic_batch_size_checkbox = QCheckBox("Automatic Batch Size")
        self.automatic_batch_size_checkbox.setChecked(True)
        self.automatic_batch_size_checkbox.toggled.connect(self.toggle_batch_size_input)

        training_layout.addRow("Epochs:", self.epochs_input)
        training_layout.addRow("Batch Size:", self.batch_size_input)
        training_layout.addRow(self.automatic_batch_size_checkbox)
        training_layout.addRow("Learning Rate:", self.learning_rate_input)
        training_layout.addRow("Weight Decay:", self.weight_decay_input)
        training_layout.addRow("Optimizer Type:", self.optimizer_type_combo)
        training_layout.addRow("Scheduler Type:", self.scheduler_type_combo)

        training_group.setLayout(training_layout)

        checkpoint_group = QGroupBox("Checkpoint Settings")
        checkpoint_layout = QFormLayout()

        self.save_checkpoints_checkbox = QCheckBox("Enable Checkpoints")
        self.save_checkpoints_checkbox.setChecked(True)
        self.save_checkpoints_checkbox.stateChanged.connect(self.on_checkpoint_enabled_changed)

        self.checkpoint_type_combo = QComboBox()
        self.checkpoint_type_combo.addItems(['Epoch', 'Time', 'Batch'])
        self.checkpoint_type_combo.currentTextChanged.connect(self.on_checkpoint_type_changed)

        checkpoint_type_layout = QHBoxLayout()
        checkpoint_type_layout.addWidget(QLabel("Save checkpoint by:"))
        checkpoint_type_layout.addWidget(self.checkpoint_type_combo)
        checkpoint_type_layout.addStretch()

        self.checkpoint_interval_input = QLineEdit("1")
        self.checkpoint_interval_minutes_input = QLineEdit("30")
        self.checkpoint_batch_interval_input = QLineEdit("2000")

        self.epoch_interval_widget = self.create_interval_widget(
            "Every", self.checkpoint_interval_input, "epochs"
        )
        self.time_interval_widget = self.create_interval_widget(
            "Every", self.checkpoint_interval_minutes_input, "minutes"
        )
        self.batch_interval_widget = self.create_interval_widget(
            "Every", self.checkpoint_batch_interval_input, "batches"
        )

        self.on_checkpoint_type_changed(self.checkpoint_type_combo.currentText())

        self.checkpoint_path_input = QLineEdit("")
        checkpoint_browse_button = QPushButton("Browse")
        checkpoint_browse_button.clicked.connect(self.browse_checkpoint)

        checkpoint_layout.addRow(self.save_checkpoints_checkbox)
        checkpoint_layout.addRow(checkpoint_type_layout)
        checkpoint_layout.addRow(self.epoch_interval_widget)
        checkpoint_layout.addRow(self.time_interval_widget)
        checkpoint_layout.addRow(self.batch_interval_widget)
        checkpoint_layout.addRow(
            "Resume from checkpoint:",
            self.create_browse_layout(self.checkpoint_path_input, checkpoint_browse_button)
        )

        checkpoint_group.setLayout(checkpoint_layout)

        control_buttons_layout = self.create_buttons_layout()

        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label = QLabel("Time Left: Calculating...")
        self.remaining_time_label.setAlignment(Qt.AlignCenter)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.remaining_time_label)

        main_layout.addWidget(dataset_group)
        main_layout.addWidget(training_group)
        main_layout.addWidget(checkpoint_group)
        main_layout.addLayout(control_buttons_layout)
        main_layout.addLayout(progress_layout)
        main_layout.addWidget(self.log_text_edit())
        main_layout.addWidget(self.create_visualization_group())

        self.toggle_batch_size_input(self.automatic_batch_size_checkbox.isChecked())

    def create_browse_layout(self, line_edit, browse_button):
        layout = QHBoxLayout()
        layout.addWidget(line_edit)
        layout.addWidget(browse_button)
        return layout

    def create_interval_widget(self, prefix, input_field, suffix):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(prefix))
        layout.addWidget(input_field)
        layout.addWidget(QLabel(suffix))
        layout.addStretch()
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def on_checkpoint_enabled_changed(self, state):
        is_enabled = state == Qt.Checked
        self.checkpoint_type_combo.setEnabled(is_enabled)
        self.on_checkpoint_type_changed(self.checkpoint_type_combo.currentText())
        self.checkpoint_path_input.setEnabled(is_enabled)

    def on_checkpoint_type_changed(self, text):
        text = text.lower()
        self.epoch_interval_widget.setVisible(text == 'epoch')
        self.time_interval_widget.setVisible(text == 'time')
        self.batch_interval_widget.setVisible(text == 'batch')
        self.epoch_interval_widget.setEnabled(self.save_checkpoints_checkbox.isChecked())
        self.time_interval_widget.setEnabled(self.save_checkpoints_checkbox.isChecked())
        self.batch_interval_widget.setEnabled(self.save_checkpoints_checkbox.isChecked())

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

    def create_visualization_group(self):
        visualization_group = QGroupBox("Training Visualization")
        vis_layout = QVBoxLayout()
        self.visualization.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vis_layout.addWidget(self.visualization)
        visualization_group.setLayout(vis_layout)
        return visualization_group

    def log_text_edit(self):
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        return self.log_text_edit

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
            QMessageBox.warning(
                self, "Input Error", f"Please enter valid and positive parameters.\n{str(e)}"
            )
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
                QMessageBox.warning(
                    self, "Input Error", f"Batch Size must be a positive integer.\n{str(e)}"
                )
                return

        optimizer_type = self.optimizer_type_combo.currentText().lower()
        scheduler_type = self.scheduler_type_combo.currentText().lower()

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
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_checkpoints=save_checkpoints,
            checkpoint_interval=checkpoint_interval,
            dataset_path=dataset_path,
            train_indices_path=train_indices_path,
            val_indices_path=val_indices_path,
            checkpoint_path=checkpoint_path,
            automatic_batch_size=automatic_batch_size,
            checkpoint_type=checkpoint_type,
            checkpoint_interval_minutes=checkpoint_interval_minutes,
            checkpoint_batch_interval=checkpoint_batch_interval,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type
        )
        self.worker.moveToThread(self.thread)

        self.worker.log_update.connect(self.log_text_edit.append)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.batch_loss_update.connect(self.visualization.update_loss_plots)
        self.worker.batch_accuracy_update.connect(self.visualization.update_accuracy_plot)
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
        self.log_text_edit.append(
            f"Epoch {epoch} Training Losses - Policy: {losses['policy']:.4f}, Value: {losses['value']:.4f}"
        )

    def handle_val_loss(self, epoch, losses):
        self.log_text_edit.append(
            f"Epoch {epoch} Validation Losses - Policy: {losses['policy']:.4f}, Value: {losses['value']:.4f}"
        )

    def handle_epoch_accuracy(self, epoch, training_accuracy, validation_accuracy):
        self.log_text_edit.append(
            f"Epoch {epoch} Accuracy - Training: {training_accuracy*100:.2f}%, Validation: {validation_accuracy*100:.2f}%"
        )

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