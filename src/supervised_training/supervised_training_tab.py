from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, QWidget, QTextEdit,
    QHBoxLayout, QLabel, QCheckBox, QComboBox, QFileDialog, QMessageBox, QSizePolicy
)
import os
from src.supervised_training.supervised_training_visualization import SupervisedTrainingVisualization
from src.supervised_training.supervised_training_worker import SupervisedTrainingWorker
from src.base.base_tab import BaseTab

class SupervisedTrainingTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = SupervisedTrainingVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        self.dataset_group = self.create_dataset_group()
        self.training_group = self.create_training_group()
        self.checkpoint_group = self.create_checkpoint_group()
        control_buttons_layout = self.create_buttons_layout()
        progress_layout = self.create_progress_layout()
        self.log_text_edit = self.create_log_text_edit()
        self.visualization_group = self.create_visualization_group()

        main_layout.addWidget(self.dataset_group)
        main_layout.addWidget(self.training_group)
        main_layout.addWidget(self.checkpoint_group)
        main_layout.addLayout(control_buttons_layout)
        main_layout.addLayout(progress_layout)
        main_layout.addWidget(self.log_text_edit)
        main_layout.addWidget(self.visualization_group)

        self.toggle_batch_size_input(self.automatic_batch_size_checkbox.isChecked())
        self.on_checkpoint_enabled_changed(self.save_checkpoints_checkbox.isChecked())
        self.log_text_edit.setVisible(False)
        self.visualization_group.setVisible(False)

    def create_dataset_group(self):
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

        dataset_layout.addRow("Dataset Path:", self.create_browse_layout(self.dataset_input, dataset_browse_button))
        dataset_layout.addRow("Train Indices Path:", self.create_browse_layout(self.train_indices_input, train_indices_browse_button))
        dataset_layout.addRow("Validation Indices Path:", self.create_browse_layout(self.val_indices_input, val_indices_browse_button))

        dataset_group.setLayout(dataset_layout)
        return dataset_group

    def create_training_group(self):
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

        self.num_workers_input = QLineEdit("4")

        self.automatic_batch_size_checkbox = QCheckBox("Automatic Batch Size")
        self.automatic_batch_size_checkbox.setChecked(True)
        self.automatic_batch_size_checkbox.toggled.connect(self.toggle_batch_size_input)

        self.output_model_path_input = QLineEdit("models/saved_models/pre_trained_model.pth")
        output_model_browse_button = QPushButton("Browse")
        output_model_browse_button.clicked.connect(self.browse_output_model)

        training_layout.addRow("Epochs:", self.epochs_input)
        training_layout.addRow("Batch Size:", self.batch_size_input)
        training_layout.addRow(self.automatic_batch_size_checkbox)
        training_layout.addRow("Learning Rate:", self.learning_rate_input)
        training_layout.addRow("Weight Decay:", self.weight_decay_input)
        training_layout.addRow("Optimizer Type:", self.optimizer_type_combo)
        training_layout.addRow("Scheduler Type:", self.scheduler_type_combo)
        training_layout.addRow("Number of Workers:", self.num_workers_input)
        training_layout.addRow("Output Model Path:", self.create_browse_layout(self.output_model_path_input, output_model_browse_button))

        training_group.setLayout(training_layout)
        return training_group

    def create_checkpoint_group(self):
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
        return checkpoint_group

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

    def create_log_text_edit(self):
        log_text_edit = QTextEdit()
        log_text_edit.setReadOnly(True)
        self.log_text_edit = log_text_edit
        return log_text_edit

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
        self.checkpoint_path_input.setEnabled(is_enabled)
        self.on_checkpoint_type_changed(self.checkpoint_type_combo.currentText())

    def on_checkpoint_type_changed(self, text):
        text = text.lower()
        self.epoch_interval_widget.setVisible(text == 'epoch')
        self.time_interval_widget.setVisible(text == 'time')
        self.batch_interval_widget.setVisible(text == 'batch')
        is_enabled = self.save_checkpoints_checkbox.isChecked()
        self.epoch_interval_widget.setEnabled(is_enabled)
        self.time_interval_widget.setEnabled(is_enabled)
        self.batch_interval_widget.setEnabled(is_enabled)

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

    def browse_output_model(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output Model File", self.output_model_path_input.text(), "PyTorch Files (*.pth *.pt)"
        )
        if file_path:
            self.output_model_path_input.setText(file_path)

    def start_training(self):
        try:
            epochs = int(self.epochs_input.text())
            learning_rate = float(self.learning_rate_input.text())
            weight_decay = float(self.weight_decay_input.text())
            num_workers = int(self.num_workers_input.text())
            if any(v <= 0 for v in [epochs, learning_rate, weight_decay, num_workers]):
                raise ValueError("Epochs, Learning Rate, Weight Decay, and Number of Workers must be positive.")
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
        output_model_path = self.output_model_path_input.text()

        if not all(os.path.exists(p) for p in [dataset_path, train_indices_path, val_indices_path]):
            QMessageBox.warning(self, "Error", "Dataset or indices files do not exist.")
            return

        save_checkpoints = self.save_checkpoints_checkbox.isChecked()
        checkpoint_type = self.checkpoint_type_combo.currentText().lower()
        checkpoint_interval = None
        checkpoint_interval_minutes = None
        checkpoint_batch_interval = None

        if save_checkpoints:
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

        if not os.path.exists(os.path.dirname(output_model_path)):
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

        self.dataset_group.setVisible(False)
        self.training_group.setVisible(False)
        self.checkpoint_group.setVisible(False)
        self.log_text_edit.setVisible(True)
        self.visualization_group.setVisible(True)

        started = self.start_worker(
            SupervisedTrainingWorker,
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
            scheduler_type=scheduler_type,
            output_model_path=output_model_path,
            num_workers=num_workers
        )
        if started:
            self.worker.batch_loss_update.connect(self.visualization.update_loss_plots)
            self.worker.batch_accuracy_update.connect(self.visualization.update_accuracy_plot)
            self.worker.val_loss_update.connect(self.visualization.update_validation_loss_plots)
            self.worker.epoch_accuracy_update.connect(self.visualization.update_validation_accuracy_plot)
            self.worker.training_finished.connect(self.on_training_finished)
            self.worker.paused.connect(self.on_worker_paused)
        else:
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.dataset_group.setVisible(True)
            self.training_group.setVisible(True)
            self.checkpoint_group.setVisible(True)
            self.log_text_edit.setVisible(False)
            self.visualization_group.setVisible(False)

        if not self.checkpoint_path_input.text():
            self.visualization.reset_visualization()

    def on_worker_paused(self, is_paused):
        self.pause_button.setEnabled(not is_paused)
        self.resume_button.setEnabled(is_paused)

    def pause_training(self):
        if self.worker:
            self.worker.pause()

    def resume_training(self):
        if self.worker:
            self.worker.resume()

    def stop_training(self):
        self.stop_worker()
        self.log_message("Stopping training...")
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.dataset_group.setVisible(True)
        self.training_group.setVisible(True)
        self.checkpoint_group.setVisible(True)

    def on_training_finished(self):
        self.log_message("Training process has been completed.")
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Training Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.dataset_group.setVisible(True)
        self.training_group.setVisible(True)
        self.checkpoint_group.setVisible(True)