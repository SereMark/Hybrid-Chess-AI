from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, QLabel, QCheckBox, QComboBox, QMessageBox, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt
from src.supervised.supervised_training_worker import SupervisedWorker
from src.supervised.supervised_training_visualization import SupervisedVisualization
from src.base.base_tab import BaseTab
import os

class SupervisedTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = SupervisedVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        intro_label = QLabel("Train a supervised model on processed chess data to predict moves.")
        intro_label.setWordWrap(True)
        intro_label.setAlignment(Qt.AlignLeft)
        intro_label.setToolTip("Use this tab to train a neural network in a supervised manner using labeled data.")

        self.dataset_group = self.create_dataset_group()
        self.training_group = self.create_training_group()
        self.checkpoint_group = self.create_checkpoint_group()

        progress_group = QGroupBox("Training Progress")
        pg_layout = QVBoxLayout(progress_group)
        pg_layout.addLayout(self.create_progress_layout())

        logs_group = QGroupBox("Training Logs")
        lg_layout = QVBoxLayout(logs_group)
        self.log_text_edit = self.create_log_text_edit()
        lg_layout.addWidget(self.log_text_edit)

        self.visualization_group = self.create_visualization_group(self.visualization, "Training Visualization")

        controls_group = QGroupBox("Actions")
        cg_layout = QVBoxLayout(controls_group)
        control_buttons_layout = self.create_control_buttons(
            "Start Training",
            "Stop Training",
            self.start_training,
            self.stop_training,
            pause_text="Pause Training",
            resume_text="Resume Training",
            pause_callback=self.pause_worker,
            resume_callback=self.resume_worker
        )
        cg_layout.addLayout(control_buttons_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        main_layout.addWidget(intro_label)
        main_layout.addWidget(self.dataset_group)
        main_layout.addWidget(self.training_group)
        main_layout.addWidget(self.checkpoint_group)
        main_layout.addWidget(separator)
        main_layout.addWidget(progress_group)
        main_layout.addWidget(logs_group)
        main_layout.addWidget(self.visualization_group)
        main_layout.addWidget(controls_group)

        self.setup_batch_size_control(self.automatic_batch_size_checkbox, self.batch_size_input)
        interval_widgets = {
            'epoch': self.epoch_interval_widget,
            'time': self.time_interval_widget,
            'batch': self.batch_interval_widget,
        }
        self.setup_checkpoint_controls(self.save_checkpoints_checkbox, self.checkpoint_type_combo, interval_widgets)

        self.toggle_widget_state([self.log_text_edit, self.visualization_group], state=False, attribute="visible")

    def create_dataset_group(self):
        dataset_group = QGroupBox("Dataset Settings")
        dataset_layout = QFormLayout()

        self.dataset_input = QLineEdit("data/processed/dataset.h5")
        self.dataset_input.setPlaceholderText("Path to dataset file (HDF5)")
        self.dataset_input.setToolTip("Path to the main dataset file in HDF5 format.")

        self.train_indices_input = QLineEdit("data/processed/train_indices.npy")
        self.train_indices_input.setPlaceholderText("Path to training indices file")
        self.train_indices_input.setToolTip("NumPy file containing indices for training examples.")

        self.val_indices_input = QLineEdit("data/processed/val_indices.npy")
        self.val_indices_input.setPlaceholderText("Path to validation indices file")
        self.val_indices_input.setToolTip("NumPy file containing indices for validation examples.")

        dataset_browse_button = QPushButton("Browse")
        dataset_browse_button.setToolTip("Browse for the dataset file.")
        dataset_browse_button.clicked.connect(lambda: self.browse_file(self.dataset_input, "Select Dataset File", "HDF5 Files (*.h5 *.hdf5)"))

        train_indices_browse_button = QPushButton("Browse")
        train_indices_browse_button.setToolTip("Browse for the train indices file.")
        train_indices_browse_button.clicked.connect(lambda: self.browse_file(self.train_indices_input, "Select Train Indices File", "NumPy Files (*.npy)"))

        val_indices_browse_button = QPushButton("Browse")
        val_indices_browse_button.setToolTip("Browse for the validation indices file.")
        val_indices_browse_button.clicked.connect(lambda: self.browse_file(self.val_indices_input, "Select Validation Indices File", "NumPy Files (*.npy)"))

        dataset_layout.addRow("Dataset Path:", self.create_browse_layout(self.dataset_input, dataset_browse_button))
        dataset_layout.addRow("Train Indices Path:", self.create_browse_layout(self.train_indices_input, train_indices_browse_button))
        dataset_layout.addRow("Validation Indices Path:", self.create_browse_layout(self.val_indices_input, val_indices_browse_button))

        dataset_group.setLayout(dataset_layout)
        return dataset_group

    def create_training_group(self):
        training_group = QGroupBox("Training Hyperparameters")
        training_layout = QFormLayout()

        self.epochs_input = QLineEdit("25")
        self.epochs_input.setPlaceholderText("e.g. 25")
        self.epochs_input.setToolTip("Number of training epochs.")

        self.batch_size_input = QLineEdit("128")
        self.batch_size_input.setPlaceholderText("e.g. 128")
        self.batch_size_input.setToolTip("Batch size for training (if not automatic).")

        self.learning_rate_input = QLineEdit("0.0005")
        self.learning_rate_input.setPlaceholderText("e.g. 0.0005")
        self.learning_rate_input.setToolTip("Learning rate for the optimizer.")

        self.weight_decay_input = QLineEdit("2e-4")
        self.weight_decay_input.setPlaceholderText("e.g. 2e-4")
        self.weight_decay_input.setToolTip("Weight decay (L2 regularization) rate.")

        self.optimizer_type_combo = QComboBox()
        self.optimizer_type_combo.addItems(['AdamW', 'SGD'])
        self.optimizer_type_combo.setToolTip("Select the optimizer type.")

        self.scheduler_type_combo = QComboBox()
        self.scheduler_type_combo.addItems(['CosineAnnealingWarmRestarts', 'StepLR'])
        self.scheduler_type_combo.setToolTip("Select the learning rate scheduler.")

        self.num_workers_input = QLineEdit("4")
        self.num_workers_input.setPlaceholderText("e.g. 4")
        self.num_workers_input.setToolTip("Number of worker threads for data loading.")

        self.automatic_batch_size_checkbox = QCheckBox("Automatic Batch Size")
        self.automatic_batch_size_checkbox.setChecked(True)
        self.automatic_batch_size_checkbox.setToolTip("Enable automatic batch size determination.")

        self.output_model_path_input = QLineEdit("models/saved_models/pre_trained_model.pth")
        self.output_model_path_input.setPlaceholderText("Path to save the trained model")
        self.output_model_path_input.setToolTip("Path where the trained model will be saved.")

        output_model_browse_button = QPushButton("Browse")
        output_model_browse_button.setToolTip("Browse for the output model file path.")
        output_model_browse_button.clicked.connect(lambda: self.browse_file(self.output_model_path_input, "Select Output Model File", "PyTorch Files (*.pth *.pt)"))

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
        self.save_checkpoints_checkbox.setToolTip("Enable saving model checkpoints at intervals.")

        self.checkpoint_type_combo = QComboBox()
        self.checkpoint_type_combo.addItems(['Epoch', 'Time', 'Batch'])
        self.checkpoint_type_combo.setToolTip("Select checkpoint saving criterion.")

        checkpoint_type_layout = QHBoxLayout()
        cpl = QLabel("Save checkpoint by:")
        cpl.setToolTip("Choose how checkpoints are triggered.")
        checkpoint_type_layout.addWidget(cpl)
        checkpoint_type_layout.addWidget(self.checkpoint_type_combo)
        checkpoint_type_layout.addStretch()

        self.checkpoint_interval_input = QLineEdit("1")
        self.checkpoint_interval_input.setPlaceholderText("e.g. 1")
        self.checkpoint_interval_input.setToolTip("Interval for epoch-based checkpoints.")

        self.checkpoint_interval_minutes_input = QLineEdit("30")
        self.checkpoint_interval_minutes_input.setPlaceholderText("e.g. 30")
        self.checkpoint_interval_minutes_input.setToolTip("Interval in minutes for time-based checkpoints.")

        self.checkpoint_batch_interval_input = QLineEdit("2000")
        self.checkpoint_batch_interval_input.setPlaceholderText("e.g. 2000")
        self.checkpoint_batch_interval_input.setToolTip("Interval in batches for batch-based checkpoints.")

        self.epoch_interval_widget = self.create_interval_widget("Every", self.checkpoint_interval_input, "epochs")
        self.time_interval_widget = self.create_interval_widget("Every", self.checkpoint_interval_minutes_input, "minutes")
        self.batch_interval_widget = self.create_interval_widget("Every", self.checkpoint_batch_interval_input, "batches")

        self.checkpoint_path_input = QLineEdit("")
        self.checkpoint_path_input.setPlaceholderText("Path to resume from checkpoint (optional)")
        self.checkpoint_path_input.setToolTip("If resuming training, specify the checkpoint file path here.")

        checkpoint_browse_button = QPushButton("Browse")
        checkpoint_browse_button.setToolTip("Browse for the checkpoint file to resume from.")
        checkpoint_browse_button.clicked.connect(lambda: self.browse_file(self.checkpoint_path_input, "Select Checkpoint File", "PyTorch Files (*.pth *.pt)"))

        checkpoint_layout.addRow(self.save_checkpoints_checkbox)
        checkpoint_layout.addRow(checkpoint_type_layout)
        checkpoint_layout.addRow(self.epoch_interval_widget)
        checkpoint_layout.addRow(self.time_interval_widget)
        checkpoint_layout.addRow(self.batch_interval_widget)
        checkpoint_layout.addRow("Resume from checkpoint:", self.create_browse_layout(self.checkpoint_path_input, checkpoint_browse_button))

        checkpoint_group.setLayout(checkpoint_layout)
        return checkpoint_group

    def start_training(self):
        try:
            epochs = int(self.epochs_input.text())
            learning_rate = float(self.learning_rate_input.text())
            weight_decay = float(self.weight_decay_input.text())
            num_workers = int(self.num_workers_input.text())
            if any(v <= 0 for v in [epochs, learning_rate, weight_decay, num_workers]):
                raise ValueError("Epochs, Learning Rate, Weight Decay, and Number of Workers must be positive.")
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

        self.toggle_widget_state([self.start_button], state=False, attribute="enabled")
        self.toggle_widget_state([self.stop_button, self.pause_button], state=True, attribute="enabled")
        self.toggle_widget_state([self.resume_button], state=False, attribute="enabled")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()

        if not os.path.exists(os.path.dirname(output_model_path)):
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

        self.toggle_widget_state([self.dataset_group, self.training_group, self.checkpoint_group], state=False, attribute="visible")
        self.toggle_widget_state([self.log_text_edit, self.visualization_group], state=True, attribute="visible")

        started = self.start_worker(
            SupervisedWorker,
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
            self.worker.task_finished.connect(self.on_training_finished)
            self.worker.paused.connect(self.on_worker_paused)
        else:
            self.toggle_widget_state([self.start_button], state=True, attribute="enabled")
            self.toggle_widget_state([self.pause_button, self.stop_button, self.resume_button], state=False, attribute="enabled")
            self.toggle_widget_state([self.dataset_group, self.training_group, self.checkpoint_group], state=True, attribute="visible")
            self.toggle_widget_state([self.log_text_edit, self.visualization_group], state=False, attribute="visible")

        if not self.checkpoint_path_input.text():
            self.visualization.reset_visualization()

    def stop_training(self):
        self.stop_worker()
        self.toggle_widget_state([self.start_button], state=True, attribute="enabled")
        self.toggle_widget_state([self.pause_button, self.resume_button, self.stop_button], state=False, attribute="enabled")
        self.toggle_widget_state([self.dataset_group, self.training_group, self.checkpoint_group], state=True, attribute="visible")

    def on_training_finished(self):
        self.toggle_widget_state([self.start_button], state=True, attribute="enabled")
        self.toggle_widget_state([self.pause_button, self.resume_button, self.stop_button], state=False, attribute="enabled")
        self.progress_bar.setFormat("Training Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.toggle_widget_state([self.dataset_group, self.training_group, self.checkpoint_group], state=True, attribute="visible")