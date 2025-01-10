from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QGridLayout, QLineEdit, QPushButton, QLabel, QCheckBox, QComboBox, QMessageBox, QHBoxLayout
from src.training.supervised.supervised_training_worker import SupervisedWorker
from src.training.supervised.supervised_training_visualization import SupervisedVisualization
from src.base.base_tab import BaseTab
import os

class SupervisedTrainingSubTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = SupervisedVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.setup_subtab(
            main_layout,
            "Train a supervised model on processed chess data to predict moves.",
            "Training Progress",
            "Training Logs",
            "Training Visualization",
            self.visualization,
            {
                "start_text": "Start Training",
                "stop_text": "Stop Training",
                "start_callback": self.start_training,
                "stop_callback": self.stop_training,
                "pause_text": "Pause Training",
                "resume_text": "Resume Training",
                "pause_callback": self.pause_worker,
                "resume_callback": self.resume_worker,
                "start_new_callback": self.reset_to_initial_state
            },
            "Configure Parameters"
        )

        self.input_settings_group = self.create_input_settings_group()
        self.dataset_group = self.create_dataset_group()
        self.training_group = self.create_training_group()
        self.checkpoint_group = self.create_checkpoint_group()

        self.layout().insertWidget(1, self.input_settings_group)
        self.layout().insertWidget(2, self.dataset_group)
        self.layout().insertWidget(3, self.training_group)
        self.layout().insertWidget(4, self.checkpoint_group)

        self.setup_batch_size_control(self.automatic_batch_size_checkbox, self.batch_size_input)

        interval_widgets = {
            'epoch': self.epoch_interval_widget,
            'time': self.time_interval_widget,
            'batch': self.batch_interval_widget
        }
        self.setup_checkpoint_controls(self.save_checkpoints_checkbox, self.checkpoint_type_combo, interval_widgets)

        if self.stop_button:
            self.stop_button.setEnabled(False)
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)

    def create_input_settings_group(self):
        group = QGroupBox("Input Settings")
        layout = QGridLayout()

        label_model_path = QLabel("Model Path (optional):")
        self.model_path_input = QLineEdit("")
        model_browse_button = QPushButton("Browse")
        model_browse_button.clicked.connect(lambda: self.browse_file(self.model_path_input, "Select Model File", "PyTorch Files (*.pth *.pt)"))
        layout.addWidget(label_model_path, 0, 0)
        layout.addLayout(self.create_browse_layout(self.model_path_input, model_browse_button), 0, 1, 1, 3)

        label_filters = QLabel("Filter Amount:")
        self.filters_input = QLineEdit("64")

        label_res_blocks = QLabel("Residual Blocks:")
        self.res_blocks_input = QLineEdit("5")

        label_inplace = QLabel("Inplace ReLU:")
        self.inplace_checkbox = QCheckBox()
        self.inplace_checkbox.setChecked(True)

        layout.addWidget(label_filters, 1, 0)
        layout.addWidget(self.filters_input, 1, 1)
        layout.addWidget(label_res_blocks, 1, 2)
        layout.addWidget(self.res_blocks_input, 1, 3)
        layout.addWidget(label_inplace, 2, 0)
        layout.addWidget(self.inplace_checkbox, 2, 1)

        group.setLayout(layout)

        self.model_path_input.textChanged.connect(self.on_model_path_changed)
        self.on_model_path_changed(self.model_path_input.text())

        return group

    def on_model_path_changed(self, text):
        has_model_path = bool(text.strip())
        self.filters_input.setEnabled(not has_model_path)
        self.res_blocks_input.setEnabled(not has_model_path)
        self.inplace_checkbox.setEnabled(not has_model_path)

    def create_dataset_group(self):
        group = QGroupBox("Dataset Settings")
        layout = QGridLayout()

        label1 = QLabel("Dataset Path:")
        self.dataset_input = QLineEdit("data/processed/dataset.h5")
        dataset_browse_button = QPushButton("Browse")
        dataset_browse_button.clicked.connect(lambda: self.browse_file(self.dataset_input, "Select Dataset File", "HDF5 Files (*.h5 *.hdf5)"))

        label2 = QLabel("Train Indices Path:")
        self.train_indices_input = QLineEdit("data/processed/train_indices.npy")
        train_indices_browse_button = QPushButton("Browse")
        train_indices_browse_button.clicked.connect(lambda: self.browse_file(self.train_indices_input, "Select Train Indices File", "NumPy Files (*.npy)"))

        label3 = QLabel("Validation Indices Path:")
        self.val_indices_input = QLineEdit("data/processed/val_indices.npy")
        val_indices_browse_button = QPushButton("Browse")
        val_indices_browse_button.clicked.connect(lambda: self.browse_file(self.val_indices_input, "Select Validation Indices File", "NumPy Files (*.npy)"))

        layout.addWidget(label1, 0, 0)
        layout.addLayout(self.create_browse_layout(self.dataset_input, dataset_browse_button), 0, 1, 1, 3)
        layout.addWidget(label2, 1, 0)
        layout.addLayout(self.create_browse_layout(self.train_indices_input, train_indices_browse_button), 1, 1, 1, 3)
        layout.addWidget(label3, 2, 0)
        layout.addLayout(self.create_browse_layout(self.val_indices_input, val_indices_browse_button), 2, 1, 1, 3)

        group.setLayout(layout)
        return group

    def create_training_group(self):
        group = QGroupBox("Training Hyperparameters")
        layout = QVBoxLayout()

        main_params_box = QGroupBox("Main Hyperparameters")
        main_params_layout = QGridLayout()

        label1 = QLabel("Epochs:")
        self.epochs_input = QLineEdit("25")

        label2 = QLabel("Batch Size:")
        self.batch_size_input = QLineEdit("128")
        self.automatic_batch_size_checkbox = QCheckBox("Automatic Batch Size")
        self.automatic_batch_size_checkbox.setChecked(True)

        label3 = QLabel("Learning Rate:")
        self.learning_rate_input = QLineEdit("0.0005")

        label4 = QLabel("Weight Decay:")
        self.weight_decay_input = QLineEdit("2e-4")

        label5 = QLabel("Optimizer Type:")
        self.optimizer_type_combo = QComboBox()
        self.optimizer_type_combo.addItems(["AdamW", "SGD"])

        label6 = QLabel("Scheduler Type:")
        self.scheduler_type_combo = QComboBox()
        self.scheduler_type_combo.addItems(["CosineAnnealingWarmRestarts", "StepLR"])

        label7 = QLabel("Number of Workers:")
        self.num_workers_input = QLineEdit("4")

        label8 = QLabel("Random Seed:")
        self.random_seed_input = QLineEdit("42")

        label9 = QLabel("Output Model Path:")
        self.output_model_path_input = QLineEdit("models/saved_models/pre_trained_model.pth")
        output_model_browse_button = QPushButton("Browse")
        output_model_browse_button.clicked.connect(lambda: self.browse_file(self.output_model_path_input, "Select Output Model File", "PyTorch Files (*.pth *.pt)"))

        main_params_layout.addWidget(label1, 0, 0)
        main_params_layout.addWidget(self.epochs_input, 0, 1)
        main_params_layout.addWidget(label2, 0, 2)
        main_params_layout.addWidget(self.batch_size_input, 0, 3)
        main_params_layout.addWidget(self.automatic_batch_size_checkbox, 0, 4)

        main_params_layout.addWidget(label3, 1, 0)
        main_params_layout.addWidget(self.learning_rate_input, 1, 1)
        main_params_layout.addWidget(label4, 1, 2)
        main_params_layout.addWidget(self.weight_decay_input, 1, 3)

        main_params_layout.addWidget(label5, 2, 0)
        main_params_layout.addWidget(self.optimizer_type_combo, 2, 1)
        main_params_layout.addWidget(label6, 2, 2)
        main_params_layout.addWidget(self.scheduler_type_combo, 2, 3)

        main_params_layout.addWidget(label7, 3, 0)
        main_params_layout.addWidget(self.num_workers_input, 3, 1)

        main_params_layout.addWidget(label8, 4, 0)
        main_params_layout.addWidget(self.random_seed_input, 4, 1)

        main_params_layout.addWidget(label9, 5, 0)
        main_params_layout.addLayout(self.create_browse_layout(self.output_model_path_input, output_model_browse_button), 5, 1, 1, 3)

        main_params_box.setLayout(main_params_layout)

        layout.addWidget(main_params_box)
        group.setLayout(layout)
        return group

    def create_checkpoint_group(self):
        group = QGroupBox("Checkpoint Settings")
        layout = QVBoxLayout()

        self.save_checkpoints_checkbox = QCheckBox("Enable Checkpoints")
        self.save_checkpoints_checkbox.setChecked(True)
        layout.addWidget(self.save_checkpoints_checkbox)

        row1 = QHBoxLayout()
        label1 = QLabel("Save checkpoint by:")
        row1.addWidget(label1)
        self.checkpoint_type_combo = QComboBox()
        self.checkpoint_type_combo.addItems(["Epoch", "Time", "Batch"])
        row1.addWidget(self.checkpoint_type_combo)
        row1.addStretch()
        layout.addLayout(row1)

        self.checkpoint_interval_input = QLineEdit("1")
        self.checkpoint_interval_minutes_input = QLineEdit("30")
        self.checkpoint_batch_interval_input = QLineEdit("2000")

        self.epoch_interval_widget = self.create_interval_widget("Every", self.checkpoint_interval_input, "epochs")
        self.time_interval_widget = self.create_interval_widget("Every", self.checkpoint_interval_minutes_input, "minutes")
        self.batch_interval_widget = self.create_interval_widget("Every", self.checkpoint_batch_interval_input, "batches")

        layout.addWidget(self.epoch_interval_widget)
        layout.addWidget(self.time_interval_widget)
        layout.addWidget(self.batch_interval_widget)

        group.setLayout(layout)
        return group

    def start_training(self):
        try:
            epochs = int(self.epochs_input.text())
            learning_rate = float(self.learning_rate_input.text())
            weight_decay = float(self.weight_decay_input.text())
            num_workers = int(self.num_workers_input.text())
            if any(v <= 0 for v in [epochs, learning_rate, weight_decay, num_workers]):
                raise ValueError("Epochs, Learning Rate, Weight Decay, and Number of Workers must be positive.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
            return

        automatic_batch_size = self.automatic_batch_size_checkbox.isChecked()
        if automatic_batch_size:
            batch_size = None
        else:
            try:
                b = int(self.batch_size_input.text())
                if b <= 0:
                    raise ValueError("Batch Size must be a positive integer.")
                batch_size = b
            except ValueError as e:
                QMessageBox.warning(self, "Input Error", str(e))
                return

        optimizer_type = self.optimizer_type_combo.currentText().lower()
        scheduler_type = self.scheduler_type_combo.currentText().lower()
        dataset_path = self.dataset_input.text()
        train_indices_path = self.train_indices_input.text()
        val_indices_path = self.val_indices_input.text()
        model_path = self.model_path_input.text().strip() if self.model_path_input.text().strip() else None
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
            if checkpoint_type == "epoch":
                try:
                    ci = int(self.checkpoint_interval_input.text())
                    if ci <= 0:
                        raise ValueError("Epoch interval must be positive.")
                    checkpoint_interval = ci
                except ValueError as er:
                    QMessageBox.warning(self, "Input Error", str(er))
                    return
            elif checkpoint_type == "time":
                try:
                    ci_m = int(self.checkpoint_interval_minutes_input.text())
                    if ci_m <= 0:
                        raise ValueError("Time interval must be positive.")
                    checkpoint_interval_minutes = ci_m
                except ValueError as er:
                    QMessageBox.warning(self, "Input Error", str(er))
                    return
            elif checkpoint_type == "batch":
                try:
                    ci_b = int(self.checkpoint_batch_interval_input.text())
                    if ci_b <= 0:
                        raise ValueError("Batch interval must be positive.")
                    checkpoint_batch_interval = ci_b
                except ValueError as er:
                    QMessageBox.warning(self, "Input Error", str(er))
                    return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        if self.pause_button:
            self.pause_button.setEnabled(True)
        if self.resume_button:
            self.resume_button.setEnabled(False)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()

        if not os.path.exists(os.path.dirname(output_model_path)):
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

        self.input_settings_group.setVisible(False)
        self.dataset_group.setVisible(False)
        self.training_group.setVisible(False)
        self.checkpoint_group.setVisible(False)
        self.progress_group.setVisible(True)
        self.control_group.setVisible(True)
        self.log_group.setVisible(True)

        if self.visualization_group:
            self.visualization_group.setVisible(False)
        if self.show_logs_button:
            self.show_logs_button.setVisible(True)
        if self.show_graphs_button:
            self.show_graphs_button.setVisible(True)
        if self.start_new_button:
            self.start_new_button.setVisible(False)

        self.init_ui_state = False

        try:
            random_seed = int(self.random_seed_input.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Random Seed must be an integer.")
            self.reset_to_initial_state()
            return

        if model_path and os.path.isfile(model_path):
            filters = None
            res_blocks = None
            inplace_relu = None
        else:
            try:
                filters = int(self.filters_input.text())
                res_blocks = int(self.res_blocks_input.text())
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Filter Amount and Residual Blocks must be integers.")
                self.reset_to_initial_state()
                return
            inplace_relu = self.inplace_checkbox.isChecked()

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
            model_path=model_path,
            automatic_batch_size=automatic_batch_size,
            checkpoint_type=checkpoint_type,
            checkpoint_interval_minutes=checkpoint_interval_minutes,
            checkpoint_batch_interval=checkpoint_batch_interval,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            output_model_path=output_model_path,
            num_workers=num_workers,
            random_seed=random_seed,
            filters=filters,
            res_blocks=res_blocks,
            inplace_relu=inplace_relu
        )

        if started:
            self.worker.batch_loss_update.connect(self.visualization.update_loss_plots)
            self.worker.batch_accuracy_update.connect(self.visualization.update_accuracy_plot)
            self.worker.val_loss_update.connect(self.visualization.update_validation_loss_plots)
            self.worker.epoch_accuracy_update.connect(self.visualization.update_validation_accuracy_plot)
            self.worker.task_finished.connect(self.on_training_finished)
            self.worker.progress_update.connect(self.update_progress)
        else:
            self.reset_to_initial_state()

        if not model_path:
            self.visualization.reset_visualization()

    def stop_training(self):
        self.stop_worker()
        self.reset_to_initial_state()

    def on_training_finished(self):
        if self.start_button:
            self.start_button.setEnabled(True)
        if self.stop_button:
            self.stop_button.setEnabled(False)
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)

        self.progress_bar.setFormat("Training Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        if self.start_new_button:
            self.start_new_button.setVisible(True)

    def reset_to_initial_state(self):
        self.input_settings_group.setVisible(True)
        self.dataset_group.setVisible(True)
        self.training_group.setVisible(True)
        self.checkpoint_group.setVisible(True)
        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)

        if self.visualization_group:
            self.visualization_group.setVisible(False)
        if self.start_new_button:
            self.start_new_button.setVisible(False)
        if self.show_logs_button:
            self.show_logs_button.setVisible(False)
        if self.show_graphs_button:
            self.show_graphs_button.setVisible(False)
        if self.show_logs_button:
            self.show_logs_button.setChecked(True)
        if self.show_graphs_button:
            self.show_graphs_button.setChecked(False)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()

        if self.start_button:
            self.start_button.setEnabled(True)
        if self.stop_button:
            self.stop_button.setEnabled(False)
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)

        self.init_ui_state = True

    def show_logs_view(self):
        super().show_logs_view()
        if self.visualization:
            self.visualization.update_visualization()

    def show_graphs_view(self):
        super().show_graphs_view()
        if self.visualization:
            self.visualization.update_visualization()