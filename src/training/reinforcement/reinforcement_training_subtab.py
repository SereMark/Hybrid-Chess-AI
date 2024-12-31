from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QGridLayout, QLineEdit, QPushButton, QLabel, QCheckBox, QComboBox, QMessageBox, QHBoxLayout
from src.training.reinforcement.reinforcement_training_worker import ReinforcementWorker
from src.training.reinforcement.reinforcement_training_visualization import ReinforcementVisualization
from src.base.base_tab import BaseTab
import os

class ReinforcementTrainingSubTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = ReinforcementVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.setup_subtab(
            main_layout,
            "Train a chess model through self-play reinforcement learning.",
            "Self-Play Progress",
            "Self-Play Logs",
            "Self-Play Visualization",
            self.visualization,
            {
                "start_text": "Start Self-Play",
                "stop_text": "Stop Self-Play",
                "start_callback": self.start_self_play,
                "stop_callback": self.stop_self_play,
                "pause_text": "Pause Training",
                "resume_text": "Resume Training",
                "pause_callback": self.pause_worker,
                "resume_callback": self.resume_worker,
                "start_new_callback": self.reset_to_initial_state
            },
            "Start New"
        )
        self.model_output_group = self.create_model_output_group()
        self.parameters_group = self.create_parameters_group()
        self.checkpoint_group = self.create_checkpoint_group()
        self.layout().insertWidget(1, self.model_output_group)
        self.layout().insertWidget(2, self.parameters_group)
        self.layout().insertWidget(3, self.checkpoint_group)
        self.setup_batch_size_control(self.automatic_batch_size_checkbox, self.batch_size_input)
        interval_widgets = {
            'iteration': self.iteration_interval_widget,
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

    def create_model_output_group(self):
        group = QGroupBox("Output Settings")
        layout = QGridLayout()
        label1 = QLabel("Model Path:")
        self.model_path_input = QLineEdit("models/saved_models/pre_trained_model.pth")
        model_browse_button = QPushButton("Browse")
        model_browse_button.clicked.connect(lambda: self.browse_file(self.model_path_input, "Select Model File", "PyTorch Files (*.pth *.pt)"))
        layout.addWidget(label1, 0, 0)
        layout.addLayout(self.create_browse_layout(self.model_path_input, model_browse_button), 0, 1, 1, 3)
        group.setLayout(layout)
        return group

    def create_parameters_group(self):
        group = QGroupBox("Self-Play Parameters")
        layout = QGridLayout()
        label1 = QLabel("Number of Iterations:")
        self.num_iterations_input = QLineEdit("10")
        label2 = QLabel("Games per Iteration:")
        self.num_games_per_iteration_input = QLineEdit("100")
        label3 = QLabel("Simulations per Move:")
        self.simulations_input = QLineEdit("800")
        label4 = QLabel("c_puct:")
        self.c_puct_input = QLineEdit("1.4")
        label5 = QLabel("Temperature:")
        self.temperature_input = QLineEdit("1.0")
        label6 = QLabel("Training Epochs per Iteration:")
        self.num_epochs_input = QLineEdit("5")
        label7 = QLabel("Batch Size:")
        self.batch_size_input = QLineEdit("128")
        self.automatic_batch_size_checkbox = QCheckBox("Automatic Batch Size")
        label8 = QLabel("Number of Threads:")
        self.num_threads_input = QLineEdit("4")
        label9 = QLabel("Random Seed:")
        self.random_seed_input = QLineEdit("42")
        label10 = QLabel("Optimizer Type:")
        self.optimizer_type_combo = QComboBox()
        self.optimizer_type_combo.addItems(["Adam", "AdamW", "SGD"])
        label11 = QLabel("Learning Rate:")
        self.learning_rate_input = QLineEdit("0.0005")
        label12 = QLabel("Weight Decay:")
        self.weight_decay_input = QLineEdit("0.0001")
        label13 = QLabel("Scheduler Type:")
        self.scheduler_type_combo = QComboBox()
        self.scheduler_type_combo.addItems(["None", "StepLR", "CosineAnnealing", "CosineAnnealingWarmRestarts"])
        self.scheduler_type_combo.setCurrentText("CosineAnnealing")
        layout.addWidget(label1, 0, 0)
        layout.addWidget(self.num_iterations_input, 0, 1)
        layout.addWidget(label2, 0, 2)
        layout.addWidget(self.num_games_per_iteration_input, 0, 3)
        layout.addWidget(label3, 1, 0)
        layout.addWidget(self.simulations_input, 1, 1)
        layout.addWidget(label4, 1, 2)
        layout.addWidget(self.c_puct_input, 1, 3)
        layout.addWidget(label5, 2, 0)
        layout.addWidget(self.temperature_input, 2, 1)
        layout.addWidget(label6, 2, 2)
        layout.addWidget(self.num_epochs_input, 2, 3)
        layout.addWidget(label7, 3, 0)
        layout.addWidget(self.batch_size_input, 3, 1)
        layout.addWidget(self.automatic_batch_size_checkbox, 3, 2)
        layout.addWidget(label8, 4, 0)
        layout.addWidget(self.num_threads_input, 4, 1)
        layout.addWidget(label9, 4, 2)
        layout.addWidget(self.random_seed_input, 4, 3)
        layout.addWidget(label10, 5, 0)
        layout.addWidget(self.optimizer_type_combo, 5, 1)
        layout.addWidget(label11, 5, 2)
        layout.addWidget(self.learning_rate_input, 5, 3)
        layout.addWidget(label12, 6, 0)
        layout.addWidget(self.weight_decay_input, 6, 1)
        layout.addWidget(label13, 6, 2)
        layout.addWidget(self.scheduler_type_combo, 6, 3)
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
        self.checkpoint_type_combo.addItems(["Iteration", "Epoch", "Time", "Batch"])
        row1.addWidget(self.checkpoint_type_combo)
        row1.addStretch()
        layout.addLayout(row1)
        self.checkpoint_interval_input = QLineEdit("1")
        self.checkpoint_interval_minutes_input = QLineEdit("30")
        self.checkpoint_batch_interval_input = QLineEdit("2000")
        self.iteration_interval_widget = self.create_interval_widget("Every", self.checkpoint_interval_input, "iterations")
        self.epoch_interval_widget = self.create_interval_widget("Every", self.checkpoint_interval_input, "epochs")
        self.time_interval_widget = self.create_interval_widget("Every", self.checkpoint_interval_minutes_input, "minutes")
        self.batch_interval_widget = self.create_interval_widget("Every", self.checkpoint_batch_interval_input, "batches")
        layout.addWidget(self.iteration_interval_widget)
        layout.addWidget(self.epoch_interval_widget)
        layout.addWidget(self.time_interval_widget)
        layout.addWidget(self.batch_interval_widget)
        row2 = QHBoxLayout()
        label2 = QLabel("Resume from Checkpoint:")
        row2.addWidget(label2)
        self.checkpoint_path_input = QLineEdit("")
        checkpoint_browse_button = QPushButton("Browse")
        checkpoint_browse_button.clicked.connect(lambda: self.browse_file(self.checkpoint_path_input, "Select Checkpoint File", "PyTorch Files (*.pth *.pt)"))
        row2.addLayout(self.create_browse_layout(self.checkpoint_path_input, checkpoint_browse_button))
        layout.addLayout(row2)
        group.setLayout(layout)
        return group

    def start_self_play(self):
        try:
            ni = int(self.num_iterations_input.text())
            ng = int(self.num_games_per_iteration_input.text())
            sim = int(self.simulations_input.text())
            cp = float(self.c_puct_input.text())
            temp = float(self.temperature_input.text())
            ne = int(self.num_epochs_input.text())
            nt = int(self.num_threads_input.text())
            if self.automatic_batch_size_checkbox.isChecked():
                batch_size = None
            else:
                bs = int(self.batch_size_input.text())
                if bs <= 0:
                    raise ValueError("Batch Size must be positive.")
                batch_size = bs
            if any(v <= 0 for v in [ni, ng, sim, cp, temp, ne, nt]):
                raise ValueError("All parameters must be positive.")
            opt_type = self.optimizer_type_combo.currentText().lower()
            lr = float(self.learning_rate_input.text())
            wd = float(self.weight_decay_input.text())
            sched_type = self.scheduler_type_combo.currentText().lower()
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
            return
        model_path = self.model_path_input.text()
        checkpoint_path = self.checkpoint_path_input.text() if self.checkpoint_path_input.text() else None
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Model file does not exist.")
            return
        save_checkpoints = self.save_checkpoints_checkbox.isChecked()
        ctype = self.checkpoint_type_combo.currentText().lower()
        ci = None
        ci_m = None
        ci_b = None
        if save_checkpoints:
            if ctype in ["iteration", "epoch"]:
                try:
                    t = int(self.checkpoint_interval_input.text())
                    if t <= 0:
                        raise ValueError("Interval must be positive.")
                    ci = t
                except ValueError as er:
                    QMessageBox.warning(self, "Input Error", str(er))
                    return
            elif ctype == "time":
                try:
                    t = int(self.checkpoint_interval_minutes_input.text())
                    if t <= 0:
                        raise ValueError("Time interval must be positive.")
                    ci_m = t
                except ValueError as er:
                    QMessageBox.warning(self, "Input Error", str(er))
                    return
            elif ctype == "batch":
                try:
                    t = int(self.checkpoint_batch_interval_input.text())
                    if t <= 0:
                        raise ValueError("Batch interval must be positive.")
                    ci_b = t
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
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model_output_group.setVisible(False)
        self.parameters_group.setVisible(False)
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
        started = self.start_worker(
            ReinforcementWorker,
            model_path=model_path,
            num_iterations=ni,
            num_games_per_iteration=ng,
            simulations=sim,
            c_puct=cp,
            temperature=temp,
            num_epochs=ne,
            batch_size=batch_size,
            automatic_batch_size=self.automatic_batch_size_checkbox.isChecked(),
            num_threads=nt,
            checkpoint_path=checkpoint_path,
            random_seed=int(self.random_seed_input.text()),
            save_checkpoints=save_checkpoints,
            checkpoint_interval=ci,
            checkpoint_type=ctype,
            checkpoint_interval_minutes=ci_m,
            checkpoint_batch_interval=ci_b,
            optimizer_type=opt_type,
            learning_rate=lr,
            weight_decay=wd,
            scheduler_type=sched_type
        )
        if started:
            self.worker.stats_update.connect(self.visualization.update_stats)
            self.worker.task_finished.connect(self.on_self_play_finished)
            self.worker.progress_update.connect(self.update_progress)
        else:
            self.reset_to_initial_state()

    def stop_self_play(self):
        self.stop_worker()
        self.reset_to_initial_state()

    def on_self_play_finished(self):
        if self.start_button:
            self.start_button.setEnabled(True)
        if self.stop_button:
            self.stop_button.setEnabled(False)
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)
        self.progress_bar.setFormat("Self-Play Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        if self.start_new_button:
            self.start_new_button.setVisible(True)

    def reset_to_initial_state(self):
        self.model_output_group.setVisible(True)
        self.parameters_group.setVisible(True)
        self.checkpoint_group.setVisible(True)
        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)
        if self.visualization_group:
            self.visualization_group.setVisible(False)
        self.controls_group = getattr(self, 'control_group', None)
        if self.controls_group:
            self.controls_group.setVisible(True)
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