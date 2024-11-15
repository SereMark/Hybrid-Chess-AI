from PyQt5.QtWidgets import (
    QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton,
    QHBoxLayout, QFileDialog, QMessageBox, QCheckBox
)
import os
from src.self_play.self_play_visualization import SelfPlayVisualization
from src.self_play.self_play_worker import SelfPlayWorker
from src.base.base_tab import BaseTab

class SelfPlayTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = SelfPlayVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        self.model_output_group = self.create_model_output_group()
        self.parameters_group = self.create_parameters_group()
        control_buttons_layout = self.create_control_buttons()
        progress_layout = self.create_progress_layout()
        self.log_text_edit = self.create_log_text_edit()
        self.visualization_group = self.create_visualization_group()

        main_layout.addWidget(self.model_output_group)
        main_layout.addWidget(self.parameters_group)
        main_layout.addLayout(control_buttons_layout)
        main_layout.addLayout(progress_layout)
        main_layout.addWidget(self.log_text_edit)
        main_layout.addWidget(self.visualization_group)

        self.log_text_edit.setVisible(False)
        self.visualization_group.setVisible(False)

    def create_model_output_group(self):
        model_output_group = QGroupBox("Model and Output Settings")
        model_output_layout = QFormLayout()

        self.model_path_input = QLineEdit("models/saved_models/pre_trained_model.pth")
        self.output_dir_input = QLineEdit("data/self_play")
        self.checkpoint_path_input = QLineEdit("")

        model_browse_button = QPushButton("Browse")
        model_browse_button.clicked.connect(self.browse_model)
        output_dir_browse_button = QPushButton("Browse")
        output_dir_browse_button.clicked.connect(self.browse_output_dir)
        checkpoint_browse_button = QPushButton("Browse")
        checkpoint_browse_button.clicked.connect(self.browse_checkpoint)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(model_browse_button)

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_input)
        output_dir_layout.addWidget(output_dir_browse_button)

        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(self.checkpoint_path_input)
        checkpoint_layout.addWidget(checkpoint_browse_button)

        model_output_layout.addRow("Model Path:", model_layout)
        model_output_layout.addRow("Output Directory:", output_dir_layout)
        model_output_layout.addRow("Resume from Checkpoint:", checkpoint_layout)

        model_output_group.setLayout(model_output_layout)
        return model_output_group

    def create_parameters_group(self):
        parameters_group = QGroupBox("Self-Play Parameters")
        parameters_layout = QFormLayout()

        self.num_iterations_input = QLineEdit("10")
        self.num_games_per_iteration_input = QLineEdit("100")
        self.simulations_input = QLineEdit("800")
        self.c_puct_input = QLineEdit("1.4")
        self.temperature_input = QLineEdit("1.0")
        self.num_epochs_input = QLineEdit("5")
        self.batch_size_input = QLineEdit("128")
        self.automatic_batch_size_checkbox = QCheckBox("Automatic Batch Size")
        self.automatic_batch_size_checkbox.setChecked(False)
        self.automatic_batch_size_checkbox.toggled.connect(self.toggle_batch_size_input)
        self.num_threads_input = QLineEdit("4")
        self.random_seed_input = QLineEdit("42")

        parameters_layout.addRow("Number of Iterations:", self.num_iterations_input)
        parameters_layout.addRow("Games per Iteration:", self.num_games_per_iteration_input)
        parameters_layout.addRow("Simulations per Move:", self.simulations_input)
        parameters_layout.addRow("c_puct:", self.c_puct_input)
        parameters_layout.addRow("Temperature:", self.temperature_input)
        parameters_layout.addRow("Training Epochs per Iteration:", self.num_epochs_input)
        parameters_layout.addRow("Batch Size:", self.batch_size_input)
        parameters_layout.addRow(self.automatic_batch_size_checkbox)
        parameters_layout.addRow("Number of Threads:", self.num_threads_input)
        parameters_layout.addRow("Random Seed:", self.random_seed_input)

        parameters_group.setLayout(parameters_layout)
        return parameters_group

    def create_control_buttons(self):
        layout = QHBoxLayout()
        self.start_button = QPushButton("Start Self-Play")
        self.pause_button = QPushButton("Pause Training")
        self.resume_button = QPushButton("Resume Training")
        self.stop_button = QPushButton("Stop Self-Play")
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.resume_button)
        layout.addWidget(self.stop_button)
        layout.addStretch()
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_self_play)
        self.pause_button.clicked.connect(self.pause_training)
        self.resume_button.clicked.connect(self.resume_training)
        self.stop_button.clicked.connect(self.stop_self_play)
        return layout

    def create_visualization_group(self):
        visualization_group = QGroupBox("Self-Play Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        visualization_group.setLayout(vis_layout)
        return visualization_group

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", self.model_path_input.text(), "PyTorch Files (*.pth *.pt)"
        )
        if file_path:
            self.model_path_input.setText(file_path)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir_input.text()
        )
        if dir_path:
            self.output_dir_input.setText(dir_path)

    def browse_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", self.checkpoint_path_input.text(), "PyTorch Files (*.pth *.pt)"
        )
        if file_path:
            self.checkpoint_path_input.setText(file_path)

    def toggle_batch_size_input(self, checked):
        self.batch_size_input.setEnabled(not checked)

    def start_self_play(self):
        try:
            num_iterations = int(self.num_iterations_input.text())
            num_games_per_iteration = int(self.num_games_per_iteration_input.text())
            simulations = int(self.simulations_input.text())
            c_puct = float(self.c_puct_input.text())
            temperature = float(self.temperature_input.text())
            num_epochs = int(self.num_epochs_input.text())
            num_threads = int(self.num_threads_input.text())
            random_seed = int(self.random_seed_input.text())

            if self.automatic_batch_size_checkbox.isChecked():
                batch_size = None
            else:
                batch_size = int(self.batch_size_input.text())
                if batch_size <= 0:
                    raise ValueError("Batch Size must be a positive integer.")

            if any(v <= 0 for v in [
                num_iterations, num_games_per_iteration, simulations,
                c_puct, temperature, num_epochs, num_threads
            ]):
                raise ValueError("All numerical parameters must be positive.")
        except ValueError as e:
            QMessageBox.warning(
                self, "Input Error", f"Please enter valid and positive parameters.\n{str(e)}"
            )
            return

        model_path = self.model_path_input.text()
        output_dir = self.output_dir_input.text()
        checkpoint_path = self.checkpoint_path_input.text() if self.checkpoint_path_input.text() else None
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Model file does not exist.")
            return
        os.makedirs(output_dir, exist_ok=True)

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()

        self.visualization.reset_visualization()

        self.model_output_group.setVisible(False)
        self.parameters_group.setVisible(False)
        self.log_text_edit.setVisible(True)
        self.visualization_group.setVisible(True)

        started = self.start_worker(
            SelfPlayWorker,
            model_path=model_path,
            output_dir=output_dir,
            num_iterations=num_iterations,
            num_games_per_iteration=num_games_per_iteration,
            simulations=simulations,
            c_puct=c_puct,
            temperature=temperature,
            num_epochs=num_epochs,
            batch_size=batch_size,
            automatic_batch_size=self.automatic_batch_size_checkbox.isChecked(),
            num_threads=num_threads,
            checkpoint_path=checkpoint_path,
            random_seed=random_seed
        )
        if started:
            self.worker.stats_update.connect(self.visualization.update_stats)
            self.worker.training_finished.connect(self.on_self_play_finished)
            self.worker.paused.connect(self.on_worker_paused)
        else:
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.model_output_group.setVisible(True)
            self.parameters_group.setVisible(True)
            self.log_text_edit.setVisible(False)
            self.visualization_group.setVisible(False)

    def pause_training(self):
        if self.worker:
            self.worker.pause()

    def resume_training(self):
        if self.worker:
            self.worker.resume()

    def stop_self_play(self):
        self.stop_worker()
        self.log_message("Stopping self-play...")
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.model_output_group.setVisible(True)
        self.parameters_group.setVisible(True)

    def on_worker_paused(self, is_paused):
        self.pause_button.setEnabled(not is_paused)
        self.resume_button.setEnabled(is_paused)

    def on_self_play_finished(self):
        self.log_message("Self-play process has been completed.")
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Self-Play Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.model_output_group.setVisible(True)
        self.parameters_group.setVisible(True)