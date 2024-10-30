from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton,
    QHBoxLayout, QLabel, QProgressBar, QTextEdit, QFileDialog, QMessageBox, QCheckBox
)
import os, threading
from src.gui.visualizations.self_play_visualization import SelfPlayVisualization
from src.gui.workers.self_play_worker import SelfPlayWorker


class SelfPlayTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.worker = None
        self.visualization = SelfPlayVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        model_output_group = QGroupBox("Model and Output Settings")
        model_output_layout = QFormLayout()

        self.model_path_input = QLineEdit("models/saved_models/best_model.pth")
        self.output_dir_input = QLineEdit("data/self_play")

        model_browse_button = QPushButton("Browse")
        model_browse_button.clicked.connect(self.browse_model)
        output_dir_browse_button = QPushButton("Browse")
        output_dir_browse_button.clicked.connect(self.browse_output_dir)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(model_browse_button)

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_input)
        output_dir_layout.addWidget(output_dir_browse_button)

        model_output_layout.addRow("Model Path:", model_layout)
        model_output_layout.addRow("Output Directory:", output_dir_layout)

        model_output_group.setLayout(model_output_layout)

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

        parameters_layout.addRow("Number of Iterations:", self.num_iterations_input)
        parameters_layout.addRow("Games per Iteration:", self.num_games_per_iteration_input)
        parameters_layout.addRow("Simulations per Move:", self.simulations_input)
        parameters_layout.addRow("c_puct:", self.c_puct_input)
        parameters_layout.addRow("Temperature:", self.temperature_input)
        parameters_layout.addRow("Training Epochs per Iteration:", self.num_epochs_input)
        parameters_layout.addRow("Batch Size:", self.batch_size_input)
        parameters_layout.addRow(self.automatic_batch_size_checkbox)
        parameters_layout.addRow("Number of Threads:", self.num_threads_input)

        parameters_group.setLayout(parameters_layout)

        control_buttons_layout = self.create_buttons_layout()

        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label = QLabel("Time Left: Calculating...")
        self.remaining_time_label.setAlignment(Qt.AlignCenter)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.remaining_time_label)

        main_layout.addWidget(model_output_group)
        main_layout.addWidget(parameters_group)
        main_layout.addLayout(control_buttons_layout)
        main_layout.addLayout(progress_layout)
        main_layout.addWidget(self.create_log_text_edit())
        main_layout.addWidget(self.create_visualization_group())

    def create_buttons_layout(self):
        layout = QHBoxLayout()
        self.start_button = QPushButton("Start Self-Play")
        self.stop_button = QPushButton("Stop Self-Play")
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addStretch()
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_self_play)
        self.stop_button.clicked.connect(self.stop_self_play)
        return layout

    def create_visualization_group(self):
        visualization_group = QGroupBox("Self-Play Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        visualization_group.setLayout(vis_layout)
        return visualization_group

    def create_log_text_edit(self):
        log_edit = QTextEdit()
        log_edit.setReadOnly(True)
        self.log_text_edit_widget = log_edit
        return log_edit

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

            if self.automatic_batch_size_checkbox.isChecked():
                batch_size = None
            else:
                batch_size = int(self.batch_size_input.text())
                if batch_size <= 0:
                    raise ValueError

            if any(v <= 0 for v in [
                num_iterations, num_games_per_iteration, simulations,
                c_puct, temperature, num_epochs, num_threads
            ]):
                raise ValueError
        except ValueError:
            QMessageBox.warning(
                self, "Input Error", "Please enter valid and positive parameters."
            )
            return

        model_path = self.model_path_input.text()
        output_dir = self.output_dir_input.text()
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Model file does not exist.")
            return
        os.makedirs(output_dir, exist_ok=True)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit_widget.clear()

        self.visualization.reset_visualization()

        self.thread = QThread()
        self.worker = SelfPlayWorker(
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
            stop_event=threading.Event()
        )
        self.worker.moveToThread(self.thread)
        self.worker.log_update.connect(self.log_text_edit_widget.append)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.time_left_update.connect(self.update_time_left)
        self.worker.stats_update.connect(self.visualization.update_stats)
        self.worker.finished.connect(self.on_self_play_finished)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def stop_self_play(self):
        if self.worker and self.thread.isRunning():
            self.worker.stop_event.set()
            self.log_text_edit_widget.append("Stopping self-play...")
            self.stop_button.setEnabled(False)

    def on_self_play_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Self-Play Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_text_edit_widget.append("Self-play process finished.")
        self.thread.quit()
        self.thread.wait()
        self.worker = None
        self.thread = None

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Progress: {value}%")

    def update_time_left(self, time_left_str):
        self.remaining_time_label.setText(f"Time Left: {time_left_str}")