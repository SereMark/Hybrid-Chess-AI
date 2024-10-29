import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton,
    QFileDialog, QHBoxLayout, QProgressBar, QLabel, QTextEdit, QMessageBox
)
from src.gui.visualizations.evaluation_visualization import EvaluationVisualization
from src.gui.workers.evaluation_worker import EvaluationWorker

class EvaluationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = EvaluationVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        settings_group = QGroupBox("Evaluation Settings")
        settings_layout = QFormLayout()

        self.model_path_input = QLineEdit("models/saved_models/final_model.pth")
        self.evaluation_dataset_indices_input = QLineEdit("data/processed/test_indices.npy")

        model_browse_button = QPushButton("Browse")
        model_browse_button.clicked.connect(lambda: self.browse_model(self.model_path_input))
        dataset_browse_button = QPushButton("Browse")
        dataset_browse_button.clicked.connect(lambda: self.browse_dataset(self.evaluation_dataset_indices_input))

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(model_browse_button)

        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.evaluation_dataset_indices_input)
        dataset_layout.addWidget(dataset_browse_button)

        settings_layout.addRow("Model Path:", model_layout)
        settings_layout.addRow("Evaluation Dataset Indices:", dataset_layout)

        settings_group.setLayout(settings_layout)

        control_buttons_layout = self.create_buttons_layout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label = QLabel("Time Left: Calculating...")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        main_layout.addWidget(settings_group)
        main_layout.addLayout(control_buttons_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.remaining_time_label)
        main_layout.addWidget(self.log_text_edit)
        main_layout.addWidget(self.create_visualization_group())

    def create_buttons_layout(self):
        layout = QHBoxLayout()
        self.start_button = QPushButton("Start Evaluation")
        self.stop_button = QPushButton("Stop Evaluation")
        self.stop_button.setEnabled(False)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addStretch()
        self.start_button.clicked.connect(self.start_evaluation)
        self.stop_button.clicked.connect(self.stop_evaluation)
        return layout

    def create_visualization_group(self):
        visualization_group = QGroupBox("Evaluation Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        visualization_group.setLayout(vis_layout)
        return visualization_group

    def browse_model(self, input_field):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", input_field.text(), "PyTorch Model (*.pth *.pt)")
        if file_path:
            input_field.setText(file_path)

    def browse_dataset(self, input_field):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Evaluation Dataset Indices File", input_field.text(), "NumPy Array (*.npy)")
        if file_path:
            input_field.setText(file_path)

    def start_evaluation(self):
        model_path = self.model_path_input.text()
        dataset_indices_path = self.evaluation_dataset_indices_input.text()

        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Model file does not exist.")
            return
        if not os.path.exists(dataset_indices_path):
            QMessageBox.warning(self, "Error", "Dataset indices file does not exist.")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting Evaluation...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()

        self.worker = EvaluationWorker(model_path, dataset_indices_path)
        self.worker.log_update.connect(self.update_log)
        self.worker.metrics_update.connect(self.visualization.update_metrics_visualization)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.time_left_update.connect(self.update_time_left)
        self.worker.finished.connect(self.on_evaluation_finished)
        self.worker.start()

    def stop_evaluation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.log_text_edit.append("Stopping evaluation...")

    def update_log(self, message):
        self.log_text_edit.append(message)

    def on_evaluation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Evaluation Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_text_edit.append("Evaluation process finished.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Progress: {value}%")

    def update_time_left(self, time_left_str):
        self.remaining_time_label.setText(f"Time Left: {time_left_str}")