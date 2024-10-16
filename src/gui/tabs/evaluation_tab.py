from PyQt5.QtWidgets import (
    QMessageBox, QVBoxLayout, QFileDialog, QProgressBar, QTextEdit,
    QWidget, QLabel, QLineEdit, QHBoxLayout, QPushButton, QGroupBox, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QWaitCondition, QMutex
from src.neural_network.evaluate import ModelEvaluator
from src.gui.visualizations.evaluation_visualization import EvaluationVisualization
import os
import numpy as np

class EvaluationWorker(QThread):
    log_update = pyqtSignal(str)
    metrics_update = pyqtSignal(float, dict, dict, np.ndarray, float, list)
    progress_update = pyqtSignal(int)
    time_left_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_path, dataset_indices_path):
        super().__init__()
        self.model_path = model_path
        self.dataset_indices_path = dataset_indices_path
        self._is_stopped = False
        self._pause_condition = QWaitCondition()
        self._mutex = QMutex()

    def run(self):
        try:
            evaluator = ModelEvaluator(
                model_path=self.model_path,
                dataset_indices_path=self.dataset_indices_path,
                log_fn=self.log_update.emit,
                metrics_fn=self.metrics_update.emit,
                progress_fn=self.progress_update.emit,
                time_left_fn=self.time_left_update.emit,
                stop_fn=lambda: self._is_stopped
            )
            evaluator.evaluate_model()
            if not self._is_stopped:
                self.log_update.emit("Evaluation completed successfully.")
        except Exception as e:
            self.log_update.emit(f"Error during evaluation: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._mutex.lock()
        self._is_stopped = True
        self._mutex.unlock()
        self.log_update.emit("Evaluation stopped...")

class EvaluationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = EvaluationVisualization()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        layout.addLayout(self.create_input_layout(
            "Model Path:", "models/saved_models/final_model.pth", "model_browse_button", self.browse_model
        ))
        layout.addLayout(self.create_input_layout(
            "Evaluation Dataset:", "data/processed/test_indices.npy", "dataset_browse_button", self.browse_dataset
        ))

        self.compare_baseline_checkbox = QCheckBox("Compare with Baseline")
        self.compare_baseline_checkbox.setChecked(False)
        layout.addWidget(self.compare_baseline_checkbox)

        self.start_button = QPushButton("Start Evaluation")
        self.stop_button = QPushButton("Stop Evaluation")
        self.stop_button.setEnabled(False)
        layout.addLayout(self.create_buttons_layout())

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        layout.addWidget(self.progress_bar)

        self.remaining_time_label = QLabel("Time Left: Calculating...")
        layout.addWidget(self.remaining_time_label)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        layout.addWidget(self.log_text_edit)

        vis_group = QGroupBox("Evaluation Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

    def create_input_layout(self, label_text, default_value, button_name, button_callback):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        input_field = QLineEdit(default_value)
        button = QPushButton("Browse")
        button.clicked.connect(button_callback)
        setattr(self, button_name, button)
        setattr(self, f"{button_name}_input", input_field)
        layout.addWidget(label)
        layout.addWidget(input_field)
        layout.addWidget(button)
        return layout

    def create_buttons_layout(self):
        layout = QHBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.start_button.clicked.connect(self.start_evaluation)
        self.stop_button.clicked.connect(self.stop_evaluation)
        return layout

    def browse_model(self):
        self.browse_file("model_browse_button_input", "Select Model File", "PyTorch Model (*.pth *.pt)")

    def browse_dataset(self):
        self.browse_file("dataset_browse_button_input", "Select Evaluation Dataset Indices File", "NumPy Array (*.npy)")

    def browse_file(self, input_attr, title, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(self, title, getattr(self, input_attr).text(), file_filter)
        if file_path:
            getattr(self, input_attr).setText(file_path)

    def start_evaluation(self):
        model_path = self.model_browse_button_input.text()
        dataset_indices_path = self.dataset_browse_button_input.text()
        if not all(os.path.exists(p) for p in [model_path, dataset_indices_path]):
            QMessageBox.warning(self, "Error", "Model or dataset indices file does not exist.")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting Evaluation...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()

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