import os, numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFileDialog, QProgressBar, QTextEdit,
    QLabel, QLineEdit, QHBoxLayout, QPushButton, QGroupBox, QCheckBox, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from src.neural_network.evaluate import ModelEvaluator
from src.gui.visualizations.evaluation_visualization import EvaluationVisualization


class EvaluationWorker(QThread):
    log_update = pyqtSignal(str)
    metrics_update = pyqtSignal(float, float, dict, dict, np.ndarray, list)
    progress_update = pyqtSignal(int)
    time_left_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_path, dataset_indices_path, compare_baseline=False):
        super().__init__()
        self.model_path = model_path
        self.dataset_indices_path = dataset_indices_path
        self.compare_baseline = compare_baseline
        self._is_stopped = False

    def run(self):
        try:
            evaluator = ModelEvaluator(
                model_path=self.model_path,
                dataset_indices_path=self.dataset_indices_path,
                log_fn=self.log_update.emit,
                metrics_fn=self.metrics_update.emit,
                progress_fn=self.progress_update.emit,
                time_left_fn=self.time_left_update.emit,
                stop_fn=lambda: self._is_stopped,
                compare_baseline=self.compare_baseline
            )
            evaluator.evaluate_model()
            if not self._is_stopped:
                self.log_update.emit("Evaluation completed successfully.")
        except Exception as e:
            self.log_update.emit(f"Error during evaluation: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._is_stopped = True
        self.log_update.emit("Evaluation stopped by user.")


class EvaluationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = EvaluationVisualization()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        model_layout = self.create_input_layout(
            "Model Path:", "models/saved_models/final_model.pth", self.browse_model
        )
        dataset_layout = self.create_input_layout(
            "Evaluation Dataset Indices:", "data/processed/test_indices.npy", self.browse_dataset
        )

        self.compare_baseline_checkbox = QCheckBox("Compare with Baseline")
        self.compare_baseline_checkbox.setChecked(False)

        control_buttons_layout = self.create_buttons_layout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")

        self.remaining_time_label = QLabel("Time Left: Calculating...")

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        layout.addLayout(model_layout)
        layout.addLayout(dataset_layout)
        layout.addWidget(self.compare_baseline_checkbox)
        layout.addLayout(control_buttons_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.remaining_time_label)
        layout.addWidget(self.log_text_edit)
        layout.addWidget(self.create_visualization_group())

    def create_input_layout(self, label_text, default_value, browse_callback):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        input_field = QLineEdit(default_value)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(lambda: browse_callback(input_field))
        layout.addWidget(label)
        layout.addWidget(input_field)
        layout.addWidget(browse_button)
        setattr(self, label_text.strip(':').lower().replace(' ', '_') + '_input', input_field)
        return layout

    def create_buttons_layout(self):
        layout = QHBoxLayout()
        self.start_button = QPushButton("Start Evaluation")
        self.stop_button = QPushButton("Stop Evaluation")
        self.stop_button.setEnabled(False)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
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
        compare_baseline = self.compare_baseline_checkbox.isChecked()

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

        self.worker = EvaluationWorker(model_path, dataset_indices_path, compare_baseline)
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