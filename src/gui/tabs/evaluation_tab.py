from PyQt5.QtWidgets import QMessageBox, QVBoxLayout, QFileDialog, QProgressBar, QTextEdit, QWidget, QLabel, QLineEdit, QHBoxLayout, QPushButton, QGroupBox
from PyQt5.QtCore import QThread, pyqtSignal
from src.neural_network.evaluate import ModelEvaluator
from src.gui.visualizations.evaluation_visualization import EvaluationVisualization
import os

class EvaluationWorker(QThread):
    log_update = pyqtSignal(str)
    metrics_update = pyqtSignal(float, dict, dict)
    finished = pyqtSignal()

    def __init__(self, model_path, dataset_path):
        super().__init__()
        self.model_path, self.dataset_path, self._is_stopped = model_path, dataset_path, False

    def run(self):
        try:
            evaluator = ModelEvaluator(
                model_path=self.model_path, dataset_path=self.dataset_path,
                log_fn=self.log_update.emit, metrics_fn=self.metrics_update.emit,
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
        self._is_stopped = True
        self.log_update.emit("Evaluation stopped...")

class EvaluationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker, self.visualization = None, EvaluationVisualization()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        layout.addLayout(self.create_input_layout("Model Path:", "models/saved_models/final_model.pth", "model_browse_button", self.browse_model))
        layout.addLayout(self.create_input_layout("Evaluation Dataset:", "data/processed/val_indices.npy", "dataset_browse_button", self.browse_dataset))

        self.start_button, self.stop_button = QPushButton("Start Evaluation"), QPushButton("Stop Evaluation")
        self.stop_button.setEnabled(False)
        layout.addLayout(self.create_buttons_layout())

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        layout.addWidget(self.progress_bar)

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
        self.browse_file("dataset_browse_button_input", "Select Evaluation Dataset File", "NumPy Array (*.npy);;HDF5 File (*.h5)")

    def browse_file(self, input_attr, title, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(self, title, getattr(self, input_attr).text(), file_filter)
        if file_path:
            getattr(self, input_attr).setText(file_path)

    def start_evaluation(self):
        model_path, dataset_path = self.model_browse_button_input.text(), self.dataset_browse_button_input.text()
        if not all(os.path.exists(p) for p in [model_path, dataset_path]):
            QMessageBox.warning(self, "Error", "Model or dataset file does not exist.")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting Evaluation...")
        self.log_text_edit.clear()

        self.worker = EvaluationWorker(model_path, dataset_path)
        self.worker.log_update.connect(self.update_log)
        self.worker.metrics_update.connect(self.visualization.update_metrics_visualization)
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
        if "Progress:" in message:
            try:
                progress = int(message.split("Progress:")[1].strip('%').strip())
                self.progress_bar.setValue(progress)
                self.progress_bar.setFormat(f"Progress: {progress}%")
            except (IndexError, ValueError):
                pass
        elif "Accuracy:" in message:
            try:
                accuracy = float(message.split("Accuracy:")[1].strip('%').strip())
                self.progress_bar.setFormat(f"Accuracy: {accuracy}%")
            except (IndexError, ValueError):
                pass

    def on_evaluation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Evaluation Finished")
        self.log_text_edit.append("Evaluation process finished.")