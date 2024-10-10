from PyQt5.QtWidgets import QMessageBox, QVBoxLayout, QFileDialog, QGroupBox
from PyQt5.QtCore import QThread, pyqtSignal
from src.neural_network.evaluate import ModelEvaluator
from src.common.base_tab import BaseTab
from src.common.common_widgets import create_labeled_input
from src.gui.visualizations.evaluation_visualization import EvaluationVisualization
import os

class EvaluationWorker(QThread):
    log_update = pyqtSignal(str)
    metrics_update = pyqtSignal(float, dict, dict)
    finished = pyqtSignal()

    def __init__(self, model_path, dataset_path):
        super().__init__()
        self.model_path = model_path
        self.dataset_path = dataset_path
        self._is_stopped = False

    def run(self):
        try:
            evaluator = ModelEvaluator(
                model_path=self.model_path,
                dataset_path=self.dataset_path,
                log_fn=self.log_update.emit,
                metrics_fn=self.metrics_update.emit,
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

class EvaluationTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = EvaluationVisualization()
        self.init_specific_ui()

    def init_specific_ui(self):
        self.model_label, self.model_path_input, self.model_browse_button, model_layout = create_labeled_input(
            "Model Path:", "models/saved_models/final_model.pth", "Browse", "PyTorch Model (*.pth *.pt)"
        )
        self.dataset_label, self.dataset_path_input, self.dataset_browse_button, dataset_layout = create_labeled_input(
            "Evaluation Dataset:", "data/processed/val_indices.npy", "Browse", "NumPy Array (*.npy);;HDF5 File (*.h5)"
        )
    
        self.config_layout.addWidget(self.model_label)
        self.config_layout.addLayout(model_layout)
        self.config_layout.addWidget(self.dataset_label)
        self.config_layout.addLayout(dataset_layout)
    
        self.model_browse_button.clicked.connect(lambda: self.browse_path("model_path", "PyTorch Model (*.pth *.pt)"))
        self.dataset_browse_button.clicked.connect(lambda: self.browse_path("dataset_path", "NumPy Array (*.npy);;HDF5 File (*.h5)"))
    
        self.buttons["start"].setText("Start Evaluation")
        self.buttons["stop"].setText("Stop Evaluation")
        self.buttons["pause"].setVisible(False)
        self.buttons["resume"].setVisible(False)
    
        self.buttons["start"].clicked.connect(self.start_evaluation)
        self.buttons["stop"].clicked.connect(self.stop_evaluation)
    
        self.visualization_group = QGroupBox("Evaluation Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        self.visualization_group.setLayout(vis_layout)
        self.layout.addWidget(self.visualization_group)
    
    def browse_path(self, key, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select {key.replace('_', ' ').title()} File", getattr(self, f"{key}_input").text(), filter=file_filter)
        if file_path:
            getattr(self, f"{key}_input").setText(file_path)
    
    def start_evaluation(self):
        paths = {
            'model_path': self.model_path_input.text(),
            'dataset_path': self.dataset_path_input.text()
        }
        if not all(os.path.exists(path) for path in paths.values()):
            QMessageBox.warning(self, "Error", "Model or dataset file does not exist.")
            return
    
        self.buttons['start'].setEnabled(False)
        self.buttons['stop'].setEnabled(True)
    
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting Evaluation...")
        self.log_text_edit.clear()
    
        self.worker = EvaluationWorker(**paths)
        self.worker.log_update.connect(self.update_log)
        self.worker.metrics_update.connect(self.visualization.update_metrics_visualization)
        self.worker.finished.connect(self.on_evaluation_finished)
        self.worker.start()
    
    def stop_evaluation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.buttons['stop'].setEnabled(False)
            self.log_signal.emit("Stopping evaluation...")
    
    def update_log(self, message):
        self.log_text_edit.append(message)
        progress_indicators = {
            "progress:": lambda x: int(float(x.split("Progress:")[1].strip().strip('%'))) if "Progress:" in x else 0,
            "accuracy:": lambda x: int(float(x.split("Accuracy:")[1].split('%')[0])) if "Accuracy:" in x else 0,
            "confusion matrix": "Confusion Matrix Computed",
            "classification report": "Classification Report Generated",
            "macro avg": "Aggregate Metrics Computed",
            "weighted avg": "Aggregate Metrics Computed",
            "evaluation completed successfully.": "Evaluation Completed",
            "error": "Evaluation Error",
            "failed": "Evaluation Error"
        }
    
        message_lower = message.lower()
        for key, action in progress_indicators.items():
            if key in message_lower:
                if callable(action):
                    try:
                        progress = action(message_lower)
                        self.progress_bar.setValue(progress)
                        if key == "accuracy:":
                            self.progress_bar.setFormat(f"Accuracy: {progress}%")
                    except (IndexError, ValueError):
                        pass
                else:
                    self.progress_bar.setFormat(action)
                break
    
        if any(x in message_lower for x in ["error", "failed"]):
            self.buttons['start'].setEnabled(True)
            self.buttons['stop'].setEnabled(False)
    
    def on_evaluation_finished(self):
        self.progress_bar.setFormat("Evaluation Finished" if not self.worker._is_stopped else "Evaluation Stopped")
        self.buttons['start'].setEnabled(True)
        self.buttons['stop'].setEnabled(False)
    
    def update_metrics_visualization(self, accuracy, macro_avg, weighted_avg):
        self.visualization.update_metrics_visualization(accuracy, macro_avg, weighted_avg)
    
    def init_visualization(self):
        pass