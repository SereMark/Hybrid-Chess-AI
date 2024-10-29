import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from src.models.evaluate import ModelEvaluator


class EvaluationWorker(QThread):
    log_update = pyqtSignal(str)
    metrics_update = pyqtSignal(float, float, dict, dict, np.ndarray, list)
    progress_update = pyqtSignal(int)
    time_left_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_path, dataset_indices_path):
        super().__init__()
        self.model_path = model_path
        self.dataset_indices_path = dataset_indices_path
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
        self.log_update.emit("Evaluation stopped by user.")