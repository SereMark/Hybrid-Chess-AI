import numpy as np
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.evaluation.evaluate import ModelEvaluator


class EvaluationWorker(BaseWorker):
    metrics_update = pyqtSignal(float, float, dict, dict, np.ndarray, list)

    def __init__(self, model_path: str, dataset_indices_path: str, h5_file_path: str):
        super().__init__()
        self.model_path = model_path
        self.dataset_indices_path = dataset_indices_path
        self.h5_file_path = h5_file_path

    def run_task(self):
        self.log_update.emit("Starting model evaluation...")
        evaluator = ModelEvaluator(
            model_path=self.model_path,
            dataset_indices_path=self.dataset_indices_path,
            h5_file_path=self.h5_file_path,
            log_fn=self.log_update.emit,
            metrics_fn=self.metrics_update.emit,
            progress_fn=self.progress_update.emit,
            time_left_fn=self.time_left_update.emit,
            stop_fn=self._is_stopped
        )
        evaluator.evaluate_model()
        if self._is_stopped.is_set():
            self.log_update.emit("Evaluation stopped by user request.")