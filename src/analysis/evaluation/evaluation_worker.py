import os
import time
from typing import Optional, List
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from models.tempname import TransformerChessModel
from src.utils.datasets import H5Dataset
from src.utils.common_utils import update_progress_time_left, wait_if_paused
from src.utils.train_utils import initialize_random_seeds
from src.utils.chess_utils import get_total_moves, get_move_mapping

class EvaluationWorker(BaseWorker):
    metrics_update = pyqtSignal(float, float, np.ndarray, list)

    def __init__(self, model_path: str, dataset_indices_path: str, h5_file_path: str):
        super().__init__()
        self.model_path = model_path
        self.dataset_indices_path = dataset_indices_path
        self.h5_file_path = h5_file_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_task(self):
        self.logger.info("Starting evaluation worker.")
        initialize_random_seeds(42)

        model = self._load_model()
        if model is None:
            self.logger.error("Model loading failed. Aborting evaluation.")
            return

        dataset = self._load_dataset()
        if dataset is None:
            self.logger.error("Dataset loading failed. Aborting evaluation.")
            return

        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)

        all_predictions: List[int] = []
        all_actuals: List[int] = []
        topk_predictions: List[np.ndarray] = []

        total_batches = len(loader)
        done_steps = 0
        total_steps = total_batches
        start_time = time.time()

        self.logger.info("Evaluating model on dataset.")

        for _, (inputs, policy_targets, _) in enumerate(loader, 1):
            if self._is_stopped.is_set():
                return

            wait_if_paused(self._is_paused)

            inputs = inputs.to(self.device, non_blocking=True)
            policy_targets = policy_targets.to(self.device, non_blocking=True)

            with torch.no_grad():
                policy_outputs, _ = model(inputs)
                _, preds = torch.max(policy_outputs, 1)
                all_predictions.extend(preds.cpu().numpy())
                all_actuals.extend(policy_targets.cpu().numpy())
                _, topk_preds = torch.topk(policy_outputs, 5, dim=1)
                topk_predictions.extend(topk_preds.cpu().numpy())

            done_steps += 1

            update_progress_time_left(self.progress_update, self.time_left_update, start_time, done_steps, total_steps)

        del dataset
        torch.cuda.empty_cache()

        self._compute_metrics(all_predictions, all_actuals, topk_predictions)

    def _load_model(self) -> Optional[TransformerChessModel]:
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.logger.info(f"Loaded checkpoint from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Could not load model from {self.model_path}: {str(e)}")
            return None

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            state_dict = checkpoint
            self.logger.warning("Checkpoint does not contain architecture parameters. Using default settings.")

        try:
            model = TransformerChessModel(num_moves=get_total_moves())
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.logger.info("Model loaded and set to evaluation mode.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load state_dict into model: {str(e)}")
            return None

    def _load_dataset(self) -> Optional[H5Dataset]:
        if not os.path.exists(self.h5_file_path):
            self.logger.error(f"Dataset file not found at {self.h5_file_path}.")
            return None
        if not os.path.exists(self.dataset_indices_path):
            self.logger.error(f"Indices file not found at {self.dataset_indices_path}.")
            return None

        try:
            idxs = np.load(self.dataset_indices_path)
            dataset = H5Dataset(self.h5_file_path, idxs)
            self.logger.info(f"Loaded dataset indices from {self.dataset_indices_path}")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}")
            return None

    def _compute_metrics(self, all_predictions: List[int], all_actuals: List[int], topk_predictions: List[np.ndarray]):
        # Convert lists to numpy arrays for efficient computation
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        topk_predictions = np.array(topk_predictions)

        # Compute Accuracy
        accuracy = np.mean(all_predictions == all_actuals)
        self.logger.info(f"Accuracy: {accuracy * 100:.2f}%")

        # Compute Top-K Accuracy (Top-5)
        topk_correct = np.array([actual in preds for actual, preds in zip(all_actuals, topk_predictions)])
        topk_accuracy = np.mean(topk_correct)
        self.logger.info(f"Top-5 Accuracy: {topk_accuracy * 100:.2f}%")

        # Compute Confusion Matrix
        unique_labels = np.unique(all_actuals)
        num_classes = len(unique_labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for actual, pred in zip(all_actuals, all_predictions):
            confusion_matrix[label_to_idx[actual], label_to_idx[pred]] += 1

        # Emit metrics
        move_mapping = get_move_mapping()
        class_labels = [move.uci() if (move := move_mapping.get_move_by_index(label)) else f"Unknown({label})" for label in unique_labels]
        if self.metrics_update:
            self.metrics_update.emit(accuracy, topk_accuracy, confusion_matrix, class_labels)