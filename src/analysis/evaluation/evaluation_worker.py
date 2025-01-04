from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
import os
import time
import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader
from typing import Optional
from src.utils.chess_utils import get_total_moves, get_move_mapping
from src.models.model import ChessModel
from src.utils.datasets import H5Dataset
from src.utils.common_utils import format_time_left, initialize_random_seeds

class EvaluationWorker(BaseWorker):
    metrics_update = pyqtSignal(float, float, dict, dict, np.ndarray, list)

    def __init__(self, model_path: str, dataset_indices_path: str, h5_file_path: str):
        super().__init__()
        self.model_path = model_path
        self.dataset_indices_path = dataset_indices_path
        self.h5_file_path = h5_file_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = 42
        self.move_mapping = get_move_mapping()

    def run_task(self):
        self.logger.info("Starting evaluation worker.")
        initialize_random_seeds(self.random_seed)
        model = self._load_model()
        if model is None:
            return
        dataset = self._load_dataset()
        if dataset is None:
            return
        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
        all_predictions = []
        all_actuals = []
        topk_predictions = []
        total_batches = len(loader)
        steps_done = 0
        total_steps = total_batches
        start_time = time.time()
        self.logger.info("Evaluating model on the dataset.")
        for batch_idx, (inputs, policy_targets, _) in enumerate(loader, 1):
            if self._is_stopped.is_set():
                self.logger.info("Evaluation stopped by user request.")
                return
            while not self._is_paused.is_set():
                time.sleep(0.1)
            inputs = inputs.to(self.device, non_blocking=True)
            policy_targets = policy_targets.to(self.device, non_blocking=True)
            with torch.no_grad():
                policy_outputs, _ = model(inputs)
                _, preds = torch.max(policy_outputs, 1)
                all_predictions.extend(preds.cpu().numpy())
                all_actuals.extend(policy_targets.cpu().numpy())
                _, topk_preds = torch.topk(policy_outputs, 5, dim=1)
                topk_predictions.extend(topk_preds.cpu().numpy())
            steps_done += 1
            progress = int(((steps_done) / total_steps) * 100)
            elapsed = time.time() - start_time
            self._update_progress(progress)
            self._update_time_left(elapsed, steps_done, total_steps)
        del dataset
        torch.cuda.empty_cache()
        self._compute_metrics(all_predictions, all_actuals, topk_predictions)
        if self._is_stopped.is_set():
            self.logger.info("Evaluation ended by user.")

    def _load_model(self) -> Optional[ChessModel]:
        num_moves = get_total_moves()
        model = ChessModel(num_moves=num_moves)
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()
            self.logger.info(f"Loaded model from {self.model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Could not load model from {self.model_path}: {str(e)}")
            return None

    def _load_dataset(self) -> Optional[H5Dataset]:
        if not os.path.exists(self.h5_file_path):
            self.logger.error(f"Dataset file not found at {self.h5_file_path}. Aborting evaluation.")
            return None
        if not os.path.exists(self.dataset_indices_path):
            self.logger.error(f"Indices file not found at {self.dataset_indices_path}. Aborting evaluation.")
            return None
        try:
            dataset_indices = np.load(self.dataset_indices_path)
            dataset = H5Dataset(self.h5_file_path, dataset_indices)
            self.logger.info(f"Dataset indices loaded from {self.dataset_indices_path}")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}")
            return None

    def _compute_metrics(self, all_predictions, all_actuals, topk_predictions):
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        topk_predictions = np.array(topk_predictions)
        accuracy = np.mean(all_predictions == all_actuals)
        self.logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        topk_correct = sum(1 for actual, preds in zip(all_actuals, topk_predictions) if actual in preds)
        topk_accuracy = topk_correct / len(all_actuals)
        self.logger.info(f"Top-5 Accuracy: {topk_accuracy * 100:.2f}%")
        N = 10
        class_counts = Counter(all_actuals)
        most_common_classes = [item[0] for item in class_counts.most_common(N)]
        indices = np.isin(all_actuals, most_common_classes)
        filtered_actuals = all_actuals[indices]
        filtered_predictions = all_predictions[indices]
        confusion = self._compute_confusion_matrix(filtered_actuals, filtered_predictions, most_common_classes)
        report = self._compute_classification_report(filtered_actuals, filtered_predictions, most_common_classes)
        macro_avg = report.get('macro avg', {})
        weighted_avg = report.get('weighted avg', {})
        class_labels = []
        for cls_idx in most_common_classes:
            move_obj = self.move_mapping.get_move_by_index(cls_idx)
            if move_obj:
                class_labels.append(move_obj.uci())
            else:
                class_labels.append(f"Unknown({cls_idx})")
        self.logger.info(f"Macro Avg - Precision: {macro_avg.get('precision', 0.0):.4f}, Recall: {macro_avg.get('recall', 0.0):.4f}, F1: {macro_avg.get('f1-score', 0.0):.4f}")
        self.logger.info(f"Weighted Avg - Precision: {weighted_avg.get('precision', 0.0):.4f}, Recall: {weighted_avg.get('recall', 0.0):.4f}, F1: {weighted_avg.get('f1-score', 0.0):.4f}")
        if self.metrics_update:
            self.metrics_update.emit(accuracy, topk_accuracy, macro_avg, weighted_avg, confusion, class_labels)
        self.logger.info("Evaluation metrics have been computed successfully.")

    def _compute_confusion_matrix(self, actuals, predictions, labels):
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for actual, pred in zip(actuals, predictions):
            if actual in label_to_index and pred in label_to_index:
                matrix[label_to_index[actual], label_to_index[pred]] += 1
        return matrix

    def _compute_classification_report(self, actuals, predictions, labels):
        report = {}
        support = Counter(actuals)
        tp = {label: 0 for label in labels}
        fp = {label: 0 for label in labels}
        fn = {label: 0 for label in labels}
        for actual, pred in zip(actuals, predictions):
            if actual == pred:
                tp[actual] += 1
            else:
                if pred in fp:
                    fp[pred] += 1
                fn[actual] += 1
        precision = {}
        recall = {}
        f1_score = {}
        for label in labels:
            p_denom = tp[label] + fp[label]
            r_denom = tp[label] + fn[label]
            precision[label] = tp[label] / p_denom if p_denom > 0 else 0.0
            recall[label] = tp[label] / r_denom if r_denom > 0 else 0.0
            if precision[label] + recall[label] > 0:
                f1_score[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label])
            else:
                f1_score[label] = 0.0
        macro_precision = np.mean(list(precision.values()))
        macro_recall = np.mean(list(recall.values()))
        macro_f1 = np.mean(list(f1_score.values()))
        total_support = sum(support[label] for label in labels)
        weighted_precision = sum(precision[l] * support[l] for l in labels) / total_support if total_support > 0 else 0.0
        weighted_recall = sum(recall[l] * support[l] for l in labels) / total_support if total_support > 0 else 0.0
        weighted_f1 = sum(f1_score[l] * support[l] for l in labels) / total_support if total_support > 0 else 0.0
        for label in labels:
            report[label] = {
                'precision': precision[label],
                'recall': recall[label],
                'f1-score': f1_score[label],
                'support': support[label]
            }
        report['macro avg'] = {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1-score': macro_f1,
            'support': total_support
        }
        report['weighted avg'] = {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1-score': weighted_f1,
            'support': total_support
        }
        return report

    def _update_progress(self, progress: int):
        if self.progress_update:
            self.progress_update.emit(progress)

    def _update_time_left(self, elapsed_time: float, steps_done: int, total_steps: int):
        if steps_done > 0 and self.time_left_update:
            estimated_total_time = (elapsed_time / steps_done) * total_steps
            time_left = max(0, estimated_total_time - elapsed_time)
            time_left_str = format_time_left(time_left)
            self.time_left_update.emit(time_left_str)