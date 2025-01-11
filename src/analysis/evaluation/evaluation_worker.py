from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
import os, time, numpy as np, torch
from collections import Counter
from torch.utils.data import DataLoader
from typing import Optional
from src.utils.chess_utils import get_total_moves, get_move_mapping
from src.models.model import ChessModel
from src.utils.datasets import H5Dataset
from src.utils.common_utils import initialize_random_seeds, update_progress_time_left, wait_if_paused

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
        done_steps = 0
        total_steps = total_batches
        start_time = time.time()
        self.logger.info("Evaluating model on dataset.")
        for batch_idx, (inputs, policy_targets, _) in enumerate(loader, 1):
            if self._is_stopped.is_set():
                self.logger.info("Evaluation stopped by user.")
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
            update_progress_time_left(
                self.progress_update,
                self.time_left_update,
                start_time,
                done_steps,
                total_steps
            )
        del dataset
        torch.cuda.empty_cache()
        self._compute_metrics(all_predictions, all_actuals, topk_predictions)

    def _load_model(self) -> Optional[ChessModel]:
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
            model = ChessModel(num_moves=get_total_moves())
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
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

    def _compute_metrics(self, all_predictions, all_actuals, topk_predictions):
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        topk_predictions = np.array(topk_predictions)
        accuracy = np.mean(all_predictions == all_actuals)
        self.logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        topk_correct = 0
        for actual, preds in zip(all_actuals, topk_predictions):
            if actual in preds:
                topk_correct += 1
        topk_accuracy = topk_correct / len(all_actuals)
        self.logger.info(f"Top-5 Accuracy: {topk_accuracy * 100:.2f}%")
        N = 10
        class_counts = Counter(all_actuals)
        common_classes = [item[0] for item in class_counts.most_common(N)]
        indices = np.isin(all_actuals, common_classes)
        filtered_actuals = all_actuals[indices]
        filtered_predictions = all_predictions[indices]
        confusion = self._compute_confusion(filtered_actuals, filtered_predictions, common_classes)
        report = self._compute_class_report(filtered_actuals, filtered_predictions, common_classes)
        macro_avg = report.get('macro avg', {})
        weighted_avg = report.get('weighted avg', {})
        labels = []
        for idx in common_classes:
            move_obj = self.move_mapping.get_move_by_index(idx)
            if move_obj:
                labels.append(move_obj.uci())
            else:
                labels.append(f"Unknown({idx})")
        if self.metrics_update:
            self.metrics_update.emit(accuracy, topk_accuracy, macro_avg, weighted_avg, confusion, labels)

    def _compute_confusion(self, actuals, predictions, labels):
        label_to_idx = {label: i for i, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for a, p in zip(actuals, predictions):
            if a in label_to_idx and p in label_to_idx:
                matrix[label_to_idx[a], label_to_idx[p]] += 1
        return matrix

    def _compute_class_report(self, actuals, predictions, labels):
        report = {}
        support = Counter(actuals)
        tp = {label: 0 for label in labels}
        fp = {label: 0 for label in labels}
        fn = {label: 0 for label in labels}
        for a, p in zip(actuals, predictions):
            if a == p:
                tp[a] += 1
            else:
                if p in fp:
                    fp[p] += 1
                fn[a] += 1
        precision = {}
        recall = {}
        f1_score = {}
        for label in labels:
            p_denom = tp[label] + fp[label]
            r_denom = tp[label] + fn[label]
            precision[label] = tp[label] / p_denom if p_denom > 0 else 0.0
            recall[label] = tp[label] / r_denom if r_denom > 0 else 0.0
            if (precision[label] + recall[label]) > 0:
                f1_score[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label])
            else:
                f1_score[label] = 0.0
        macro_p = np.mean(list(precision.values()))
        macro_r = np.mean(list(recall.values()))
        macro_f = np.mean(list(f1_score.values()))
        total_s = sum(support[l] for l in labels)
        weighted_p = sum(precision[l] * support[l] for l in labels) / total_s if total_s else 0.0
        weighted_r = sum(recall[l] * support[l] for l in labels) / total_s if total_s else 0.0
        weighted_f = sum(f1_score[l] * support[l] for l in labels) / total_s if total_s else 0.0
        for label in labels:
            report[label] = {
                'precision': precision[label],
                'recall': recall[label],
                'f1-score': f1_score[label],
                'support': support[label]
            }
        report['macro avg'] = {
            'precision': macro_p,
            'recall': macro_r,
            'f1-score': macro_f,
            'support': total_s
        }
        report['weighted avg'] = {
            'precision': weighted_p,
            'recall': weighted_r,
            'f1-score': weighted_f,
            'support': total_s
        }
        return report