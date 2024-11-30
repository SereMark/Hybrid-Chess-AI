import os, time, numpy as np, torch
from collections import Counter
from torch.utils.data import DataLoader
from typing import Callable, Optional
from src.utils.chess_utils import get_total_moves, get_move_mapping
from src.models.model import ChessModel
from src.utils.datasets import H5Dataset
from src.utils.common_utils import (
    format_time_left,
    log_message,
    should_stop,
    initialize_random_seeds,
)

class ModelEvaluator:
    def __init__(
        self,
        model_path: str,
        dataset_indices_path: str,
        h5_file_path: str,
        log_fn: Optional[Callable[[str], None]] = None,
        metrics_fn: Optional[Callable[[float, float, dict, dict, np.ndarray, list], None]] = None,
        progress_fn: Optional[Callable[[int], None]] = None,
        time_left_fn: Optional[Callable[[str], None]] = None,
        stop_fn: Optional[Callable[[], bool]] = None,
        random_seed: int = 42
    ):
        self.model_path = model_path
        self.dataset_indices_path = dataset_indices_path
        self.h5_file_path = h5_file_path
        self.log_fn = log_fn
        self.metrics_fn = metrics_fn
        self.progress_fn = progress_fn
        self.time_left_fn = time_left_fn
        self.stop_fn = stop_fn or (lambda: False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed

    def evaluate_model(self):
        log_message(f"Using device: {self.device}", self.log_fn)
        initialize_random_seeds(self.random_seed)

        model = self._load_model()
        if model is None:
            return

        dataset = self._load_dataset()
        if dataset is None:
            return

        self._evaluate(model, dataset)

    def _load_model(self) -> Optional[ChessModel]:
        num_moves = get_total_moves()
        model = ChessModel(num_moves=num_moves)
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)  # TODO: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling.
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()
            log_message(f"Model loaded from {self.model_path}", self.log_fn)
            return model
        except Exception as e:
            log_message(f"Failed to load model: {e}", self.log_fn)
            return None

    def _load_dataset(self) -> Optional[H5Dataset]:
        if not os.path.exists(self.h5_file_path):
            log_message(f"Dataset file not found at {self.h5_file_path}.", self.log_fn)
            return None
        if not os.path.exists(self.dataset_indices_path):
            log_message(f"Dataset indices file not found at {self.dataset_indices_path}.", self.log_fn)
            return None
        try:
            dataset_indices = np.load(self.dataset_indices_path)
            dataset = H5Dataset(self.h5_file_path, dataset_indices)
            log_message(f"Loaded dataset indices from {self.dataset_indices_path}", self.log_fn)
            return dataset
        except Exception as e:
            log_message(f"Failed to load dataset: {e}", self.log_fn)
            return None

    def _evaluate(self, model: ChessModel, dataset: H5Dataset):
        loader = DataLoader(
            dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True
        )

        all_predictions = []
        all_actuals = []
        k = 5
        topk_predictions = []

        total_batches = len(loader)
        steps_done = 0
        total_steps = total_batches

        start_time = time.time()
        log_message("Starting evaluation...", self.log_fn)

        for batch_idx, (inputs, policy_targets, _) in enumerate(loader, 1):
            if should_stop(self.stop_fn):
                log_message("Evaluation stopped by user.", self.log_fn)
                return

            inputs = inputs.to(self.device, non_blocking=True)
            policy_targets = policy_targets.to(self.device, non_blocking=True)
            with torch.no_grad():
                policy_outputs, _ = model(inputs)
                _, preds = torch.max(policy_outputs, 1)
                all_predictions.extend(preds.cpu().numpy())
                all_actuals.extend(policy_targets.cpu().numpy())

                _, topk_preds = torch.topk(policy_outputs, k, dim=1)
                topk_predictions.extend(topk_preds.cpu().numpy())

            steps_done += 1
            progress = int((steps_done / total_steps) * 100)
            elapsed_time = time.time() - start_time

            self._update_progress(progress)
            self._update_time_left(elapsed_time, steps_done, total_steps)

        del dataset
        torch.cuda.empty_cache()

        self._compute_metrics(all_predictions, all_actuals, topk_predictions)

    def _compute_metrics(self, all_predictions, all_actuals, topk_predictions):
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        topk_predictions = np.array(topk_predictions)

        accuracy = np.mean(all_predictions == all_actuals)
        log_message(f"Accuracy: {accuracy * 100:.2f}%", self.log_fn)

        topk_correct = sum(1 for actual, preds in zip(all_actuals, topk_predictions) if actual in preds)
        topk_accuracy = topk_correct / len(all_actuals)
        log_message(f"Top-{topk_predictions.shape[1]} Accuracy: {topk_accuracy * 100:.2f}%", self.log_fn)

        N = 10
        class_counts = Counter(all_actuals)
        most_common_classes = [item[0] for item in class_counts.most_common(N)]
        indices = np.isin(all_actuals, most_common_classes)
        filtered_actuals = all_actuals[indices]
        filtered_predictions = all_predictions[indices]

        confusion = self._compute_confusion_matrix(filtered_actuals, filtered_predictions, most_common_classes)
        log_message("Confusion Matrix computed.", self.log_fn)

        report = self._compute_classification_report(filtered_actuals, filtered_predictions, most_common_classes)
        macro_avg = report.get('macro avg', {})
        weighted_avg = report.get('weighted avg', {})

        move_mapping = get_move_mapping()
        class_labels = []
        for cls in most_common_classes:
            move = move_mapping.get_move_by_index(cls)
            if move:
                class_labels.append(move.uci())
            else:
                class_labels.append(f"Unknown({cls})")

        log_message(
            f"Macro Avg - Precision: {macro_avg.get('precision', 0.0):.4f}, "
            f"Recall: {macro_avg.get('recall', 0.0):.4f}, "
            f"F1-Score: {macro_avg.get('f1-score', 0.0):.4f}",
            self.log_fn
        )
        log_message(
            f"Weighted Avg - Precision: {weighted_avg.get('precision', 0.0):.4f}, "
            f"Recall: {weighted_avg.get('recall', 0.0):.4f}, "
            f"F1-Score: {weighted_avg.get('f1-score', 0.0):.4f}",
            self.log_fn
        )

        if self.metrics_fn:
            self.metrics_fn(
                accuracy,
                topk_accuracy,
                macro_avg,
                weighted_avg,
                confusion,
                class_labels
            )

        log_message("Evaluation process finished.", self.log_fn)

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
            precision[label] = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
            recall[label] = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
            if precision[label] + recall[label] > 0:
                f1_score[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label])
            else:
                f1_score[label] = 0.0

        macro_precision = np.mean(list(precision.values()))
        macro_recall = np.mean(list(recall.values()))
        macro_f1 = np.mean(list(f1_score.values()))

        total_support = sum(support[label] for label in labels)
        weighted_precision = sum(precision[label] * support[label] for label in labels) / total_support if total_support > 0 else 0.0
        weighted_recall = sum(recall[label] * support[label] for label in labels) / total_support if total_support > 0 else 0.0
        weighted_f1 = sum(f1_score[label] * support[label] for label in labels) / total_support if total_support > 0 else 0.0

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
        if self.progress_fn:
            self.progress_fn(progress)

    def _update_time_left(self, elapsed_time: float, steps_done: int, total_steps: int):
        if self.time_left_fn:
            if steps_done > 0:
                estimated_total_time = (elapsed_time / steps_done) * total_steps
                time_left = estimated_total_time - elapsed_time
                time_left_str = format_time_left(time_left)
                self.time_left_fn(time_left_str)
            else:
                self.time_left_fn("Calculating...")