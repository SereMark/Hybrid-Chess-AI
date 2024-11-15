import os, time, numpy as np, torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from typing import Callable, Optional
from src.utils.chess_utils import TOTAL_MOVES, MOVE_MAPPING, initialize_move_mappings
from src.models.model import ChessModel
from src.utils.datasets import H5Dataset

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

    def evaluate_model(self):
        self._log(f"Using device: {self.device}")
        initialize_move_mappings()

        model = self._load_model()
        if model is None:
            return

        dataset = self._load_dataset()
        if dataset is None:
            return

        self._evaluate(model, dataset)

    def _load_model(self) -> Optional[ChessModel]:
        model = ChessModel(num_moves=TOTAL_MOVES)
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()
            self._log(f"Model loaded from {self.model_path}")
            return model
        except Exception as e:
            self._log(f"Failed to load model: {e}")
            return None

    def _load_dataset(self) -> Optional[H5Dataset]:
        if not os.path.exists(self.h5_file_path):
            self._log(f"Dataset file not found at {self.h5_file_path}.")
            return None
        if not os.path.exists(self.dataset_indices_path):
            self._log(f"Dataset indices file not found at {self.dataset_indices_path}.")
            return None
        try:
            dataset_indices = np.load(self.dataset_indices_path)
            dataset = H5Dataset(self.h5_file_path, dataset_indices)
            self._log(f"Loaded dataset indices from {self.dataset_indices_path}")
            return dataset
        except Exception as e:
            self._log(f"Failed to load dataset: {e}")
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
        self._log("Starting evaluation...")

        for batch_idx, (inputs, policy_targets, _) in enumerate(loader, 1):
            if self.stop_fn():
                self._log("Evaluation stopped by user.")
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

        dataset.__del__()

        self._compute_metrics(all_predictions, all_actuals, topk_predictions)

    def _compute_metrics(self, all_predictions, all_actuals, topk_predictions):
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        topk_predictions = np.array(topk_predictions)

        accuracy = np.mean(all_predictions == all_actuals)
        self._log(f"Accuracy: {accuracy * 100:.2f}%")

        topk_correct = sum(1 for actual, preds in zip(all_actuals, topk_predictions) if actual in preds)
        topk_accuracy = topk_correct / len(all_actuals)
        self._log(f"Top-{topk_predictions.shape[1]} Accuracy: {topk_accuracy * 100:.2f}%")

        N = 10
        from collections import Counter
        class_counts = Counter(all_actuals)
        most_common_classes = [item[0] for item in class_counts.most_common(N)]
        indices = np.isin(all_actuals, most_common_classes)
        filtered_actuals = all_actuals[indices]
        filtered_predictions = all_predictions[indices]

        confusion = confusion_matrix(filtered_actuals, filtered_predictions, labels=most_common_classes)
        self._log("Confusion Matrix computed.")

        report = classification_report(
            filtered_actuals,
            filtered_predictions,
            labels=most_common_classes,
            output_dict=True,
            zero_division=0
        )
        macro_avg = report.get('macro avg', {})
        weighted_avg = report.get('weighted avg', {})

        class_labels = [MOVE_MAPPING[cls].uci() for cls in most_common_classes]

        self._log(f"Macro Avg - Precision: {macro_avg.get('precision', 0.0):.4f}, "
                  f"Recall: {macro_avg.get('recall', 0.0):.4f}, "
                  f"F1-Score: {macro_avg.get('f1-score', 0.0):.4f}")
        self._log(f"Weighted Avg - Precision: {weighted_avg.get('precision', 0.0):.4f}, "
                  f"Recall: {weighted_avg.get('recall', 0.0):.4f}, "
                  f"F1-Score: {weighted_avg.get('f1-score', 0.0):.4f}")

        if self.metrics_fn:
            self.metrics_fn(
                accuracy,
                topk_accuracy,
                macro_avg,
                weighted_avg,
                confusion,
                class_labels
            )

        self._log("Evaluation process finished.")

    def _log(self, message: str):
        if self.log_fn:
            self.log_fn(message)

    def _update_progress(self, progress: int):
        if self.progress_fn:
            self.progress_fn(progress)

    def _update_time_left(self, elapsed_time: float, steps_done: int, total_steps: int):
        if self.time_left_fn:
            if steps_done > 0:
                estimated_total_time = (elapsed_time / steps_done) * total_steps
                time_left = estimated_total_time - elapsed_time
                time_left_str = self._format_time_left(time_left)
                self.time_left_fn(time_left_str)
            else:
                self.time_left_fn("Calculating...")

    def _format_time_left(self, seconds: float) -> str:
        days = int(seconds) // 86400
        remainder = int(seconds) % 86400
        hours = remainder // 3600
        minutes = (remainder % 3600) // 60
        secs = remainder % 60

        if days >= 1:
            day_str = f"{days}d "
            return f"{day_str}{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"