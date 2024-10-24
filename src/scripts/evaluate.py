# src\scripts\evaluate.py

import os, time, numpy as np, torch, h5py
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from src.scripts.train import ChessModel, H5Dataset, TOTAL_MOVES, MOVE_MAPPING


class ModelEvaluator:
    def __init__(self, model_path, dataset_indices_path, log_fn=None, metrics_fn=None, progress_fn=None,
                 time_left_fn=None, stop_fn=None):
        self.model_path = model_path
        self.dataset_indices_path = dataset_indices_path
        self.log_fn = log_fn
        self.metrics_fn = metrics_fn
        self.progress_fn = progress_fn
        self.time_left_fn = time_left_fn
        self.stop_fn = stop_fn or (lambda: False)

    def format_time_left(self, seconds):
            days = seconds // 86400
            remainder = seconds % 86400
            hours = remainder // 3600
            minutes = (remainder % 3600) // 60
            secs = remainder % 60

            if days >= 1:
                day_str = f"{int(days)}d " if days > 1 else "1d "
                return f"{day_str}{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
            else:
                return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

    def evaluate_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.log_fn:
            self.log_fn(f"Using device: {device}")

        model = ChessModel(num_moves=TOTAL_MOVES)
        try:
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            if self.log_fn:
                self.log_fn(f"Model loaded from {self.model_path}")
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Failed to load model: {e}")
            return

        model.to(device)
        model.eval()

        data_dir = os.path.dirname(self.dataset_indices_path)
        h5_path = os.path.join(data_dir, 'dataset.h5')
        if not os.path.exists(h5_path):
            if self.log_fn:
                self.log_fn(f"Dataset file not found at {h5_path}.")
            return
        if not os.path.exists(self.dataset_indices_path):
            if self.log_fn:
                self.log_fn(f"Dataset indices file not found at {self.dataset_indices_path}.")
            return

        try:
            h5_file = h5py.File(h5_path, 'r')
            dataset_indices = np.load(self.dataset_indices_path)
            dataset = H5Dataset(h5_file, dataset_indices)
            if self.log_fn:
                self.log_fn(f"Loaded dataset indices from {self.dataset_indices_path}")
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Failed to load dataset: {e}")
            return

        loader = DataLoader(
            dataset, batch_size=1024, shuffle=False, num_workers=0
        )

        all_predictions = []
        all_actuals = []
        total_batches = len(loader)
        k = 5
        topk_predictions = []

        start_time = time.time()
        steps_done = 0
        total_steps = total_batches

        if self.log_fn:
            self.log_fn("Starting evaluation...")

        for batch_idx, (inputs, policy_targets, _) in enumerate(loader, 1):
            if self.stop_fn():
                if self.log_fn:
                    self.log_fn("Evaluation stopped by user.")
                return

            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
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

            if self.progress_fn:
                self.progress_fn(progress)

            if self.time_left_fn:
                estimated_total_time = (elapsed_time / steps_done) * total_steps
                time_left = estimated_total_time - elapsed_time
                time_left_str = self.format_time_left(time_left)
                self.time_left_fn(time_left_str)

        h5_file.close()

        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        topk_predictions = np.array(topk_predictions)

        accuracy = np.mean(all_predictions == all_actuals)
        if self.log_fn:
            self.log_fn(f"Accuracy: {accuracy * 100:.2f}%")

        topk_correct = sum([1 for actual, preds in zip(all_actuals, topk_predictions) if actual in preds])
        topk_accuracy = topk_correct / len(all_actuals)
        if self.log_fn:
            self.log_fn(f"Top-{k} Accuracy: {topk_accuracy * 100:.2f}%")

        N = 10
        from collections import Counter
        class_counts = Counter(all_actuals)
        most_common_classes = [item[0] for item in class_counts.most_common(N)]
        indices = np.isin(all_actuals, most_common_classes)
        filtered_actuals = all_actuals[indices]
        filtered_predictions = all_predictions[indices]

        confusion = confusion_matrix(filtered_actuals, filtered_predictions, labels=most_common_classes)
        if self.log_fn:
            self.log_fn("Confusion Matrix computed.")

        report = classification_report(filtered_actuals, filtered_predictions, labels=most_common_classes,
                                       output_dict=True, zero_division=0)
        macro_avg = report.get('macro avg', {})
        weighted_avg = report.get('weighted avg', {})

        class_labels = [MOVE_MAPPING[cls].uci() for cls in most_common_classes]

        if self.log_fn:
            self.log_fn(f"Macro Avg - Precision: {macro_avg.get('precision', 0.0):.4f}, "
                        f"Recall: {macro_avg.get('recall', 0.0):.4f}, "
                        f"F1-Score: {macro_avg.get('f1-score', 0.0):.4f}")
            self.log_fn(f"Weighted Avg - Precision: {weighted_avg.get('precision', 0.0):.4f}, "
                        f"Recall: {weighted_avg.get('recall', 0.0):.4f}, "
                        f"F1-Score: {weighted_avg.get('f1-score', 0.0):.4f}")

        if self.metrics_fn:
            self.metrics_fn(accuracy, topk_accuracy, macro_avg, weighted_avg, confusion, class_labels)

        if self.log_fn:
            self.log_fn("Evaluation process finished.")