import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from .train import ChessModel, ChessTrainingDataset, TOTAL_MOVES

class ModelEvaluator:
    def __init__(self, model_path, dataset_path, log_fn=None, metrics_fn=None, stop_fn=None):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.log_fn = log_fn
        self.metrics_fn = metrics_fn
        self.stop_fn = stop_fn or (lambda: False)

    def evaluate_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.log_fn:
            self.log_fn(f"Using device: {device}")

        model = ChessModel(num_moves=TOTAL_MOVES)
        try:
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=True)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            self.log_fn(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.log_fn(f"Failed to load model: {e}")
            return

        model.to(device)
        model.eval()

        try:
            if self.dataset_path.endswith('.npy'):
                val_indices = np.load(self.dataset_path)
                dataset = ChessTrainingDataset(os.path.join('data', 'processed', 'dataset.h5'), val_indices)
                self.log_fn(f"Loaded validation indices from {self.dataset_path}")
            elif self.dataset_path.endswith('.h5'):
                dataset = ChessTrainingDataset(self.dataset_path)
                self.log_fn(f"Loaded dataset from {self.dataset_path}")
            else:
                self.log_fn("Unsupported dataset format. Use .npy for indices or .h5 for dataset.")
                return
        except Exception as e:
            self.log_fn(f"Failed to load dataset: {e}")
            return

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, shuffle=False, num_workers=4
        )

        all_predictions = []
        all_actuals = []
        total_batches = len(loader)
        self.log_fn("Starting evaluation...")

        for batch_idx, (inputs, policy_targets, value_targets) in enumerate(loader, 1):
            if self.stop_fn():
                self.log_fn("Evaluation stopped by user.")
                return

            inputs = inputs.to(device)
            with torch.no_grad():
                policy_outputs, _ = model(inputs)
                _, preds = torch.max(policy_outputs, 1)
                all_predictions.extend(preds.cpu().numpy())
                all_actuals.extend(policy_targets.numpy())

            if batch_idx % 10 == 0 or batch_idx == total_batches:
                progress = int((batch_idx / total_batches) * 100)
                self.log_fn(f"Progress: {progress}%")
                if self.stop_fn():
                    self.log_fn("Evaluation stopped by user.")
                    return

        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)

        accuracy = np.mean(all_predictions == all_actuals)
        self.log_fn(f"Accuracy: {accuracy * 100:.2f}%")

        confusion = confusion_matrix(all_actuals, all_predictions, labels=range(TOTAL_MOVES))
        self.log_fn("Confusion Matrix computed.")

        report = classification_report(all_actuals, all_predictions, output_dict=True, zero_division=0)
        macro_avg = report.get('macro avg', {})
        weighted_avg = report.get('weighted avg', {})

        self.log_fn(f"Macro Avg - Precision: {macro_avg.get('precision', 0.0):.4f}, "
                   f"Recall: {macro_avg.get('recall', 0.0):.4f}, "
                   f"F1-Score: {macro_avg.get('f1-score', 0.0):.4f}")
        self.log_fn(f"Weighted Avg - Precision: {weighted_avg.get('precision', 0.0):.4f}, "
                   f"Recall: {weighted_avg.get('recall', 0.0):.4f}, "
                   f"F1-Score: {weighted_avg.get('f1-score', 0.0):.4f}")

        if self.metrics_fn:
            self.metrics_fn(accuracy, macro_avg, weighted_avg)

        self.log_fn("Evaluation process finished.")