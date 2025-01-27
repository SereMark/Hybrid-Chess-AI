import torch
import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader
from src.utils.datasets import H5Dataset
from src.models.transformer import TransformerChessModel
from src.utils.train_utils import initialize_random_seeds
from src.utils.chess_utils import get_total_moves

class EvaluationWorker:
    def __init__(self, model_path: str, dataset_indices_path: str, h5_file_path: str, progress_callback=None, status_callback=None):
        self.model_path = model_path
        self.dataset_indices_path = dataset_indices_path
        self.h5_file_path = h5_file_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.progress_callback = progress_callback
        self.status_callback = status_callback

    def run(self) -> Dict:
        initialize_random_seeds(42)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = TransformerChessModel(get_total_moves())
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
        model.to(self.device)
        model.eval()
        dataset = H5Dataset(self.h5_file_path, np.load(self.dataset_indices_path))
        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
        all_predictions: List[int] = []
        all_actuals: List[int] = []
        topk_predictions: List[np.ndarray] = []
        total_batches = len(loader)
        done_batches = 0
        for _, (inputs, policy_targets, _) in enumerate(loader, 1):
            inputs = inputs.to(self.device, non_blocking=True)
            policy_targets = policy_targets.to(self.device, non_blocking=True)
            with torch.no_grad():
                policy_outputs, _ = model(inputs)
                _, preds = torch.max(policy_outputs, 1)
                all_predictions.extend(preds.cpu().numpy())
                all_actuals.extend(policy_targets.cpu().numpy())
                _, topk_preds = torch.topk(policy_outputs, 5, dim=1)
                topk_predictions.extend(topk_preds.cpu().numpy())
            done_batches += 1
            self.progress_callback(done_batches / total_batches)
            self.status_callback(f"Batch {done_batches}/{total_batches} done.")
        del dataset
        torch.cuda.empty_cache()
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        topk_predictions = np.array(topk_predictions)
        unique_labels = np.unique(np.concatenate((all_actuals, all_predictions)))
        num_classes = len(unique_labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for actual, pred in zip(all_actuals, all_predictions):
            confusion_matrix[label_to_idx[actual], label_to_idx[pred]] += 1
        self.confusion_matrix = confusion_matrix
        self.accuracy = np.sum(all_predictions == all_actuals) / len(all_actuals)
        correct_topk = 0
        for actual, topk in zip(all_actuals, topk_predictions):
            if actual in topk:
                correct_topk += 1
        self.topk_accuracy = np.array([correct_topk / len(all_actuals)])
        metrics = {
            "confusion_matrix": self.confusion_matrix.tolist(),
            "accuracy": self.accuracy,
            "topk_accuracy": self.topk_accuracy.tolist(),
            "dataset_path": self.h5_file_path
        }
        return metrics