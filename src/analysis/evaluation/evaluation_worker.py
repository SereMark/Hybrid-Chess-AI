import torch, numpy as np
from torch.utils.data import DataLoader
from src.utils.datasets import H5Dataset
from src.models.transformer import TransformerChessModel
from src.utils.train_utils import initialize_random_seeds
from src.utils.chess_utils import get_total_moves

class EvaluationWorker:
    def __init__(self, model_path: str, dataset_indices_path: str, h5_file_path: str, progress_callback=None, status_callback=None):
        self.model_path, self.dataset_indices_path, self.h5_file_path = model_path, dataset_indices_path, h5_file_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.progress_callback, self.status_callback = progress_callback, status_callback

    def run(self):
        initialize_random_seeds(42)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = TransformerChessModel(get_total_moves())
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
        model.to(self.device).eval()
        dataset = H5Dataset(self.h5_file_path, np.load(self.dataset_indices_path))
        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
        all_predictions, all_actuals = [], []
        total_batches = len(loader)
        for done_batches, (inputs, policy_targets, _) in enumerate(loader, 1):
            inputs, policy_targets = inputs.to(self.device, non_blocking=True), policy_targets.to(self.device, non_blocking=True)
            with torch.no_grad():
                preds = torch.max(model(inputs)[0], 1)[1]
                all_predictions.extend(preds.cpu().numpy())
                all_actuals.extend(policy_targets.cpu().numpy())
            self.progress_callback(done_batches / total_batches)
            self.status_callback(f"Batch {done_batches}/{total_batches} done.")
        del dataset
        torch.cuda.empty_cache()
        all_predictions, all_actuals = np.array(all_predictions), np.array(all_actuals)
        unique_labels = np.unique(np.concatenate((all_actuals, all_predictions)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        for actual, pred in zip(all_actuals, all_predictions):
            confusion_matrix[label_to_idx[actual], label_to_idx[pred]] += 1
        accuracy = np.sum(all_predictions == all_actuals) / len(all_actuals)
        self.status_callback(f"Evaluation done: Confusion matrix: \n{confusion_matrix}\nAccuracy: {accuracy:.4f}")
        return True