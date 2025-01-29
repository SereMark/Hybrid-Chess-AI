import torch, numpy as np, random
from torch.utils.data import DataLoader
from src.utils.datasets import H5Dataset
from src.models.transformer import TransformerChessModel
from src.utils.train_utils import initialize_random_seeds
from src.utils.chess_utils import get_total_moves

class EvaluationWorker:
    def __init__(self, model_path: str, dataset_indices_path: str, h5_file_path: str, wandb_flag=False, progress_callback=None, status_callback=None):
        self.model_path, self.dataset_indices_path, self.h5_file_path = model_path, dataset_indices_path, h5_file_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.progress_callback, self.status_callback = progress_callback, status_callback
        self.wandb = wandb_flag

    def run(self):
        if self.wandb:
            import wandb
            wandb.init(entity="chess_ai", project="chess_ai_app", name="evaluation", config=self.__dict__, reinit=True)
        initialize_random_seeds(42)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = TransformerChessModel(get_total_moves())
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
        model.to(self.device).eval()
        dataset = H5Dataset(self.h5_file_path, np.load(self.dataset_indices_path))
        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
        all_predictions, all_actuals = [], []
        accuracy_per_batch = []
        total_batches = len(loader)
        for done_batches, (inputs, policy_targets, _) in enumerate(loader, 1):
            inputs, policy_targets = inputs.to(self.device, non_blocking=True), policy_targets.to(self.device, non_blocking=True)
            with torch.no_grad():
                preds = torch.max(model(inputs)[0], 1)[1]
                batch_accuracy = (preds == policy_targets).float().mean().item()
                accuracy_per_batch.append(batch_accuracy)
                all_predictions.extend(preds.cpu().numpy())
                all_actuals.extend(policy_targets.cpu().numpy())
            self.progress_callback(done_batches / total_batches * 100)
            self.status_callback(f"Batch {done_batches}/{total_batches} done.")
        del dataset
        torch.cuda.empty_cache()
        all_predictions, all_actuals = np.array(all_predictions), np.array(all_actuals)
        unique_labels = np.unique(np.concatenate((all_actuals, all_predictions)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        mapped_actuals = np.array([label_to_idx[actual] for actual in all_actuals])
        mapped_predictions = np.array([label_to_idx[pred] for pred in all_predictions])
        for actual, pred in zip(mapped_actuals, mapped_predictions):
            confusion_matrix[actual, pred] += 1
        accuracy = np.sum(mapped_predictions == mapped_actuals) / len(mapped_actuals)
        if self.wandb:
            wandb.log({"accuracy": accuracy, "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=mapped_actuals, preds=mapped_predictions, class_names=[str(l) for l in unique_labels])})
            if len(unique_labels) > 1:
                y_probas = np.eye(len(unique_labels))[mapped_predictions]
                indices = np.arange(0, len(y_probas), max(1, len(y_probas) // 1000))
                wandb.log({"pr_curve": wandb.plot.pr_curve(y_true=mapped_actuals[indices], y_probas=y_probas[indices]), "roc_curve": wandb.plot.roc_curve(y_true=mapped_actuals[indices], y_probas=y_probas[indices])})
            binned_counts = np.bincount(np.digitize(mapped_predictions, bins=np.linspace(0, len(unique_labels), min(50, len(unique_labels) + 1))) - 1, minlength=len(unique_labels))
            wandb.log({"prediction_distribution": wandb.plot.bar(table=wandb.Table(data=[[str(i), c] for i, c in enumerate(binned_counts)], columns=["Bin", "Count"]), label="Bin", value="Count", title="Binned Predicted Class Distribution")})
            factor = max(1, len(accuracy_per_batch) // 100)
            wandb.log({"accuracy_per_batch": wandb.plot.line_series(xs=list(range(1, len(accuracy_per_batch) + 1, factor)), ys=[[np.mean(accuracy_per_batch[i:i+factor]) for i in range(0, len(accuracy_per_batch), factor)]], keys=["Batch Accuracy"], title="Accuracy per Batch (Downsampled)")})
            sample_size = min(5000, len(mapped_actuals))
            indices = random.sample(range(len(mapped_actuals)), sample_size) if sample_size < len(mapped_actuals) else slice(None)
            wandb.log({"actual_vs_predicted": wandb.plot.scatter(table=wandb.Table(data=list(zip(mapped_actuals[indices], mapped_predictions[indices])), columns=["Actual", "Predicted"]), x="Actual", y="Predicted", title="Actual vs Predicted Scatter Plot (Downsampled)")})
            wandb.run.summary.update({"accuracy": accuracy})
            wandb.finish()
        return True