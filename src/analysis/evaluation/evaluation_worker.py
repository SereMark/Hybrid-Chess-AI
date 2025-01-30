import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from src.utils.datasets import H5Dataset
from src.models.transformer import TransformerChessModel
from src.utils.train_utils import initialize_random_seeds
from src.utils.chess_utils import get_total_moves

class EvaluationWorker:
    def __init__(
        self,
        model_path: str,
        indices_path: str,
        h5_path: str,
        wandb_flag: bool = False,
        progress_callback=None,
        status_callback=None
    ):
        self.model_path = model_path
        self.indices_path = indices_path
        self.h5_path = h5_path
        self.wandb_flag = wandb_flag

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.progress_callback = progress_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x: None)

        self.max_scatter_points = 5000

    def run(self) -> bool:
        if self.wandb_flag:
            try:
                import wandb
                wandb.init(
                    entity="chess_ai",
                    project="chess_ai_app",
                    name="evaluation",
                    config=self.__dict__,
                    reinit=True
                )
            except ImportError:
                self.status_callback("⚠️ wandb not installed. Proceeding without wandb logging.")
                self.wandb_flag = False

        initialize_random_seeds(42)

        model = self._load_model()
        if model is None:
            self._wandb_finish()
            return False

        loader = self._prepare_test_loader()
        if loader is None:
            self._wandb_finish()
            return False

        preds, actuals, batch_accuracies = self._inference(model, loader)

        if len(preds) == 0 or len(actuals) == 0:
            self.status_callback("❌ No predictions were collected; evaluation aborted.")
            self._wandb_finish()
            return False

        overall_accuracy = float(np.mean(preds == actuals))
        average_batch_accuracy = float(np.mean(batch_accuracies))

        self._compute_and_log_metrics(
            preds=preds,
            actuals=actuals,
            overall_accuracy=overall_accuracy,
            average_batch_accuracy=average_batch_accuracy
        )

        self.status_callback(
            f"✅ Evaluation complete. "
            f"Overall accuracy = {overall_accuracy:.4f}, "
            f"Avg batch accuracy = {average_batch_accuracy:.4f}"
        )
        return True

    def _load_model(self) -> torch.nn.Module:
        if not os.path.isfile(self.model_path):
            self.status_callback(f"❌ Model checkpoint file not found: {self.model_path}")
            return None

        self.status_callback("Loading model checkpoint...")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        except Exception as e:
            self.status_callback(f"❌ Error loading model checkpoint: {e}")
            return None

        model = TransformerChessModel(num_moves=get_total_moves()).to(self.device)
        model.eval()

        try:
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            self.status_callback(f"❌ Error setting model state dict: {e}")
            return None

        return model

    def _prepare_test_loader(self) -> DataLoader:
        self.status_callback("Loading dataset & indices...")

        if not os.path.isfile(self.indices_path):
            self.status_callback(f"❌ Indices file not found: {self.indices_path}")
            return None
        if not os.path.isfile(self.h5_path):
            self.status_callback(f"❌ H5 dataset file not found: {self.h5_path}")
            return None

        try:
            test_indices = np.load(self.indices_path)
        except Exception as e:
            self.status_callback(f"❌ Error loading indices from {self.indices_path}: {e}")
            return None

        if len(test_indices) == 0:
            self.status_callback("❌ Test set is empty! Aborting evaluation.")
            return None

        dataset = H5Dataset(self.h5_path, test_indices)
        loader = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda")
        )

        if len(loader) == 0:
            self.status_callback("❌ No valid batches in the test set! Aborting.")
            return None

        return loader

    def _inference(self, model: torch.nn.Module, loader: DataLoader):
        preds, actuals = [], []
        batch_accuracies = []

        self.status_callback("Starting evaluation...")
        total_batches = len(loader)

        with torch.no_grad():
            for i, (input_tensor, policy_tensor, _) in enumerate(loader, start=1):
                input_tensor = input_tensor.to(self.device, non_blocking=True)
                policy_tensor = policy_tensor.to(self.device, non_blocking=True)

                outputs = model(input_tensor)[0]
                pred_classes = outputs.argmax(dim=1)

                batch_acc = (pred_classes == policy_tensor).float().mean().item()
                batch_accuracies.append(batch_acc)

                preds.extend(pred_classes.cpu().numpy())
                actuals.extend(policy_tensor.cpu().numpy())

                progress_percent = (i / total_batches) * 100
                self.progress_callback(min(100.0, progress_percent))
                self.status_callback(f"Evaluating batch {i}/{total_batches} [acc={batch_acc:.4f}]")

        torch.cuda.empty_cache()

        return np.array(preds), np.array(actuals), batch_accuracies

    def _compute_and_log_metrics(
        self,
        preds: np.ndarray,
        actuals: np.ndarray,
        overall_accuracy: float,
        average_batch_accuracy: float
    ) -> None:
        self.status_callback("Computing confusion matrix and advanced metrics...")

        unique_labels = np.unique(np.concatenate([actuals, preds]))
        if len(unique_labels) <= 1:
            self.status_callback("⚠️ Only one unique label in the test set or predictions. Limited metrics available.")
        else:
            label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            mapped_actuals = np.vectorize(label_to_index.get)(actuals)
            mapped_preds = np.vectorize(label_to_index.get)(preds)

        if self.wandb_flag:
            import wandb
            wandb.log({
                "accuracy": overall_accuracy,
                "average_batch_accuracy": average_batch_accuracy
            })

            if len(unique_labels) > 1 and len(unique_labels) <= 50:
                wandb.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=mapped_actuals,
                        preds=mapped_preds,
                        class_names=[str(lbl) for lbl in unique_labels]
                    )
                })
            else:
                self.status_callback(
                    "⚠️ Too many classes to plot full confusion matrix. Skipping."
                )

            if len(unique_labels) == 2:
                one_hot_actuals = np.zeros((len(actuals), 2), dtype=np.float32)
                one_hot_preds = np.zeros((len(preds), 2), dtype=np.float32)
                for i, (act, pred) in enumerate(zip(mapped_actuals, mapped_preds)):
                    one_hot_actuals[i, act] = 1.0
                    one_hot_preds[i, pred] = 1.0

                wandb.log({
                    "pr_curve": wandb.plot.pr_curve(
                        y_true=one_hot_actuals,
                        y_probas=one_hot_preds,
                        labels=[str(lbl) for lbl in unique_labels]
                    ),
                    "roc_curve": wandb.plot.roc_curve(
                        y_true=one_hot_actuals,
                        y_probas=one_hot_preds,
                        labels=[str(lbl) for lbl in unique_labels]
                    )
                })
            else:
                self.status_callback("Skipping PR/ROC curves for multi-class or single-class scenario.")

            pred_class_counts = np.bincount(mapped_preds, minlength=len(unique_labels))
            if len(unique_labels) <= 50:
                table_data = [
                    (str(lbl), int(cnt)) for lbl, cnt in zip(unique_labels, pred_class_counts)
                ]
            else:
                sorted_indices = np.argsort(-pred_class_counts)
                top_50_indices = sorted_indices[:50]
                table_data = [
                    (str(unique_labels[i]), int(pred_class_counts[i])) for i in top_50_indices
                ]
                self.status_callback("⚠️ Too many classes; only logging top 50 for distribution chart.")

            dist_table = wandb.Table(data=table_data, columns=["Class", "Count"])
            wandb.log({
                "prediction_distribution": wandb.plot.bar(
                    dist_table,
                    "Class",
                    "Count",
                    title="Predicted Class Distribution"
                )
            })

            if len(actuals) > self.max_scatter_points:
                sample_indices = random.sample(range(len(actuals)), self.max_scatter_points)
            else:
                sample_indices = range(len(actuals))

            scatter_data = list(zip(actuals[sample_indices], preds[sample_indices]))
            scatter_table = wandb.Table(data=scatter_data, columns=["Actual", "Predicted"])
            wandb.log({
                "actual_vs_predicted_scatter": wandb.plot.scatter(
                    scatter_table,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs. Predicted Scatter"
                )
            })

            wandb.run.summary.update({
                "final_accuracy": overall_accuracy,
                "average_batch_accuracy": average_batch_accuracy
            })

        self.status_callback(
            f"Confusion matrix & advanced metrics computed. Overall accuracy={overall_accuracy:.4f}."
        )

        self._wandb_finish()

    def _wandb_finish(self) -> None:
        if self.wandb_flag:
            import wandb
            try:
                wandb.finish()
            except Exception as e:
                self.status_callback(f"⚠️ Error finishing wandb run: {e}")