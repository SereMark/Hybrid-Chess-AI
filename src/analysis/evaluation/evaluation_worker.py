import os
import torch
import numpy as np
import random
import time
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss, smooth_l1_loss
from src.utils.train_utils import initialize_random_seeds
from src.utils.chess_utils import H5Dataset
from src.utils.common import load_model_from_checkpoint, wandb_log
try:
    import wandb
except ImportError:
    wandb = None

class EvaluationWorker:
    def __init__(self, model_path, indices_path, h5_path, wandb_flag=False,
                 progress_callback=None, status_callback=None):
        self.model_path = model_path
        self.indices_path = indices_path
        self.h5_path = h5_path
        self.wandb_flag = wandb_flag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.progress_callback = progress_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x: None)
        self.max_scatter_points = 5000

    def run(self):
        initialize_random_seeds(42)
        model = self._load_model()
        if model is None:
            return False
        loader = self._prepare_test_loader()
        if loader is None:
            return False
        preds, actuals, batch_accs, logits = self._inference(model, loader)
        if len(preds) == 0 or len(actuals) == 0:
            self.status_callback("No predictions; evaluation aborted.")
            return False
        overall_acc = float(np.mean(preds == actuals))
        avg_batch_acc = float(np.mean(batch_accs))
        self._compute_and_log_metrics(preds, actuals, overall_acc, avg_batch_acc)
        self._gradient_sensitivity_analysis(model, loader)
        self._feature_importance_knockout(model, loader)
        self._prediction_error_analysis(logits, actuals)
        self._robustness_test(model)
        self._explainability_tools(model, loader)
        self.status_callback(f"Evaluation complete. Overall={overall_acc:.4f}, AvgBatch={avg_batch_acc:.4f}")
        return True

    def _load_model(self):
        if not os.path.isfile(self.model_path):
            self.status_callback(f"Model checkpoint not found: {self.model_path}")
            return None
        try:
            model = load_model_from_checkpoint(self.model_path, self.device)
        except Exception as e:
            self.status_callback(f"Error loading model checkpoint: {e}")
            return None
        return model

    def _prepare_test_loader(self):
        if not os.path.isfile(self.indices_path):
            self.status_callback(f"Indices file not found: {self.indices_path}")
            return None
        if not os.path.isfile(self.h5_path):
            self.status_callback(f"H5 dataset file not found: {self.h5_path}")
            return None
        try:
            test_indices = np.load(self.indices_path)
        except Exception as e:
            self.status_callback(f"Error loading indices: {e}")
            return None
        if len(test_indices) == 0:
            self.status_callback("Test set empty, aborting.")
            return None
        loader = DataLoader(H5Dataset(self.h5_path, test_indices), batch_size=1024,
                            shuffle=False, num_workers=0, pin_memory=(self.device.type == "cuda"))
        if len(loader) == 0:
            self.status_callback("No valid batches in test set.")
            return None
        return loader

    def _inference(self, model, loader):
        preds, actuals, batch_accs, logits_list = [], [], [], []
        total_batches = len(loader)
        self.status_callback("Starting evaluation...")
        with torch.no_grad():
            for i, (inp, pol, _) in enumerate(loader, start=1):
                inp = inp.to(self.device, non_blocking=True)
                pol = pol.to(self.device, non_blocking=True)
                out = model(inp)[0]
                pred = out.argmax(dim=1)
                acc = (pred == pol).float().mean().item()
                batch_accs.append(acc)
                preds.extend(pred.cpu().numpy())
                actuals.extend(pol.cpu().numpy())
                logits_list.append(out.cpu())
                self.progress_callback((i / total_batches) * 100)
                self.status_callback(f"🚀 Evaluating batch {i}/{total_batches} | Acc={acc:.4f}")
        torch.cuda.empty_cache()
        logits_cat = torch.cat(logits_list, dim=0).numpy() if logits_list else np.array([])
        return np.array(preds), np.array(actuals), batch_accs, logits_cat

    def _compute_and_log_metrics(self, preds, actuals, overall_acc, avg_batch_acc):
        self.status_callback("Computing advanced metrics...")
        unique_labels = np.unique(np.concatenate([actuals, preds]))
        if len(unique_labels) > 1:
            label_to_index = {l: i for i, l in enumerate(unique_labels)}
            actual_indices = np.vectorize(label_to_index.get)(actuals)
            pred_indices = np.vectorize(label_to_index.get)(preds)
        else:
            self.status_callback("One-class scenario, skipping confusion matrix.")
            actual_indices, pred_indices = actuals, preds
        if self.wandb_flag and wandb is not None:
            wandb_log({"accuracy": overall_acc, "average_batch_accuracy": avg_batch_acc})
            if len(unique_labels) > 1 and len(unique_labels) <= 50:
                wandb_log({"confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=actual_indices, preds=pred_indices,
                    class_names=[str(x) for x in unique_labels])})
            if len(unique_labels) == 2:
                one_hot_actual = np.zeros((len(actuals), 2), dtype=np.float32)
                one_hot_pred = np.zeros((len(preds), 2), dtype=np.float32)
                for i, (a, r) in enumerate(zip(actual_indices, pred_indices)):
                    one_hot_actual[i, a] = 1
                    one_hot_pred[i, r] = 1
                wandb_log({
                    "pr_curve": wandb.plot.pr_curve(one_hot_actual, one_hot_pred, [str(x) for x in unique_labels]),
                    "roc_curve": wandb.plot.roc_curve(one_hot_actual, one_hot_pred, [str(x) for x in unique_labels])
                })
            counts = np.bincount(pred_indices, minlength=len(unique_labels))
            table_data = [(str(lbl), int(ct)) for lbl, ct in zip(unique_labels, counts)] if len(unique_labels) <= 50 else []
            if self.wandb_flag and wandb is not None and table_data:
                tb = wandb.Table(data=table_data, columns=["Class", "Count"])
                wandb_log({"prediction_distribution": wandb.plot.bar(tb, "Class", "Count", title="Predicted Class Distribution")})
            sample_indices = random.sample(range(len(actuals)), min(self.max_scatter_points, len(actuals)))
            scatter_data = list(zip(actuals[sample_indices], preds[sample_indices]))
            sct = wandb.Table(data=scatter_data, columns=["Actual", "Predicted"])
            wandb_log({"actual_vs_predicted_scatter": wandb.plot.scatter(sct, "Actual", "Predicted", title="Actual vs. Predicted")})
            wandb.run.summary.update({"final_accuracy": overall_acc, "average_batch_accuracy": avg_batch_acc})
        self.status_callback(f"Done. Overall accuracy={overall_acc:.4f}.")

    def _gradient_sensitivity_analysis(self, model, loader, num_samples=1):
        try:
            self.status_callback("Performing gradient-based sensitivity analysis...")
            data_iter = iter(loader)
            for _ in range(num_samples):
                inp, pol, _ = next(data_iter)
                inp = inp.to(self.device)
                pol = pol.to(self.device)
                inp.requires_grad_(True)
                out = model(inp)[0]
                loss = torch.nn.functional.cross_entropy(out, pol.long())
                loss.backward()
                grad = inp.grad.detach().abs().mean(dim=0).cpu().numpy()
                if self.wandb_flag and wandb is not None:
                    bar_data = [[i, float(g)] for i, g in enumerate(grad.mean(axis=0))]
                    bar_table = wandb.Table(data=bar_data, columns=["FeatureIndex", "GradientMean"])
                    wandb_log({"gradient_sensitivity": wandb.plot.bar(bar_table, "FeatureIndex", "GradientMean", title="Gradient-Based Sensitivity")})
                inp.grad = None
        except StopIteration:
            pass
        except Exception as e:
            self.status_callback(f"Error in gradient analysis: {e}")

    def _feature_importance_knockout(self, model, loader, knock_idx=0):
        try:
            self.status_callback("Performing feature knockout tests...")
            total_correct, total_seen = 0, 0
            for inp, pol, _ in loader:
                inp = inp.to(self.device)
                pol = pol.to(self.device)
                if knock_idx < inp.size(2):
                    inp[:, :, knock_idx] = 0
                with torch.no_grad():
                    out = model(inp)[0]
                    pred = out.argmax(dim=1)
                    total_correct += (pred == pol).float().sum().item()
                    total_seen += pol.size(0)
            acc = total_correct / total_seen if total_seen else 0
            if self.wandb_flag and wandb is not None:
                wandb_log({"knockout_feature_index": knock_idx, "knockout_accuracy": acc})
            self.status_callback(f"Knockout accuracy with feature {knock_idx} removed: {acc:.4f}")
        except Exception as e:
            self.status_callback(f"Error in feature knockout: {e}")

    def _prediction_error_analysis(self, logits, actuals):
        try:
            self.status_callback("Analyzing prediction errors with MSE and Huber metrics...")
            if logits.size == 0:
                return
            idx = np.arange(len(actuals))
            selected_logits = logits[idx, actuals]
            selected_torch = torch.tensor(selected_logits, dtype=torch.float32)
            mse_value = float(mse_loss(selected_torch, torch.zeros_like(selected_torch)))
            huber_value = float(smooth_l1_loss(selected_torch, torch.zeros_like(selected_torch)))
            if self.wandb_flag and wandb is not None:
                wandb_log({"pred_error_mse": mse_value, "pred_error_huber": huber_value})
            self.status_callback(f"MSE: {mse_value:.4f}, Huber: {huber_value:.4f}")
        except Exception as e:
            self.status_callback(f"Error in error analysis: {e}")

    def _robustness_test(self, model):
        try:
            self.status_callback("Testing robustness on edge cases...")
            edge_case = torch.zeros((1, 64, 144), dtype=torch.float32, device=self.device)
            edge_case[0, 0, 0] = 1
            with torch.no_grad():
                out = model(edge_case)[0]
                val = out.argmax(dim=1).cpu().item()
            if self.wandb_flag and wandb is not None:
                wandb_log({"edge_case_results": val})
            self.status_callback(f"Edge case result: {val}")
        except Exception as e:
            self.status_callback(f"Error in robustness test: {e}")

    def _explainability_tools(self, model, loader, num_samples=1):
        try:
            self.status_callback("Explaining model decisions with SHAP/LIME if available...")
            import shap
            data_iter = iter(loader)
            samples = []
            for _ in range(num_samples):
                try:
                    inp, _, _ = next(data_iter)
                    samples.append(inp[0].numpy())
                except StopIteration:
                    break
            if not samples:
                return
            def forward_func(x):
                x_tensor = torch.from_numpy(x).float().to(self.device)
                with torch.no_grad():
                    out = model(x_tensor)[0]
                return out.cpu().numpy()
            explainer = shap.Explainer(forward_func, np.array(samples))
            shap_values = explainer(samples)
            if self.wandb_flag and wandb is not None and hasattr(shap_values, "values"):
                wandb_log({"shap_values_example": str(shap_values.values)})
        except ImportError:
            self.status_callback("SHAP not installed, skipping explainability.")
        except Exception as e:
            self.status_callback(f"Error in explainability: {e}")