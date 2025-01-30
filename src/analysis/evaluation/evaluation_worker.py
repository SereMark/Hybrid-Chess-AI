import os, torch, numpy as np, random
from torch.utils.data import DataLoader
from src.utils.datasets import H5Dataset
from src.models.transformer import SimpleTransformerChessModel
from src.utils.train_utils import initialize_random_seeds
from src.utils.chess_utils import get_total_moves

class EvaluationWorker:
    def __init__(self, model_path, indices_path, h5_path, wandb_flag=False, progress_callback=None, status_callback=None):
        self.model_path = model_path
        self.indices_path = indices_path
        self.h5_path = h5_path
        self.wandb_flag = wandb_flag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.progress_callback = progress_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x: None)
        self.max_scatter_points = 5000
    def run(self):
        if self.wandb_flag:
            try:
                import wandb
                wandb.init(entity="chess_ai", project="chess_ai_app", name="evaluation", config=self.__dict__, reinit=True)
            except ImportError:
                self.status_callback("wandb not installed, proceeding without logging.")
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
        preds, actuals, batch_accs = self._inference(model, loader)
        if len(preds) == 0 or len(actuals) == 0:
            self.status_callback("No predictions; evaluation aborted.")
            self._wandb_finish()
            return False
        overall_acc = float(np.mean(preds == actuals))
        avg_batch_acc = float(np.mean(batch_accs))
        self._compute_and_log_metrics(preds, actuals, overall_acc, avg_batch_acc)
        self.status_callback(f"Evaluation complete. Overall={overall_acc:.4f}, AvgBatch={avg_batch_acc:.4f}")
        return True
    def _load_model(self):
        if not os.path.isfile(self.model_path):
            self.status_callback(f"Model checkpoint not found: {self.model_path}")
            return None
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        except Exception as e:
            self.status_callback(f"Error loading model checkpoint: {e}")
            return None
        model = SimpleTransformerChessModel(num_moves=get_total_moves()).to(self.device)
        model.eval()
        try:
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            self.status_callback(f"Error setting model state dict: {e}")
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
        ds = H5Dataset(self.h5_path, test_indices)
        loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=(self.device.type=="cuda"))
        if len(loader) == 0:
            self.status_callback("No valid batches in test set.")
            return None
        return loader
    def _inference(self, model, loader):
        preds, actuals, batch_accs = [], [], []
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
                p = (i / total_batches) * 100
                self.progress_callback(p)
                self.status_callback(f"Evaluating batch {i}/{total_batches} [acc={acc:.4f}]")
        torch.cuda.empty_cache()
        return np.array(preds), np.array(actuals), batch_accs
    def _compute_and_log_metrics(self, preds, actuals, overall_acc, avg_batch_acc):
        self.status_callback("Computing advanced metrics...")
        unique_lbls = np.unique(np.concatenate([actuals, preds]))
        if len(unique_lbls) > 1:
            lbl_to_ix = {l: i for i, l in enumerate(unique_lbls)}
            mapped_actuals = np.vectorize(lbl_to_ix.get)(actuals)
            mapped_preds = np.vectorize(lbl_to_ix.get)(preds)
        else:
            self.status_callback("One-class scenario, skipping confusion matrix.")
            mapped_actuals = actuals
            mapped_preds = preds
        if self.wandb_flag:
            import wandb
            wandb.log({"accuracy": overall_acc, "average_batch_accuracy": avg_batch_acc})
            if len(unique_lbls) > 1 and len(unique_lbls) <= 50:
                wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,y_true=mapped_actuals,preds=mapped_preds,class_names=[str(x) for x in unique_lbls])})
            if len(unique_lbls) == 2:
                oh_actuals = np.zeros((len(actuals), 2), dtype=np.float32)
                oh_preds = np.zeros((len(preds), 2), dtype=np.float32)
                for i, (a, r) in enumerate(zip(mapped_actuals, mapped_preds)):
                    oh_actuals[i, a] = 1
                    oh_preds[i, r] = 1
                wandb.log({
                    "pr_curve": wandb.plot.pr_curve(oh_actuals, oh_preds, [str(x) for x in unique_lbls]),
                    "roc_curve": wandb.plot.roc_curve(oh_actuals, oh_preds, [str(x) for x in unique_lbls])
                })
            cnts = np.bincount(mapped_preds, minlength=len(unique_lbls))
            tb_data = []
            if len(unique_lbls) <= 50:
                for lbl, c in zip(unique_lbls, cnts):
                    tb_data.append((str(lbl), int(c)))
            else:
                idxs = np.argsort(-cnts)
                top_50 = idxs[:50]
                for i in top_50:
                    tb_data.append((str(unique_lbls[i]), int(cnts[i])))
                self.status_callback("Logged top 50 predicted classes only.")
            dist_tb = wandb.Table(data=tb_data, columns=["Class", "Count"])
            wandb.log({"prediction_distribution": wandb.plot.bar(dist_tb,"Class","Count",title="Predicted Class Distribution")})
            sample_idx = random.sample(range(len(actuals)), min(self.max_scatter_points, len(actuals)))
            sc_data = list(zip(actuals[sample_idx], preds[sample_idx]))
            sc_table = wandb.Table(data=sc_data, columns=["Actual","Predicted"])
            wandb.log({"actual_vs_predicted_scatter": wandb.plot.scatter(sc_table,"Actual","Predicted",title="Actual vs. Predicted")})
            wandb.run.summary.update({"final_accuracy": overall_acc,"average_batch_accuracy": avg_batch_acc})
        self.status_callback(f"Done. Overall accuracy={overall_acc:.4f}.")
        self._wandb_finish()
    def _wandb_finish(self):
        if self.wandb_flag:
            import wandb
            try:
                wandb.finish()
            except Exception as e:
                self.status_callback(f"Error finishing wandb: {e}")