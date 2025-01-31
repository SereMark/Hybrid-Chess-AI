import os, torch, numpy as np, random
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss, smooth_l1_loss
from src.utils.datasets import H5Dataset
from src.models.transformer import TransformerCNNChessModel
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
        preds, actuals, batch_accs, logits = self._inference(model, loader)
        if len(preds)==0 or len(actuals)==0:
            self.status_callback("No predictions; evaluation aborted.")
            self._wandb_finish()
            return False
        overall_acc = float(np.mean(preds==actuals))
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
            c = torch.load(self.model_path, map_location=self.device)
        except Exception as e:
            self.status_callback(f"Error loading model checkpoint: {e}")
            return None
        m = TransformerCNNChessModel(num_moves=get_total_moves()).to(self.device)
        m.eval()
        try:
            if isinstance(c, dict) and "model_state_dict" in c:
                m.load_state_dict(c["model_state_dict"], strict=False)
            else:
                m.load_state_dict(c, strict=False)
        except Exception as e:
            self.status_callback(f"Error setting model state dict: {e}")
            return None
        return m
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
        if len(test_indices)==0:
            self.status_callback("Test set empty, aborting.")
            return None
        ds = H5Dataset(self.h5_path, test_indices)
        loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=(self.device.type=="cuda"))
        if len(loader)==0:
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
                acc = (pred==pol).float().mean().item()
                batch_accs.append(acc)
                preds.extend(pred.cpu().numpy())
                actuals.extend(pol.cpu().numpy())
                logits_list.append(out.cpu())
                p = (i/total_batches)*100
                self.progress_callback(p)
                self.status_callback(f"Evaluating batch {i}/{total_batches} [acc={acc:.4f}]")
        torch.cuda.empty_cache()
        if logits_list:
            logits_cat = torch.cat(logits_list, dim=0).numpy()
        else:
            logits_cat = np.array([])
        return np.array(preds), np.array(actuals), batch_accs, logits_cat
    def _compute_and_log_metrics(self, preds, actuals, overall_acc, avg_batch_acc):
        self.status_callback("Computing advanced metrics...")
        u = np.unique(np.concatenate([actuals, preds]))
        if len(u)>1:
            lbl_to_ix = {l:i for i,l in enumerate(u)}
            ma = np.vectorize(lbl_to_ix.get)(actuals)
            mp = np.vectorize(lbl_to_ix.get)(preds)
        else:
            self.status_callback("One-class scenario, skipping confusion matrix.")
            ma, mp = actuals, preds
        if self.wandb_flag:
            import wandb
            wandb.log({"accuracy": overall_acc,"average_batch_accuracy": avg_batch_acc})
            if len(u)>1 and len(u)<=50:
                wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,y_true=ma,preds=mp,class_names=[str(x) for x in u])})
            if len(u)==2:
                oh_a = np.zeros((len(actuals),2), dtype=np.float32)
                oh_p = np.zeros((len(preds),2), dtype=np.float32)
                for i,(a,r) in enumerate(zip(ma,mp)):
                    oh_a[i,a] = 1
                    oh_p[i,r] = 1
                wandb.log({"pr_curve":wandb.plot.pr_curve(oh_a, oh_p, [str(x) for x in u]),"roc_curve":wandb.plot.roc_curve(oh_a, oh_p, [str(x) for x in u])})
            c = np.bincount(mp, minlength=len(u))
            tb_data = []
            if len(u)<=50:
                for lbl,ct in zip(u,c):
                    tb_data.append((str(lbl),int(ct)))
            else:
                idxs = np.argsort(-c)
                top_50 = idxs[:50]
                for i in top_50:
                    tb_data.append((str(u[i]),int(c[i])))
                self.status_callback("Logged top 50 predicted classes only.")
            dtb = wandb.Table(data=tb_data, columns=["Class","Count"])
            wandb.log({"prediction_distribution": wandb.plot.bar(dtb,"Class","Count",title="Predicted Class Distribution")})
            smp_idx = random.sample(range(len(actuals)), min(self.max_scatter_points,len(actuals)))
            sc_data = list(zip(actuals[smp_idx], preds[smp_idx]))
            sct = wandb.Table(data=sc_data, columns=["Actual","Predicted"])
            wandb.log({"actual_vs_predicted_scatter": wandb.plot.scatter(sct,"Actual","Predicted",title="Actual vs. Predicted")})
            wandb.run.summary.update({"final_accuracy":overall_acc,"average_batch_accuracy":avg_batch_acc})
        self.status_callback(f"Done. Overall accuracy={overall_acc:.4f}.")
        self._wandb_finish()
    def _gradient_sensitivity_analysis(self, model, loader, num_samples=1):
        try:
            self.status_callback("Performing gradient-based sensitivity analysis...")
            data_iter = iter(loader)
            with torch.enable_grad():
                for _ in range(num_samples):
                    inp, pol, _ = next(data_iter)
                    inp = inp.to(self.device)
                    pol = pol.to(self.device)
                    inp.requires_grad_(True)
                    out = model(inp)[0]
                    loss = torch.nn.functional.cross_entropy(out, pol.long())
                    loss.backward()
                    grad = inp.grad.detach().abs().mean(dim=0).cpu().numpy()
                    if self.wandb_flag:
                        import wandb
                        bar_data = [[i,float(g)] for i,g in enumerate(grad.mean(axis=0))]
                        bar_table = wandb.Table(data=bar_data, columns=["FeatureIndex","GradientMean"])
                        wandb.log({"gradient_sensitivity": wandb.plot.bar(bar_table,"FeatureIndex","GradientMean",title="Gradient-Based Sensitivity")})
                    inp.grad = None
        except StopIteration:
            pass
        except Exception as e:
            self.status_callback(f"Error in gradient analysis: {e}")
    def _feature_importance_knockout(self, model, loader, knock_idx=0):
        try:
            self.status_callback("Performing feature knockout tests...")
            total_correct, total_seen = 0,0
            for inp, pol, _ in loader:
                inp = inp.to(self.device)
                pol = pol.to(self.device)
                if knock_idx<inp.size(2):
                    inp[:,:,knock_idx] = 0
                with torch.no_grad():
                    out = model(inp)[0]
                    pred = out.argmax(dim=1)
                    total_correct += (pred==pol).float().sum().item()
                    total_seen += pol.size(0)
            acc = total_correct/total_seen if total_seen else 0
            if self.wandb_flag:
                import wandb
                wandb.log({"knockout_feature_index": knock_idx,"knockout_accuracy": acc})
            self.status_callback(f"Knockout accuracy with feature {knock_idx} removed: {acc:.4f}")
        except Exception as e:
            self.status_callback(f"Error in feature knockout: {e}")
    def _prediction_error_analysis(self, logits, actuals):
        try:
            self.status_callback("Analyzing prediction errors with MSE and Huber (Smooth L1) metrics...")
            if logits.size==0:
                return
            idx = np.arange(len(actuals))
            selected_logits = logits[idx, actuals] 
            selected_torch = torch.tensor(selected_logits, dtype=torch.float32)
            mse_v = float(mse_loss(selected_torch, torch.zeros_like(selected_torch)))
            huber_v = float(smooth_l1_loss(selected_torch, torch.zeros_like(selected_torch)))
            if self.wandb_flag:
                import wandb
                wandb.log({"pred_error_mse": mse_v,"pred_error_huber": huber_v})
            self.status_callback(f"MSE: {mse_v:.4f}, Huber: {huber_v:.4f}")
        except Exception as e:
            self.status_callback(f"Error in error analysis: {e}")
    def _robustness_test(self, model):
        try:
            self.status_callback("Testing robustness on edge cases (synthetic boards)...")
            edges = []
            b = torch.zeros((1,64,18), dtype=torch.float32, device=self.device)
            b[0,0,0] = 1 
            edges.append(b)
            results = []
            for test_b in edges:
                with torch.no_grad():
                    out = model(test_b)[0]
                    val = out.argmax(dim=1).cpu().item()
                results.append(val)
            if self.wandb_flag:
                import wandb
                wandb.log({"edge_case_results": results[0] if results else -1})
            self.status_callback(f"Edge case results: {results}")
        except Exception as e:
            self.status_callback(f"Error in robustness test: {e}")
    def _explainability_tools(self, model, loader, num_samples=1):
        try:
            self.status_callback("Explaining model decisions with SHAP or LIME if available...")
            import shap
            data_iter = iter(loader)
            samples = []
            for _ in range(num_samples):
                try:
                    inp, _, _ = next(data_iter)
                    samples.append(inp[0].numpy())
                except StopIteration:
                    break
            if not samples: return
            def f_forward(x):
                x_t = torch.from_numpy(x).float().to(self.device)
                with torch.no_grad():
                    out = model(x_t)[0]
                return out.cpu().numpy()
            e = shap.Explainer(f_forward, np.array(samples))
            sv = e(samples)
            if self.wandb_flag:
                import wandb
                shap_values = sv.values if hasattr(sv,"values") else None
                if shap_values is not None:
                    wandb.log({"shap_values_example": str(shap_values)})
        except ImportError:
            self.status_callback("SHAP not installed, skipping explainability.")
        except Exception as e:
            self.status_callback(f"Error in explainability: {e}")
    def _wandb_finish(self):
        if self.wandb_flag:
            import wandb
            try:
                wandb.finish()
            except Exception as e:
                self.status_callback(f"Error finishing wandb: {e}")