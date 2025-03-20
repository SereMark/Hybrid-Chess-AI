import os
import shap
import wandb
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from src.utils.chess_utils import H5Dataset,get_total_moves
from torch.nn.functional import mse_loss,smooth_l1_loss
from src.utils.train_utils import initialize_random_seeds
from src.models.cnn import CNNModel

class EvaluationWorker:
    def __init__(self,model_path,indices_path,h5_path,wandb_flag=False,progress_callback=None,status_callback=None):
        self.model_path=model_path
        self.indices_path=indices_path
        self.h5_path=h5_path
        self.wandb_flag=wandb_flag
        self.device=torch.device("cuda"if torch.cuda.is_available()else"cpu")
        self.progress_callback=progress_callback or (lambda x:None)
        self.status_callback=status_callback or (lambda x:None)
        self.max_scatter_points=5000
    def run(self):
        initialize_random_seeds(42)
        m=self._load_model()
        if m is None:return False
        l=self._prepare_test_loader()
        if l is None:return False
        p,a,ba,lg=self._inference(m,l)
        if len(p)==0 or len(a)==0:
            self.status_callback("No predictions; evaluation aborted.")
            return False
        oa=float(np.mean(p==a))
        ab=float(np.mean(ba))
        self._compute_and_log_metrics(p,a,oa,ab)
        self._gradient_sensitivity_analysis(m,l)
        self._feature_importance_knockout(m,l)
        self._prediction_error_analysis(lg,a)
        self._robustness_test(m)
        self._explainability_tools(m,l)
        self.status_callback(f"Evaluation complete. Overall={oa:.4f}, AvgBatch={ab:.4f}")
        return True
    def _load_model(self):
        if not os.path.isfile(self.model_path):
            self.status_callback(f"Model checkpoint not found: {self.model_path}")
            return None
        try:
            c=torch.load(self.model_path,map_location=self.device)
            m=CNNModel(num_moves=get_total_moves()).to(self.device)
            if isinstance(c,dict)and"model_state_dict"in c:m.load_state_dict(c["model_state_dict"],strict=False)
            else:m.load_state_dict(c,strict=False)
            m.eval()
        except Exception as e:
            self.status_callback(f"Error loading model checkpoint: {e}")
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
            ti=np.load(self.indices_path)
        except Exception as e:
            self.status_callback(f"Error loading indices: {e}")
            return None
        if len(ti)==0:
            self.status_callback("Test set empty, aborting.")
            return None
        d=DataLoader(H5Dataset(self.h5_path,ti),batch_size=1024,shuffle=False,num_workers=0,pin_memory=(self.device.type=="cuda"))
        if len(d)==0:
            self.status_callback("No valid batches in test set.")
            return None
        return d
    def _inference(self,m,loader):
        p,a,ba,lg=[],[],[],[]
        tb=len(loader)
        self.status_callback("Starting evaluation...")
        with torch.no_grad():
            for i,(inp,pol,_) in enumerate(loader,1):
                inp=inp.to(self.device,non_blocking=True)
                pol=pol.to(self.device,non_blocking=True)
                o=m(inp)[0]
                pr=o.argmax(dim=1)
                acc=(pr==pol).float().mean().item()
                ba.append(acc)
                p.extend(pr.cpu().numpy())
                a.extend(pol.cpu().numpy())
                lg.append(o.cpu())
                self.progress_callback(i/tb*100)
                self.status_callback(f"🚀 Evaluating batch {i}/{tb} | Acc={acc:.4f}")
        torch.cuda.empty_cache()
        if lg:
            lgc=torch.cat(lg,dim=0).numpy()
        else:
            lgc=np.array([])
        return np.array(p),np.array(a),ba,lgc
    def _compute_and_log_metrics(self,p,a,oa,ab):
        self.status_callback("Computing advanced metrics...")
        ul=np.unique(np.concatenate([a,p]))
        if len(ul)>1:
            lti={l:i for i,l in enumerate(ul)}
            ai=np.vectorize(lti.get)(a)
            pi=np.vectorize(lti.get)(p)
        else:
            self.status_callback("One-class scenario, skipping confusion matrix.")
            ai,pi=a,p
        if self.wandb_flag:
            wandb.log({"accuracy":oa,"average_batch_accuracy":ab})
            if len(ul)>1 and len(ul)<=50:
                wandb.log({"confusion_matrix":wandb.plot.confusion_matrix(probs=None,y_true=ai,preds=pi,class_names=[str(x)for x in ul])})
            if len(ul)==2:
                oha=np.zeros((len(a),2),dtype=np.float32)
                ohp=np.zeros((len(p),2),dtype=np.float32)
                for i,(x,y) in enumerate(zip(ai,pi)):
                    oha[i,x]=1
                    ohp[i,y]=1
                wandb.log({
                    "pr_curve":wandb.plot.pr_curve(oha,ohp,[str(x)for x in ul]),
                    "roc_curve":wandb.plot.roc_curve(oha,ohp,[str(x)for x in ul])
                })
            c=np.bincount(pi,minlength=len(ul))
            td=[(str(lb),int(ct))for lb,ct in zip(ul,c)] if len(ul)<=50 else[]
            if td:
                tb=wandb.Table(data=td,columns=["Class","Count"])
                wandb.log({"prediction_distribution":wandb.plot.bar(tb,"Class","Count",title="Predicted Class Distribution")})
            si=random.sample(range(len(a)),min(self.max_scatter_points,len(a)))
            sd=list(zip(a[si],p[si]))
            st=wandb.Table(data=sd,columns=["Actual","Predicted"])
            wandb.log({"actual_vs_predicted_scatter":wandb.plot.scatter(st,"Actual","Predicted",title="Actual vs. Predicted")})
            wandb.run.summary.update({"final_accuracy":oa,"average_batch_accuracy":ab})
        self.status_callback(f"Done. Overall accuracy={oa:.4f}.")
    def _gradient_sensitivity_analysis(self,m,loader,num_samples=1):
        try:
            self.status_callback("Performing gradient-based sensitivity analysis...")
            di=iter(loader)
            for _ in range(num_samples):
                inp,pol,_=next(di)
                inp=inp.to(self.device)
                pol=pol.to(self.device)
                inp.requires_grad_(True)
                o=m(inp)[0]
                l=torch.nn.functional.cross_entropy(o,pol.long())
                l.backward()
                g=inp.grad.detach().abs().mean(dim=0).cpu().numpy()
                if self.wandb_flag:
                    bd=[[i,float(v)]for i,v in enumerate(g.mean(axis=0))]
                    bt=wandb.Table(data=bd,columns=["FeatureIndex","GradientMean"])
                    wandb.log({"gradient_sensitivity":wandb.plot.bar(bt,"FeatureIndex","GradientMean",title="Gradient-Based Sensitivity")})
                inp.grad=None
        except StopIteration:pass
        except Exception as e:
            self.status_callback(f"Error in gradient analysis: {e}")
    def _feature_importance_knockout(self,m,loader,knock_idx=0):
        try:
            self.status_callback("Performing feature knockout tests...")
            c=0
            t=0
            for inp,pol,_ in loader:
                inp=inp.to(self.device)
                pol=pol.to(self.device)
                if knock_idx<inp.size(2):
                    inp[:,:,knock_idx]=0
                with torch.no_grad():
                    o=m(inp)[0]
                    pr=o.argmax(dim=1)
                    c+=(pr==pol).float().sum().item()
                    t+=pol.size(0)
            acc=c/t if t else 0
            if self.wandb_flag:wandb.log({"knockout_feature_index":knock_idx,"knockout_accuracy":acc})
            self.status_callback(f"Knockout accuracy with feature {knock_idx} removed: {acc:.4f}")
        except Exception as e:
            self.status_callback(f"Error in feature knockout: {e}")
    def _prediction_error_analysis(self,lg,a):
        try:
            self.status_callback("Analyzing prediction errors with MSE and Huber metrics...")
            if lg.size==0:return
            i=np.arange(len(a))
            sl=lg[i,a]
            st=torch.tensor(sl,dtype=torch.float32)
            me=float(mse_loss(st,torch.zeros_like(st)))
            hu=float(smooth_l1_loss(st,torch.zeros_like(st)))
            if self.wandb_flag:
                wandb.log({"pred_error_mse":me,"pred_error_huber":hu})
            self.status_callback(f"MSE: {me:.4f}, Huber: {hu:.4f}")
        except Exception as e:
            self.status_callback(f"Error in error analysis: {e}")
    def _robustness_test(self,m):
        try:
            self.status_callback("Testing robustness on edge cases...")
            ec=torch.zeros((1,64,144),dtype=torch.float32,device=self.device)
            ec[0,0,0]=1
            with torch.no_grad():
                o=m(ec)[0]
                v=o.argmax(dim=1).cpu().item()
            if self.wandb_flag:
                wandb.log({"edge_case_results":v})
            self.status_callback(f"Edge case result: {v}")
        except Exception as e:
            self.status_callback(f"Error in robustness test: {e}")
    def _explainability_tools(self,m,loader,num_samples=1):
        try:
            self.status_callback("Explaining model decisions with SHAP/LIME if available...")
            di=iter(loader)
            s=[]
            for _ in range(num_samples):
                try:
                    i,_,_=next(di)
                    s.append(i[0].numpy())
                except StopIteration:
                    break
            if not s:return
            def ff(x):
                xt=torch.from_numpy(x).float().to(self.device)
                with torch.no_grad():
                    o=m(xt)[0]
                return o.cpu().numpy()
            ex=shap.Explainer(ff,np.array(s))
            sv=ex(s)
            if self.wandb_flag and hasattr(sv,"values"):
                wandb.log({"shap_values_example":str(sv.values)})
        except Exception as e:
            self.status_callback(f"Error in explainability: {e}")