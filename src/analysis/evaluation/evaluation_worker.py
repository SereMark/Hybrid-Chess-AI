import os,random,torch,wandb,shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss,smooth_l1_loss,cross_entropy
from src.utils.chess_utils import H5Dataset,get_total_moves
from src.utils.train_utils import initialize_random_seeds
from src.models.cnn import CNNModel

class EvaluationWorker:
    def __init__(self,model_path,indices_path,h5_path,wandb_flag=False,progress_callback=None,status_callback=None):
        self.mp=model_path;self.ip=indices_path;self.hp=h5_path;self.wb_flag=wandb_flag
        self.dev=torch.device('cuda'if torch.cuda.is_available()else'cpu')
        self.pc=progress_callback or (lambda *_:None);self.sc=status_callback or (lambda *_:None)
        self.msp=7500;self._wb=None
    @property
    def wb(self):
        if not self.wb_flag:return None
        return self._wb
    def run(self):
        initialize_random_seeds(42)
        m=self._load_model();l=self._prep_loader()
        if m is None or l is None:return False
        p,a,ba,lg=self._infer(m,l)
        if len(p)==0:return False
        oa=float(np.mean(p==a));aba=float(np.mean(ba))
        self._metrics(p,a,oa,aba);self._grad(m,l);self._knock(m,l);self._err(lg,a);self._edge(m);self._shap(m,l)
        self.sc(f'Evaluation complete. Overall={oa:.4f}, AvgBatch={aba:.4f}')
        if self.wb:self.wb.finish()
        return True
    def _load_model(self):
        if not os.path.isfile(self.mp):self.sc(f'Model missing: {self.mp}');return None
        try:
            ck=torch.load(self.mp,map_location=self.dev)
            m=CNNModel(num_moves=get_total_moves()).to(self.dev)
            m.load_state_dict(ck['model_state_dict'] if isinstance(ck,dict)and'model_state_dict'in ck else ck,strict=False)
            m.eval();return m
        except Exception as e:self.sc(f'Load err: {e}');return None
    def _prep_loader(self):
        for p,_ in[(self.ip,'Idx'),(self.hp,'H5')]:
            if not os.path.isfile(p):self.sc('File missing');return None
        try:ti=np.load(self.ip)
        except Exception as e:self.sc(f'Idx err: {e}');return None
        if len(ti)==0:self.sc('Empty test');return None
        return DataLoader(H5Dataset(self.hp,ti),batch_size=1024,shuffle=False,num_workers=0,pin_memory=(self.dev.type=='cuda'))
    def _infer(self,m,l):
        p,a,ba,lg=[],[],[],[];tb=len(l);self.sc('Eval...')
        with torch.no_grad():
            for bi,(x,y,_) in enumerate(l,1):
                x,y=x.to(self.dev,non_blocking=True),y.to(self.dev,non_blocking=True)
                o=m(x)[0];pr=o.argmax(1)
                acc=(pr==y).float().mean().item()
                p+=pr.cpu().numpy().tolist();a+=y.cpu().numpy().tolist();lg.append(o.cpu());ba.append(acc)
                self.pc(bi/tb*100);self.sc(f'🚀 Batch {bi}/{tb}|Acc={acc:.4f}')
                if self.wb:self.wb.log({'batch_idx':bi,'batch_accuracy':acc})
        lgc=torch.cat(lg,0).numpy() if lg else np.empty((0,))
        torch.cuda.empty_cache();return np.array(p),np.array(a),ba,lgc
    def _cm_fig(self,cm,cls):
        f,a=plt.subplots(figsize=(6,5));im=a.imshow(cm);a.figure.colorbar(im,ax=a)
        a.set(xticks=range(len(cls)),yticks=range(len(cls)),xticklabels=cls,yticklabels=cls,ylabel='True',xlabel='Pred',title='Confusion')
        plt.setp(a.get_xticklabels(),rotation=45,ha='right')
        th=cm.max()/2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):a.text(j,i,cm[i,j],ha='center',va='center',color='white'if cm[i,j]>th else'black')
        f.tight_layout();return f
    def _pr_fig(self,y,t):
        pr,rc,_=metrics.precision_recall_curve(y,t);ap=metrics.average_precision_score(y,t)
        f,a=plt.subplots();a.plot(rc,pr);a.set(xlabel='Recall',ylabel='Precision',title=f'PR(AP={ap:.3f})');return f
    def _roc_fig(self,y,t):
        fpr,tpr,_=metrics.roc_curve(y,t);auc=metrics.auc(fpr,tpr)
        f,a=plt.subplots();a.plot(fpr,tpr);a.plot([0,1],[0,1],'--');a.set(xlabel='FPR',ylabel='TPR',title=f'ROC(AUC={auc:.3f})');return f
    def _metrics(self,p,a,oa,aba):
        wb=self.wb;cls=np.unique(np.concatenate([a,p]));sc={'overall_accuracy':oa,'average_batch_accuracy':aba}
        if wb:wb.log(sc);wb.summary.update(sc)
        if len(cls)<=50:
            cm=metrics.confusion_matrix(a,p,labels=cls)
            if wb:wb.log({'cm':wandb.Image(self._cm_fig(cm,[str(c)for c in cls]))})
            pc=cm.diagonal()/cm.sum(1,where=cm.sum(1)!=0)
            f,ax=plt.subplots(figsize=(8,4));ax.bar(range(len(cls)),pc);ax.set(xticks=range(len(cls)),xticklabels=[str(c)for c in cls],ylabel='Acc',xlabel='Class',title='Per‑class');plt.setp(ax.get_xticklabels(),rotation=45,ha='right');f.tight_layout()
            if wb:wb.log({'per_class':wandb.Image(f)});plt.close(f)
        if len(cls)==2:
            yt=(a==cls[1]).astype(int);ys=(p==cls[1]).astype(int)
            if wb:wb.log({'pr':wandb.Image(self._pr_fig(yt,ys)),'roc':wandb.Image(self._roc_fig(yt,ys))})
        if wb and len(a)>2:
            idx=random.sample(range(len(a)),min(self.msp,len(a)));tab=wandb.Table(data=list(zip(a[idx],p[idx])),columns=['A','P'])
            wb.log({'scatter':wandb.plot.scatter(tab,'A','P',title='A vs P')})
    def _grad(self,m,l,n=1):
        try:
            self.sc('Grad...');it=iter(l)
            for _ in range(n):
                x,y,_=next(it);x,y=x.to(self.dev),y.to(self.dev);x.requires_grad_(True)
                loss=cross_entropy(m(x)[0],y.long());loss.backward()
                g=x.grad.abs().mean(0).cpu().numpy()
                f,ax=plt.subplots(figsize=(6,3));ax.bar(range(g.shape[-1]),g.mean(0));ax.set(xlabel='Feat',ylabel='|dL/dx|',title='Grad');f.tight_layout()
                if self.wb:self.wb.log({'grad':wandb.Image(f)});plt.close(f);x.grad=None
        except Exception as e:self.sc(f'Grad err:{e}')
    def _knock(self,m,l,k=0):
        try:
            self.sc('Knockout...');c=t=0
            for x,y,_ in l:
                x,y=x.to(self.dev),y.to(self.dev)
                if k<x.size(2):x[:,:,k]=0
                with torch.no_grad():pr=m(x)[0].argmax(1);c+=(pr==y).float().sum().item();t+=y.size(0)
            acc=c/t if t else 0
            if self.wb:self.wb.log({'ko_idx':k,'ko_acc':acc})
            self.sc(f'Knock idx{k}acc{acc:.4f}')
        except Exception as e:self.sc(f'Knock err:{e}')
    def _err(self,lg,a):
        try:
            self.sc('Err...')
            if lg.size==0:return
            s=lg[np.arange(len(a)),a]
            mse=float(mse_loss(torch.tensor(s),torch.zeros_like(torch.tensor(s))))
            hub=float(smooth_l1_loss(torch.tensor(s),torch.zeros_like(torch.tensor(s))))
            if self.wb:self.wb.log({'mse':mse,'huber':hub})
            self.sc(f'MSE{mse:.4f}Huber{hub:.4f}')
        except Exception as e:self.sc(f'Err met{e}')
    def _edge(self,m):
        try:
            self.sc('Edge...');ec=torch.zeros((1,64,144),device=self.dev);ec[0,0,0]=1
            with torch.no_grad():v=m(ec)[0].argmax(1).item()
            if self.wb:self.wb.log({'edge_pred':v})
            self.sc(f'Edge{v}')
        except Exception as e:self.sc(f'Edge err:{e}')
    def _shap(self,m,l,n=1):
        try:
            self.sc('SHAP...');samp=[];it=iter(l)
            for _ in range(n):
                try:x,_,_=next(it);samp.append(x[0].numpy())
                except StopIteration:break
            if not samp:return
            def f(xn):xt=torch.from_numpy(xn).float().to(self.dev);return m(xt)[0].cpu().numpy()
            sv=shap.Explainer(f,np.array(samp))(samp)
            if self.wb and hasattr(sv,'values'):
                fig=shap.plots.bar(sv,show=False);self.wb.log({'shap':wandb.Image(fig)});plt.close('all')
        except Exception as e:self.sc(f'SHAP err:{e}')