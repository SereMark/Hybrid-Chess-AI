import os
import time
import torch
import wandb
import numpy as np
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from src.utils.checkpoint_manager import CheckpointManager
from src.models.cnn import CNNModel
from src.utils.chess_utils import H5Dataset,get_total_moves
from src.utils.train_utils import initialize_optimizer,initialize_scheduler,initialize_random_seeds,validate_epoch,train_epoch

class SupervisedWorker:
    def __init__(self,epochs,batch_size,lr,weight_decay,checkpoint_interval,dataset_path,train_indices_path,val_indices_path,model_path,optimizer,scheduler,accumulation_steps,num_workers,random_seed,policy_weight,value_weight,grad_clip,momentum,wandb_flag,use_early_stopping=False,early_stopping_patience=5,progress_callback=None,status_callback=None):
        initialize_random_seeds(random_seed)
        self.device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
        self.model=CNNModel(num_moves=get_total_moves()).to(self.device)
        self.optimizer=initialize_optimizer(self.model,optimizer,lr,weight_decay,momentum)
        self.scheduler_type=scheduler
        self.scheduler=None
        self.wandb_flag=wandb_flag
        self.progress_callback=progress_callback
        self.status_callback=status_callback
        self.epochs=epochs
        self.batch_size=batch_size
        self.accumulation_steps=accumulation_steps
        self.policy_weight=policy_weight
        self.value_weight=value_weight
        self.grad_clip=grad_clip
        self.num_workers=num_workers
        self.random_seed=random_seed
        self.start_epoch=1
        self.use_early_stopping=use_early_stopping
        self.early_stopping_patience=early_stopping_patience
        self.early_stop_counter=0
        self.best_val_loss=float('inf')
        self.scaler=GradScaler(enabled=(self.device.type=='cuda'))
        self.checkpoint_manager=CheckpointManager(os.path.join('models','checkpoints','supervised'),'epoch',checkpoint_interval)
        self.dataset_path=dataset_path
        self.train_indices=np.load(train_indices_path)
        self.val_indices=np.load(val_indices_path)
        self.loaded_checkpoint=None
        if model_path and os.path.exists(model_path):
            self.loaded_checkpoint=self.checkpoint_manager.load(model_path,self.device,self.model,self.optimizer,self.scheduler)
            if self.loaded_checkpoint and'epoch'in self.loaded_checkpoint:
                self.start_epoch=self.loaded_checkpoint['epoch']+1
    def run(self):
        if self.wandb_flag:
            wandb.watch(self.model,log="all",log_freq=100)
        train_loader=DataLoader(H5Dataset(self.dataset_path,self.train_indices),batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=(self.device.type=='cuda'),persistent_workers=(self.num_workers>0))
        val_loader=DataLoader(H5Dataset(self.dataset_path,self.val_indices),batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=(self.device.type=='cuda'),persistent_workers=(self.num_workers>0))
        if self.start_epoch>self.epochs:
            raise ValueError("Start epoch exceeds total epochs.")
        total_steps=(self.epochs-self.start_epoch+1)*len(train_loader)
        if not self.scheduler:
            self.scheduler=initialize_scheduler(self.optimizer,self.scheduler_type,total_steps)
            if self.loaded_checkpoint and'scheduler_state_dict'in self.loaded_checkpoint:
                self.scheduler.load_state_dict(self.loaded_checkpoint['scheduler_state_dict'])
        bm=float('inf')
        st=time.time()
        for e in range(self.start_epoch,self.epochs+1):
            tm=train_epoch(self.model,train_loader,self.device,self.scaler,self.optimizer,self.scheduler,e,self.epochs,self.accumulation_steps,True,self.policy_weight,self.value_weight,self.grad_clip,self.progress_callback,self.status_callback,self.wandb_flag)
            vm=validate_epoch(self.model,val_loader,self.device,e,self.epochs,self.progress_callback,self.status_callback)
            cl=self.policy_weight*vm["policy_loss"]+self.value_weight*vm["value_loss"]
            if cl<bm:
                bm=cl
                self.checkpoint_manager.save(self.model,self.optimizer,self.scheduler,e,None)
            if self.checkpoint_manager.checkpoint_interval>0 and e%self.checkpoint_manager.checkpoint_interval==0:
                self.checkpoint_manager.save(self.model,self.optimizer,self.scheduler,e)
            if self.wandb_flag:
                wandb.log({
                    "val/policy_loss":vm["policy_loss"],
                    "val/value_loss":vm["value_loss"],
                    "val/accuracy":vm["accuracy"],
                    "val/composite_loss":cl,
                    "learning_rate":self.scheduler.get_last_lr()[0]
                })
            if self.use_early_stopping:
                if cl<self.best_val_loss:
                    self.best_val_loss=cl
                    self.early_stop_counter=0
                else:
                    self.early_stop_counter+=1
                    if self.early_stop_counter>=self.early_stopping_patience:
                        if self.status_callback:
                            self.status_callback(f"🔴 Early stopping triggered at epoch {e}")
                        break
        fp=os.path.join("models","saved_models","supervised_model.pth")
        self.checkpoint_manager.save(self.model,self.optimizer,self.scheduler,self.epochs,fp)
        t=time.time()-st
        if self.wandb_flag:
            wandb.run.summary.update({"best_composite_loss":bm,"training_time":t})
        return{"best_composite_loss":bm,"training_time":t}