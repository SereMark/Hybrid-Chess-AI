import os
import time
import wandb
import numpy as np
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.drive import get_drive
from src.utils.train import (
    set_seed, get_optimizer, get_scheduler,
    get_device, train_epoch, validate
)
from src.utils.checkpoint import Checkpoint
from src.utils.chess import H5Dataset, get_move_count
from src.model import ChessModel

class SupervisedPipeline:
    def __init__(self, config: Config):
        self.config = config
        
        seed = config.get('project.seed', 42)
        set_seed(seed)
        
        self.device_info = get_device()
        self.device = self.device_info["device"]
        self.device_type = self.device_info["type"]
        
        self.use_tpu = self.device_type == "tpu"
        print(f"Using device: {self.device_type}")
        
        ch = config.get('model.channels', 64)
        self.model = ChessModel(
            moves=get_move_count(), 
            ch=ch,
            use_tpu=self.use_tpu
        ).to(self.device)
        
        self.epochs = config.get('supervised.epochs', 10)
        self.batch = config.get('data.batch', 128)
        self.lr = config.get('supervised.lr', 0.001)
        self.weight_decay = config.get('supervised.weight_decay', 0.0001)
        self.optimizer_type = config.get('supervised.optimizer', 'adamw')
        self.scheduler_type = config.get('supervised.scheduler', 'onecycle')
        self.momentum = config.get('supervised.momentum', 0.9)
        self.accum_steps = config.get('supervised.accum_steps', 1)
        self.policy_weight = config.get('supervised.policy_weight', 1.0)
        self.value_weight = config.get('supervised.value_weight', 1.0)
        self.grad_clip = config.get('supervised.grad_clip', 1.0)
        self.early_stop = config.get('supervised.early_stop', False)
        self.patience = config.get('supervised.patience', 5)
        
        self.optimizer = get_optimizer(
            self.model, self.optimizer_type, self.lr, self.weight_decay, self.momentum
        )
        
        self.use_amp = (
            self.device_type == "gpu" and 
            config.get('hardware.mixed_precision', True)
        )
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.dataset = config.get('data.dataset', 'data/dataset.h5')
        self.train_idx = config.get('data.train_idx', 'data/train_indices.npy')
        self.val_idx = config.get('data.val_idx', 'data/val_indices.npy')
        
        ckpt_dir = os.path.join(
            config.get('paths.models', 'models'), 
            'checkpoints', 
            'supervised'
        )
        ckpt_interval = config.get('training.checkpoint_interval', 5)
        self.ckpt = Checkpoint(ckpt_dir, 'epoch', ckpt_interval)
        
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.scheduler = None
    
    def setup(self):
        try:
            drive = get_drive()
            local_dataset = '/content/drive/MyDrive/chess_ai/data/dataset.h5'
            local_train_idx = '/content/drive/MyDrive/chess_ai/data/train_indices.npy'
            local_val_idx = '/content/drive/MyDrive/chess_ai/data/val_indices.npy'
            
            os.makedirs('/content/drive/MyDrive/chess_ai/data', exist_ok=True)
            
            self.dataset = drive.get_dataset(self.dataset, local_dataset)
            self.train_idx = drive.load(self.train_idx, local_train_idx)
            self.val_idx = drive.load(self.val_idx, local_val_idx)
            
            print(f"Loaded dataset: {self.dataset}")
            print(f"Loaded train indices: {self.train_idx}")
            print(f"Loaded validation indices: {self.val_idx}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using original paths...")
        
        if self.config.get('wandb.enabled', True):
            try:
                wandb.init(
                    project=self.config.get('wandb.project', 'chess_ai'),
                    name=f"supervised_{self.config.mode}_{time.strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "mode": self.config.mode,
                        "epochs": self.epochs,
                        "batch": self.batch,
                        "lr": self.lr,
                        "optimizer": self.optimizer_type,
                        "scheduler": self.scheduler_type,
                        "model_ch": self.config.get('model.channels'),
                        "accum_steps": self.accum_steps,
                        "device": self.device_type,
                        "amp": self.use_amp,
                    }
                )
                wandb.watch(self.model, log="all", log_freq=100)
            except Exception as e:
                print(f"Error initializing wandb: {e}")
    
    def run(self):
        self.setup()
        
        try:
            train_idx = np.load(self.train_idx)
            val_idx = np.load(self.val_idx)
            
            workers = self.config.get('hardware.workers', 2)
            pin_memory = self.config.get('hardware.pin_memory', True) and not self.use_tpu
            prefetch = self.config.get('hardware.prefetch', 2)
            
            train_dataset = H5Dataset(self.dataset, train_idx)
            val_dataset = H5Dataset(self.dataset, val_idx)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch,
                shuffle=True,
                num_workers=workers,
                pin_memory=pin_memory,
                persistent_workers=(workers > 0),
                prefetch_factor=prefetch if workers > 0 else None
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_memory,
                persistent_workers=(workers > 0),
                prefetch_factor=prefetch if workers > 0 else None
            )
            
            print(f"Training samples: {len(train_idx)}")
            print(f"Validation samples: {len(val_idx)}")
        except Exception as e:
            print(f"Error preparing data loaders: {e}")
            return False
        
        total_steps = (self.epochs - self.start_epoch + 1) * len(train_loader)
        self.scheduler = get_scheduler(
            self.optimizer, self.scheduler_type, total_steps
        )
        
        start_time = time.time()
        best_metric = float('inf')
        
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                epoch_start = time.time()
                
                print(f"\nEpoch {epoch}/{self.epochs}")
                print("-" * 30)
                
                train_metrics = train_epoch(
                    self.model, 
                    train_loader,
                    self.device_info,
                    self.optimizer,
                    self.policy_weight,
                    self.value_weight,
                    self.accum_steps,
                    self.grad_clip,
                    self.scheduler
                )
                
                val_metrics = validate(
                    self.model,
                    val_loader,
                    self.device_info
                )
                
                epoch_time = time.time() - epoch_start
                
                combined_loss = (
                    self.policy_weight * val_metrics["policy_loss"] + 
                    self.value_weight * val_metrics["value_loss"]
                )
                
                print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
                print(f"Train - P_Loss: {train_metrics['policy_loss']:.4f}, "
                      f"V_Loss: {train_metrics['value_loss']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}")
                print(f"Valid - P_Loss: {val_metrics['policy_loss']:.4f}, "
                      f"V_Loss: {val_metrics['value_loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}")
                print(f"Combined Loss: {combined_loss:.4f}")
                
                if wandb.run is not None:
                    wandb.log({
                        "epoch": epoch,
                        "train/policy_loss": train_metrics["policy_loss"],
                        "train/value_loss": train_metrics["value_loss"],
                        "train/accuracy": train_metrics["accuracy"],
                        "val/policy_loss": val_metrics["policy_loss"],
                        "val/value_loss": val_metrics["value_loss"],
                        "val/accuracy": val_metrics["accuracy"],
                        "val/combined_loss": combined_loss,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "epoch_time": epoch_time
                    })
                
                if combined_loss < best_metric:
                    best_metric = combined_loss
                    print(f"New best model with loss: {best_metric:.4f}")
                    self.ckpt.save(
                        self.model, self.optimizer, self.scheduler, epoch, tag="best"
                    )
                
                if self.ckpt.interval > 0 and epoch % self.ckpt.interval == 0:
                    self.ckpt.save(
                        self.model, self.optimizer, self.scheduler, epoch
                    )
                
                if self.early_stop:
                    if combined_loss < self.best_loss:
                        self.best_loss = combined_loss
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        if self.early_stop_counter >= self.patience:
                            print(f"Early stopping at epoch {epoch}")
                            break
            
            final_path = os.path.join('/content/drive/MyDrive/chess_ai/models', 'supervised_model.pth')
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            self.ckpt.save(
                self.model, self.optimizer, self.scheduler, self.epochs, final_path
            )
            
            try:
                drive = get_drive()
                drive_model_path = os.path.join('models', 'supervised_model.pth')
                drive.save(final_path, drive_model_path)
                print(f"Saved final model to Drive: {drive_model_path}")
            except Exception as e:
                print(f"Error saving model to Drive: {e}")
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f}s")
            print(f"Best validation loss: {best_metric:.4f}")
            
            if wandb.run is not None:
                wandb.run.summary.update({
                    "best_loss": best_metric,
                    "total_time": training_time
                })
                wandb.finish()
            
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            if wandb.run is not None:
                wandb.finish()
            return False