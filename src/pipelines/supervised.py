import os
import time
import wandb
import numpy as np
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.train import (
    set_seed, get_optimizer, get_scheduler,
    get_device, train_epoch, validate
)
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
        
        print(f"Using device: {self.device_type}")
        
        channels = config.get('model.channels', 64)
        blocks = config.get('model.blocks', 4)
        use_attention = config.get('model.attention', True)
        
        self.model = ChessModel(
            moves=get_move_count(), 
            ch=channels,
            blocks=blocks,
            use_attn=use_attention
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
        self.min_accuracy_threshold = config.get('supervised.min_accuracy_threshold', 0.1)
        
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
        
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.scheduler = None
    
    def setup(self):
        try:
            local_dataset = '/content/drive/MyDrive/chess_ai/data/dataset.h5'
            local_train_idx = '/content/drive/MyDrive/chess_ai/data/train_indices.npy'
            local_val_idx = '/content/drive/MyDrive/chess_ai/data/val_indices.npy'
            
            os.makedirs('/content/drive/MyDrive/chess_ai/data', exist_ok=True)
            
            self.dataset = local_dataset
            self.train_idx = local_train_idx
            self.val_idx = local_val_idx
            
            print(f"Using dataset: {self.dataset}")
            print(f"Using train indices: {self.train_idx}")
            print(f"Using validation indices: {self.val_idx}")
        except Exception as e:
            print(f"Error accessing dataset files: {e}")
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
                        "model_blocks": self.config.get('model.blocks'),
                        "model_attention": self.config.get('model.attention'),
                        "accum_steps": self.accum_steps,
                        "device": self.device_type,
                        "amp": self.use_amp,
                        "min_accuracy_threshold": self.min_accuracy_threshold,
                    }
                )
                wandb.watch(self.model, log="all", log_freq=100)
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                
        return True
    
    def run(self):
        self.setup()
        
        try:
            train_idx = np.load(self.train_idx)
            val_idx = np.load(self.val_idx)
            
            workers = self.config.get('hardware.workers', 2)
            pin_memory = self.config.get('hardware.pin_memory', True)
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
        final_accuracy = 0.0
        
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
                final_accuracy = val_metrics["accuracy"]
                
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
            
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'epoch': self.epochs,
                'best_loss': best_metric,
            }, final_path)
            
            print(f"Saved final model to: {final_path}")
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f}s")
            print(f"Best validation loss: {best_metric:.4f}")
            
            passed_sanity_check = final_accuracy >= self.min_accuracy_threshold
            sanity_check_status = "PASSED" if passed_sanity_check else "FAILED"
            print(f"\n{'='*50}")
            print(f"SANITY CHECK {sanity_check_status}")
            print(f"Final validation accuracy: {final_accuracy:.4f}")
            print(f"Minimum required accuracy: {self.min_accuracy_threshold:.4f}")
            print(f"{'='*50}")
            
            if wandb.run is not None:
                wandb.run.summary.update({
                    "best_loss": best_metric,
                    "final_accuracy": final_accuracy,
                    "sanity_check_passed": passed_sanity_check,
                    "total_time": training_time
                })
                wandb.finish()
            
            if not passed_sanity_check:
                print("Supervised model failed sanity check! Accuracy is below threshold.")
                print("Recommend reviewing hyperparameters before proceeding to reinforcement learning.")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            if wandb.run is not None:
                wandb.finish()
            return False