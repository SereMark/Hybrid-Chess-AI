import os, torch, numpy as np
from typing import Optional, Dict
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from src.utils.datasets import H5Dataset
from src.models.transformer import TransformerChessModel
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.chess_utils import get_total_moves
from src.utils.train_utils import initialize_optimizer, initialize_scheduler, initialize_random_seeds, validate_epoch, train_epoch

class SupervisedWorker:
    def __init__(self, epochs:int, batch_size:int, learning_rate:float, weight_decay:float, save_checkpoints:bool, checkpoint_interval:int, dataset_path:str, train_indices_path:str, val_indices_path:str, model_path:Optional[str]=None, checkpoint_type:str='epoch', optimizer_type:str='adamw', scheduler_type:str='cosineannealingwarmrestarts', num_workers:int=4, random_seed:int=42, progress_callback=None, status_callback=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_type = checkpoint_type
        self.dataset_path = dataset_path
        self.train_indices_path = train_indices_path
        self.val_indices_path = val_indices_path
        self.model_path = model_path
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        initialize_random_seeds(self.random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerChessModel(get_total_moves()).to(self.device)
        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay)
        self.scheduler = None
        self.scaler = GradScaler()
        self.checkpoint_dir = os.path.join('models', 'checkpoints', 'supervised')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir, checkpoint_type=self.checkpoint_type, checkpoint_interval=self.checkpoint_interval)
        self.total_batches_processed = 0

    def run(self) -> Dict:
        try:
            required_files = [(self.dataset_path, "dataset file"), (self.train_indices_path, "training indices"), (self.val_indices_path, "validation indices")]
            for file_path, desc in required_files:
                if not os.path.exists(file_path):
                    if self.status_callback:
                        self.status_callback(f"Missing {desc} at {file_path}.")
                    return {}
            train_indices = np.load(self.train_indices_path)
            val_indices = np.load(self.val_indices_path)
            train_dataset = H5Dataset(self.dataset_path, train_indices)
            val_dataset = H5Dataset(self.dataset_path, val_indices)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
            start_epoch = 1
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = self.checkpoint_manager.load(self.model_path, self.device, self.model, self.optimizer, self.scheduler)
                if checkpoint:
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    training_stats = checkpoint.get('training_stats', {})
                    self.total_batches_processed = training_stats.get('total_batches_processed', 0)
            if self.scheduler is None and self.scheduler_type.lower() != 'none':
                total_steps = self.epochs * len(train_loader)
                self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, total_steps=total_steps)
            metrics = {}
            for epoch in range(start_epoch, self.epochs + 1):
                train_epoch(model=self.model, data_loader=train_loader, device=self.device, scaler=self.scaler, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, accumulation_steps=max(256 // self.batch_size, 1), batch_size=self.batch_size, smooth_policy_targets=True, compute_accuracy_flag=True, total_batches_processed=self.total_batches_processed, progress_callback=self.progress_callback, status_callback=self.status_callback)
                self.model.eval()
                validate_epoch(model=self.model, val_loader=val_loader, device=self.device, epoch=epoch, smooth_policy_targets=True, progress_callback=self.progress_callback, status_callback=self.status_callback)
                self.model.train()
                if self.save_checkpoints:
                    self.checkpoint_manager.save(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, training_stats={"total_batches_processed":self.total_batches_processed})
            final_model_path = os.path.join("models", "saved_models", "supervised_model.pth")
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            self.checkpoint_manager.save_final_model(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, training_stats={"total_batches_processed":self.total_batches_processed}, final_path=final_model_path)
            metrics = {
                "final_model_path": final_model_path,
                "total_epochs": self.epochs,
                "total_batches_processed": self.total_batches_processed
            }
            return metrics
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Training error: {e}")
            raise