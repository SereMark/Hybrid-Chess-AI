import os, torch, numpy as np
from typing import Optional
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from src.utils.datasets import H5Dataset
from src.models.transformer import TransformerChessModel
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.chess_utils import get_total_moves
from src.utils.train_utils import initialize_optimizer, initialize_scheduler, initialize_random_seeds, validate_epoch, train_epoch

class SupervisedWorker:
    def __init__(self, epochs:int, batch_size:int, learning_rate:float, weight_decay:float, checkpoint_interval:int, dataset_path:str, train_indices_path:str, val_indices_path:str, model_path:Optional[str]=None, optimizer_type:str='adamw', scheduler_type:str='cosineannealingwarmrestarts', accumulation_steps:int=3, num_workers:int=4, random_seed:int=42, policy_weight:float=1.0, value_weight:float=2.0, progress_callback=None, status_callback=None):
        self.epochs, self.batch_size, self.learning_rate, self.weight_decay, self.checkpoint_interval = epochs, batch_size, learning_rate, weight_decay, checkpoint_interval
        self.dataset_path, self.train_indices_path, self.val_indices_path = dataset_path, train_indices_path, val_indices_path
        self.model_path, self.optimizer_type, self.scheduler_type = model_path, optimizer_type, scheduler_type
        self.num_workers, self.random_seed = num_workers, random_seed
        self.progress_callback, self.status_callback = progress_callback, status_callback
        self.policy_weight, self.value_weight = policy_weight, value_weight
        initialize_random_seeds(self.random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerChessModel(get_total_moves()).to(self.device)
        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay)
        self.scheduler = None
        self.scaler = GradScaler()
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = os.path.join('models', 'checkpoints', 'supervised')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir, checkpoint_type='epoch', checkpoint_interval=self.checkpoint_interval)
    
    def run(self):
        train_loader = DataLoader(H5Dataset(self.dataset_path, np.load(self.train_indices_path)), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(H5Dataset(self.dataset_path, np.load(self.val_indices_path)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        start_epoch = 1
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = self.checkpoint_manager.load(self.model_path, self.device, self.model, self.optimizer, self.scheduler)
            if checkpoint:
                start_epoch = checkpoint.get('epoch', 0) + 1
        if not self.scheduler and self.scheduler_type.lower() != 'none':
            self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, total_steps=self.epochs * len(train_loader))
        for epoch in range(start_epoch, self.epochs + 1):
            train_epoch(model=self.model, data_loader=train_loader, device=self.device, scaler=self.scaler, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, accumulation_steps=self.accumulation_steps, batch_size=self.batch_size, smooth_policy_targets=True, compute_accuracy_flag=True, progress_callback=self.progress_callback, status_callback=self.status_callback, policy_weight=self.policy_weight, value_weight=self.value_weight)
            validate_epoch(model=self.model, val_loader=val_loader, device=self.device, epoch=epoch, smooth_policy_targets=True, progress_callback=self.progress_callback, status_callback=self.status_callback, policy_weight=self.policy_weight, value_weight=self.value_weight)
            if self.checkpoint_interval and self.checkpoint_interval > 0:
                self.checkpoint_manager.save(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch)
        self.checkpoint_manager.save_final_model(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=self.epochs, final_path=os.path.join("models", "saved_models", "supervised_model.pth"))
        return True