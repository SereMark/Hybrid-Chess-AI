import os, torch, time, numpy as np
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from src.utils.datasets import H5Dataset
from src.models.transformer import TransformerChessModel
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.chess_utils import get_total_moves
from src.utils.train_utils import initialize_optimizer, initialize_scheduler, initialize_random_seeds, validate_epoch, train_epoch

class SupervisedWorker:
    def __init__(self, epochs, batch_size, learning_rate, weight_decay, checkpoint_interval, dataset_path, train_indices_path, val_indices_path, model_path, optimizer_type, scheduler_type, accumulation_steps, num_workers, random_seed, policy_weight, value_weight, grad_clip, momentum, progress_callback, status_callback):
        self.epochs, self.batch_size, self.learning_rate, self.weight_decay, self.checkpoint_interval = epochs, batch_size, learning_rate, weight_decay, checkpoint_interval
        self.dataset_path, self.train_indices_path, self.val_indices_path = dataset_path, train_indices_path, val_indices_path
        self.model_path, self.optimizer_type, self.scheduler_type, self.grad_clip, self.momentum = model_path, optimizer_type, scheduler_type, grad_clip, momentum
        self.num_workers, self.random_seed = num_workers, random_seed
        self.progress_callback, self.status_callback = progress_callback, status_callback
        self.policy_weight, self.value_weight = policy_weight, value_weight
        initialize_random_seeds(self.random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerChessModel(get_total_moves()).to(self.device)
        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay, momentum)
        self.scheduler = None
        self.scaler = GradScaler()
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = os.path.join('models', 'checkpoints', 'supervised')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir, checkpoint_type='epoch', checkpoint_interval=self.checkpoint_interval)
    
    def run(self):
        train_loader = DataLoader(H5Dataset(self.dataset_path, np.load(self.train_indices_path)), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(H5Dataset(self.dataset_path, np.load(self.val_indices_path)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        start_epoch, best_epoch = 1, 0
        best_metric = float('inf')
        training_start_time = time.time()
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = self.checkpoint_manager.load(self.model_path, self.device, self.model, self.optimizer, self.scheduler)
            if checkpoint:
                start_epoch = checkpoint.get('epoch', 0) + 1
        self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, total_steps=self.epochs * len(train_loader))
        for epoch in range(start_epoch, self.epochs + 1):
            train_epoch(self.model, train_loader, self.device, self.scaler, self.optimizer, self.scheduler, epoch, self.epochs, self.accumulation_steps, self.batch_size, True, True, self.policy_weight, self.value_weight, self.grad_clip, self.progress_callback, self.status_callback)
            epoch_metrics = validate_epoch(self.model, val_loader, self.device, epoch, self.epochs, True, self.progress_callback, self.status_callback)
            val_loss = epoch_metrics["policy_loss"] + epoch_metrics["value_loss"]
            if val_loss < best_metric:
                best_metric = val_loss
                best_epoch = epoch
            if self.checkpoint_interval and self.checkpoint_interval > 0:
                self.checkpoint_manager.save(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch)
        self.checkpoint_manager.save_final_model(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=self.epochs, final_path=os.path.join("models", "saved_models", "supervised_model.pth"))
        return {'metric': best_metric, 'val_loss': val_loss, 'val_accuracy': epoch_metrics["accuracy"], 'best_epoch': best_epoch, 'training_time': time.time() - training_start_time}