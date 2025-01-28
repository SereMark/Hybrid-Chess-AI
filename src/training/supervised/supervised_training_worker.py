import os, torch, time, numpy as np
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from src.utils.datasets import H5Dataset
from src.models.transformer import TransformerChessModel
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.chess_utils import get_total_moves
from src.utils.train_utils import initialize_optimizer, initialize_scheduler, initialize_random_seeds, validate_epoch, train_epoch

class SupervisedWorker:
    def __init__(self, epochs, batch_size, learning_rate, weight_decay, checkpoint_interval, dataset_path, train_indices_path, val_indices_path, model_path, optimizer_type, scheduler_type, accumulation_steps, num_workers, random_seed, policy_weight, value_weight, grad_clip, momentum, wandb_flag, progress_callback, status_callback):
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        self.weight_decay, self.checkpoint_interval = weight_decay, checkpoint_interval
        self.dataset_path, self.train_indices_path, self.val_indices_path = dataset_path, train_indices_path, val_indices_path
        self.model_path, self.optimizer_type, self.scheduler_type = model_path, optimizer_type, scheduler_type
        self.grad_clip, self.momentum = grad_clip, momentum
        self.num_workers, self.random_seed = num_workers, random_seed
        self.wandb, self.progress_callback, self.status_callback = wandb_flag, progress_callback, status_callback
        self.policy_weight, self.value_weight = policy_weight, value_weight
        initialize_random_seeds(self.random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerChessModel(get_total_moves()).to(self.device)
        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay, self.momentum)
        self.scheduler = None
        self.scaler = GradScaler()
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = os.path.join('models', 'checkpoints', 'supervised')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir, 'epoch', self.checkpoint_interval)

    def run(self):
        if self.wandb:
            import wandb
            wandb.init(entity="chess_ai", project="chess_ai_app", config=self.__dict__, reinit=True)
            wandb.watch(self.model, log="all", log_freq=100)
        train_loader = DataLoader(H5Dataset(self.dataset_path, np.load(self.train_indices_path)), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(H5Dataset(self.dataset_path, np.load(self.val_indices_path)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        start_epoch, best_epoch, best_metric, training_start = 1, 0, float('inf'), time.time()
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = self.checkpoint_manager.load(self.model_path, self.device, self.model, self.optimizer, self.scheduler)
            start_epoch = checkpoint.get('epoch', 0) + 1 if checkpoint else 1
        self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, self.epochs * len(train_loader))
        history = []
        for epoch in range(start_epoch, self.epochs + 1):
            train_metrics = train_epoch(self.model, train_loader, self.device, self.scaler, self.optimizer, self.scheduler, epoch, self.epochs, self.accumulation_steps, self.batch_size, True, True,
                                        self.policy_weight, self.value_weight, self.grad_clip, self.progress_callback, self.status_callback, self.wandb)
            epoch_metrics = validate_epoch(self.model, val_loader, self.device, epoch, self.epochs, True, self.progress_callback, self.status_callback, self.wandb)
            val_loss = epoch_metrics["policy_loss"] + epoch_metrics["value_loss"]
            if self.wandb:
                history.append([epoch, train_metrics["accuracy"], epoch_metrics["accuracy"]])
                table = wandb.Table(data=history, columns=["epoch", "train_accuracy", "val_accuracy"])
                wandb.log({
                    "epoch": epoch, "val_loss": val_loss, "policy_loss": epoch_metrics["policy_loss"],
                    "value_loss": epoch_metrics["value_loss"], "accuracy": epoch_metrics["accuracy"],
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "train_policy_loss": train_metrics["policy_loss"],
                    "train_value_loss": train_metrics["value_loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "accuracy_vs_epoch": wandb.plot.scatter(table, "epoch", "train_accuracy", title="Train Accuracy vs Epoch"),
                    "accuracy_vs_epoch_val": wandb.plot.scatter(table, "epoch", "val_accuracy", title="Val Accuracy vs Epoch")
                })
                for idx, layer in enumerate(self.model.transformer_encoder.layers[:3]):
                    wandb.log({f"hist/weight_layer_{idx}": wandb.Histogram(layer.self_attn.in_proj_weight.detach().cpu().numpy()), f"hist/grad_layer_{idx}": wandb.Histogram(layer.self_attn.in_proj_weight.grad.detach().cpu().numpy() if layer.self_attn.in_proj_weight.grad is not None else np.zeros_like(layer.self_attn.in_proj_weight.detach().cpu().numpy()))})
                attn = self.model.transformer_encoder.layers[0].self_attn.in_proj_weight
                wandb.log({"attention_mean": attn.mean().item(), "attention_std": attn.std().item()})
            if val_loss < best_metric:
                best_metric, best_epoch = val_loss, epoch
                if self.wandb:
                    wandb.run.summary.update({"best_val_loss": best_metric, "best_epoch": best_epoch})
            if self.checkpoint_interval > 0:
                self.checkpoint_manager.save(self.model, self.optimizer, self.scheduler, epoch, None)
        self.checkpoint_manager.save_final_model(self.model, self.optimizer, self.scheduler, self.epochs, None, os.path.join("models", "saved_models", "supervised_model.pth"))
        if self.wandb:
            wandb.run.summary.update({"metric": best_metric, "val_loss": val_loss, "val_accuracy": epoch_metrics["accuracy"], "best_epoch": best_epoch, "training_time": time.time() - training_start})
            wandb.finish()
        return {'metric': best_metric, 'val_loss': val_loss, 'val_accuracy': epoch_metrics["accuracy"], 'best_epoch': best_epoch, 'training_time': time.time() - training_start}