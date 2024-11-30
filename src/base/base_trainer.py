import os, time, threading, torch
from enum import Enum
from src.utils.common_utils import log_message, initialize_random_seeds, estimate_batch_size


class CheckpointType(Enum):
    EPOCH = 'epoch'
    BATCH = 'batch'
    TIME = 'time'
    ITERATION = 'iteration'


class TrainerBase:
    def __init__(
        self,
        save_checkpoints=True,
        checkpoint_interval=1,
        checkpoint_type='epoch',
        checkpoint_interval_minutes=60,
        checkpoint_batch_interval=1000,
        checkpoint_dir='models/checkpoints',
        log_fn=None,
        progress_fn=None,
        time_left_fn=None,
        stop_event=None,
        pause_event=None,
        random_seed=42,
        automatic_batch_size=False,
        batch_size=256,
        model_class=None,
        optimizer_type='adamw',
        learning_rate=0.001,
        weight_decay=1e-4,
        scheduler_type='cosineannealingwarmrestarts',
        num_workers=4,
        device=None
    ):
        self._lock = threading.Lock()
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        try:
            self.checkpoint_type = CheckpointType(checkpoint_type)
        except ValueError:
            log_message(
                f"Unsupported checkpoint type: {checkpoint_type}. Using 'epoch' by default.",
                log_fn
            )
            self.checkpoint_type = CheckpointType.EPOCH
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_batch_interval = checkpoint_batch_interval
        self.last_checkpoint_time = time.time()
        self.total_batches_processed = 0
        self.checkpoint_dir = checkpoint_dir
        self.log_fn = log_fn
        self.progress_fn = progress_fn
        self.time_left_fn = time_left_fn
        self.stop_event = stop_event or threading.Event()
        self.pause_event = pause_event or threading.Event()
        self.pause_event.set()
        self.random_seed = random_seed
        self.automatic_batch_size = automatic_batch_size
        self.batch_size = batch_size
        self.model_class = model_class
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.num_workers = num_workers
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        initialize_random_seeds(self.random_seed)

    def initialize_model(self, *args, **kwargs):
        if self.model_class is None:
            raise ValueError("model_class must be provided.")
        self.model = self.model_class(*args, **kwargs).to(self.device)
        if self.automatic_batch_size:
            self.batch_size = estimate_batch_size(self.model, self.device)
            log_message(f"Automatic batch size estimation: Using batch size {self.batch_size}", self.log_fn)
        else:
            log_message(f"Using manual batch size: {self.batch_size}", self.log_fn)

    def initialize_optimizer(self):
        optimizer_type_lower = self.optimizer_type.lower()
        if optimizer_type_lower == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif optimizer_type_lower == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        elif optimizer_type_lower == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            log_message(
                f"Unsupported optimizer type: {self.optimizer_type}. Using AdamW by default.",
                self.log_fn,
            )
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

    def initialize_scheduler(self, total_steps):
        scheduler_type_lower = self.scheduler_type.lower()
        if scheduler_type_lower == 'cosineannealingwarmrestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        elif scheduler_type_lower == 'steplr':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_type_lower == 'onecyclelr':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.learning_rate, total_steps=total_steps
            )
        elif scheduler_type_lower == 'cosineannealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10
            )
        elif scheduler_type_lower == 'none':
            self.scheduler = None
        else:
            log_message(
                f"Unsupported scheduler type: {self.scheduler_type}. Using CosineAnnealingWarmRestarts by default.",
                self.log_fn,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )

    def should_save_checkpoint(self, epoch=None, batch_idx=None, iteration=None):
        with self._lock:
            if not self.save_checkpoints:
                return False
            if self.checkpoint_type == CheckpointType.EPOCH and epoch is not None:
                return epoch % self.checkpoint_interval == 0 and batch_idx is None and iteration is None
            elif self.checkpoint_type == CheckpointType.BATCH and batch_idx is not None:
                return self.total_batches_processed % self.checkpoint_batch_interval == 0
            elif self.checkpoint_type == CheckpointType.TIME:
                current_time = time.time()
                elapsed_minutes = (current_time - self.last_checkpoint_time) / 60
                if elapsed_minutes >= self.checkpoint_interval_minutes:
                    self.last_checkpoint_time = current_time
                    return True
                return False
            elif self.checkpoint_type == CheckpointType.ITERATION and iteration is not None:
                return iteration % self.checkpoint_interval == 0 and batch_idx is None
            return False

    def save_checkpoint(self, epoch, batch_idx=None, iteration=None, additional_info=None):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f'checkpoint_{timestamp}.pth'
        temp_path = os.path.join(self.checkpoint_dir, f'.temp_{checkpoint_name}')
        final_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        try:
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') and self.scheduler else None,
                'epoch': epoch,
                'batch_idx': batch_idx,
                'iteration': iteration,
                'total_batches_processed': self.total_batches_processed,
            }
            if additional_info is not None:
                checkpoint_data.update(additional_info)
            torch.save(checkpoint_data, temp_path)
            os.replace(temp_path, final_path)
            log_message(f"Checkpoint saved: {checkpoint_name}", self.log_fn)
        except Exception as e:
            log_message(f"Error saving checkpoint: {str(e)}", self.log_fn)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as remove_e:
                    log_message(f"Failed to remove temp checkpoint: {str(remove_e)}", self.log_fn)
            raise

    def load_checkpoint(self, checkpoint_path, map_location=None):
        if not os.path.exists(checkpoint_path):
            log_message(f"Checkpoint not found at {checkpoint_path}.", self.log_fn)
            return
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location or self.device)  # TODO: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling.
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict') and hasattr(self, 'scheduler') and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.total_batches_processed = checkpoint.get('total_batches_processed', 0)
            return checkpoint
        except Exception as e:
            log_message(f"Error loading checkpoint: {str(e)}", self.log_fn)
            raise