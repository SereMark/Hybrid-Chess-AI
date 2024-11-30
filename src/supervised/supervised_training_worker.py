from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.supervised.supervised_training import SupervisedTrainer


class SupervisedWorker(BaseWorker):
    batch_loss_update = pyqtSignal(int, dict)
    batch_accuracy_update = pyqtSignal(int, float)
    epoch_loss_update = pyqtSignal(int, dict)
    epoch_accuracy_update = pyqtSignal(int, float, float)
    val_loss_update = pyqtSignal(int, dict)
    initial_batches_processed = pyqtSignal(int)

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        save_checkpoints: bool,
        checkpoint_interval: int,
        dataset_path: str,
        train_indices_path: str,
        val_indices_path: str,
        checkpoint_path: str = None,
        automatic_batch_size: bool = False,
        checkpoint_type: str = 'epoch',
        checkpoint_interval_minutes: int = 60,
        checkpoint_batch_interval: int = 1000,
        optimizer_type: str = 'adamw',
        scheduler_type: str = 'cosineannealingwarmrestarts',
        output_model_path: str = 'models/saved_models/pre_trained_model.pth',
        num_workers: int = 4
    ):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.dataset_path = dataset_path
        self.train_indices_path = train_indices_path
        self.val_indices_path = val_indices_path
        self.checkpoint_path = checkpoint_path
        self.automatic_batch_size = automatic_batch_size
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_batch_interval = checkpoint_batch_interval
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.output_model_path = output_model_path
        self.num_workers = num_workers

    def run_task(self):
        self.log_update.emit("Initializing supervised training...")
        trainer = SupervisedTrainer(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            log_fn=self.log_update.emit,
            progress_fn=self.progress_update.emit,
            loss_fn=self.epoch_loss_update.emit,
            val_loss_fn=self.val_loss_update.emit,
            accuracy_fn=self.epoch_accuracy_update.emit,
            stop_event=self._is_stopped,
            pause_event=self._is_paused,
            time_left_fn=self.time_left_update.emit,
            save_checkpoints=self.save_checkpoints,
            checkpoint_interval=self.checkpoint_interval,
            checkpoint_type=self.checkpoint_type,
            checkpoint_interval_minutes=self.checkpoint_interval_minutes,
            checkpoint_batch_interval=self.checkpoint_batch_interval,
            dataset_path=self.dataset_path,
            train_indices_path=self.train_indices_path,
            val_indices_path=self.val_indices_path,
            checkpoint_path=self.checkpoint_path,
            automatic_batch_size=self.automatic_batch_size,
            batch_loss_fn=self.batch_loss_update.emit,
            batch_accuracy_fn=self.batch_accuracy_update.emit,
            initial_batches_processed_callback=self.initial_batches_processed.emit,
            optimizer_type=self.optimizer_type,
            scheduler_type=self.scheduler_type,
            output_model_path=self.output_model_path,
            num_workers=self.num_workers
        )
        self.log_update.emit("Starting supervised training...")
        trainer.train_model()
        if self._is_stopped.is_set():
            self.log_update.emit("Supervised training stopped by user request.")