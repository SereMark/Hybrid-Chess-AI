from PyQt5.QtCore import QObject, pyqtSignal
import threading
import traceback

from src.models.train import ModelTrainer


class SupervisedTrainingWorker(QObject):
    log_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    batch_loss_update = pyqtSignal(int, dict)
    batch_accuracy_update = pyqtSignal(int, float)
    learning_rate_update = pyqtSignal(int, float)
    epoch_loss_update = pyqtSignal(int, dict)
    epoch_accuracy_update = pyqtSignal(int, float, float)
    initial_batches_processed = pyqtSignal(int)
    val_loss_update = pyqtSignal(int, dict)
    training_finished = pyqtSignal()
    time_left_update = pyqtSignal(str)

    def __init__(self, epochs, batch_size, learning_rate, weight_decay, save_checkpoints,
                 checkpoint_interval, dataset_path, train_indices_path, val_indices_path, checkpoint_path=None,
                 automatic_batch_size=False, checkpoint_type='epoch',
                 checkpoint_interval_minutes=60,
                 checkpoint_batch_interval=1000):
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
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()

    def run(self):
        try:
            self.log_update.emit("Initializing training...")
            trainer = ModelTrainer(
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                log_fn=self.log_update.emit,
                progress_fn=self.progress_update.emit,
                loss_fn=self.epoch_loss_update.emit,
                val_loss_fn=self.val_loss_update.emit,
                accuracy_fn=self.epoch_accuracy_update.emit,
                stop_event=self._stop_event,
                pause_event=self._pause_event,
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
                lr_fn=self.learning_rate_update.emit,
                initial_batches_processed_callback=self.initial_batches_processed.emit
            )
            self.log_update.emit("Starting training...")
            trainer.train_model()
            if not self._stop_event.is_set():
                self.log_update.emit("Training completed successfully.")
            else:
                self.log_update.emit("Training stopped by user request.")
        except Exception as e:
            error_msg = f"Error during training: {str(e)}\n{traceback.format_exc()}"
            self.log_update.emit(error_msg)
        finally:
            self.training_finished.emit()

    def pause(self):
        self._pause_event.clear()
        self.log_update.emit("Training paused by user.")

    def resume(self):
        self._pause_event.set()
        self.log_update.emit("Training resumed by user.")

    def stop(self):
        self._stop_event.set()
        self.log_update.emit("Training stopped by user.")