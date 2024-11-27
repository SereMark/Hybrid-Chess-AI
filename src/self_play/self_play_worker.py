from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.self_play.self_play_trainer import SelfPlayTrainer


class SelfPlayWorker(BaseWorker):
    stats_update = pyqtSignal(dict)

    def __init__(
        self,
        model_path: str,
        num_iterations: int,
        num_games_per_iteration: int,
        simulations: int,
        c_puct: float,
        temperature: float,
        num_epochs: int,
        batch_size: int,
        automatic_batch_size: bool,
        num_threads: int,
        save_checkpoints: bool,
        checkpoint_interval: int,
        checkpoint_type: str,
        checkpoint_interval_minutes: int,
        checkpoint_batch_interval: int,
        checkpoint_path: str = None,
        random_seed: int = 42,
        optimizer_type: str = 'adamw',
        learning_rate: float = 0.0005,
        weight_decay: float = 2e-4,
        scheduler_type: str = 'cosineannealingwarmrestarts',
    ):
        super().__init__()
        self.model_path = model_path
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.automatic_batch_size = automatic_batch_size
        self.num_threads = num_threads
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_batch_interval = checkpoint_batch_interval
        self.checkpoint_path = checkpoint_path
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type

    def run_task(self):
        self.log_update.emit("Starting self-play training...")
        trainer = SelfPlayTrainer(
            model_path=self.model_path,
            num_iterations=self.num_iterations,
            num_games_per_iteration=self.num_games_per_iteration,
            simulations=self.simulations,
            c_puct=self.c_puct,
            temperature=self.temperature,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            automatic_batch_size=self.automatic_batch_size,
            num_threads=self.num_threads,
            checkpoint_path=self.checkpoint_path,
            random_seed=self.random_seed,
            save_checkpoints=self.save_checkpoints,
            checkpoint_interval=self.checkpoint_interval,
            checkpoint_type=self.checkpoint_type,
            checkpoint_interval_minutes=self.checkpoint_interval_minutes,
            checkpoint_batch_interval=self.checkpoint_batch_interval,
            optimizer_type=self.optimizer_type,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            scheduler_type=self.scheduler_type,
            log_fn=self.log_update.emit,
            progress_fn=self.progress_update.emit,
            stats_fn=self.stats_update.emit,
            time_left_fn=self.time_left_update.emit,
            stop_event=self._is_stopped,
            pause_event=self._is_paused
        )
        trainer.train()
        if self._is_stopped.is_set():
            self.log_update.emit("Self-play training stopped by user request.")