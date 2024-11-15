from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.self_play.self_play_trainer import SelfPlayTrainer
import traceback

class SelfPlayWorker(BaseWorker):
    stats_update = pyqtSignal(dict)
    training_finished = pyqtSignal()

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        num_iterations: int,
        num_games_per_iteration: int,
        simulations: int,
        c_puct: float,
        temperature: float,
        num_epochs: int,
        batch_size: int,
        automatic_batch_size: bool,
        num_threads: int,
        checkpoint_path: str = None,
        random_seed: int = 42
    ):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.automatic_batch_size = automatic_batch_size
        self.num_threads = num_threads
        self.checkpoint_path = checkpoint_path
        self.random_seed = random_seed

    def run(self):
        try:
            self.log_update.emit("Starting self-play training...")
            trainer = SelfPlayTrainer(
                model_path=self.model_path,
                output_dir=self.output_dir,
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
                log_fn=self.log_update.emit,
                progress_fn=self.progress_update.emit,
                stats_fn=self.stats_update.emit,
                time_left_fn=self.time_left_update.emit,
                stop_event=self._is_stopped,
                pause_event=self._is_paused
            )
            trainer.train()
            if not self._is_stopped.is_set():
                self.log_update.emit("Self-play training completed successfully.")
                self.training_finished.emit()
            else:
                self.log_update.emit("Self-play training stopped by user request.")
        except Exception as e:
            error_msg = f"Error during self-play training: {str(e)}\n{traceback.format_exc()}"
            self.log_update.emit(error_msg)
        finally:
            self.finished.emit()