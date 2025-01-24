import os
import time
import threading
from typing import List, Tuple, Dict, Optional
import numpy as np
import chess
import chess.pgn
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool, cpu_count, Manager
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.models.transformer import TransformerChessModel
from src.utils.chess_utils import get_total_moves
from src.utils.common_utils import format_time_left, update_progress_time_left
from src.utils.train_utils import initialize_optimizer, initialize_random_seeds, initialize_scheduler, train_epoch
from src.utils.checkpoint_manager import CheckpointManager
from src.training.reinforcement.play_and_collect_worker import PlayAndCollectWorker

class ReinforcementWorker(BaseWorker):
    stats_update = pyqtSignal(dict)

    def __init__(self, model_path: Optional[str], num_iterations: int, num_games_per_iteration: int, simulations: int, c_puct: float, temperature: float, num_epochs: int, batch_size: int, 
                 num_threads: int, save_checkpoints: bool, checkpoint_interval: int, checkpoint_type: str, checkpoint_interval_minutes: int, checkpoint_batch_interval: int,
                   random_seed: int = 42, optimizer_type: str = "adamw", learning_rate: float = 0.0001, weight_decay: float = 1e-4, scheduler_type: str = "cosineannealingwarmrestarts"):
        super().__init__()

        # Checkpoint/Directory Settings
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_type = checkpoint_type.lower()
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_batch_interval = checkpoint_batch_interval
        self.checkpoint_dir = os.path.join("models", "checkpoints", "reinforcement")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training Settings
        self.model_path = model_path
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.random_seed = random_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optimizer/Scheduler Settings
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type

        # Mixed Precision Training
        self.scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

        # Initialization
        initialize_random_seeds(self.random_seed)
        self.lock = threading.Lock()

        # Tracking
        self.results: List[float] = []
        self.game_lengths: List[int] = []
        self.total_games_played = 0
        self.total_batches_processed = 0
        self.current_epoch = 1
        self.batch_idx: Optional[int] = None
        self.start_iteration = 0

        # Build Model
        self.model = TransformerChessModel(get_total_moves()).to(self.device)
        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay, logger=self.logger)
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

        # Checkpoint Manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir, checkpoint_type=self.checkpoint_type, checkpoint_interval=self.checkpoint_interval, logger=self.logger)

        # For Progress Tracking
        self_play_batches = max(self.num_games_per_iteration // max(self.batch_size, 1), 1)
        self.train_steps_per_iter = self.num_epochs * self_play_batches
        self.total_steps = (self.num_iterations * self.num_games_per_iteration + self.num_iterations * self.train_steps_per_iter)

        # For Subprocess Model State
        self.model_state_dict: Optional[Dict[str, torch.Tensor]] = None

        # Directory for self-play PGN games
        self.self_play_dir = os.path.join("data", "games", "self-play")
        os.makedirs(self.self_play_dir, exist_ok=True)

    def run_task(self):
        self.logger.info("Initializing reinforcement worker with model and optimizer.")

        # Load model checkpoint if available
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = self.checkpoint_manager.load(self.model_path, self.device, self.model, self.optimizer, self.scheduler)
            if checkpoint:
                self.start_iteration = checkpoint.get("iteration", 0)
                self.current_epoch = checkpoint.get("epoch", 1)
                self.batch_idx = checkpoint.get("batch_idx", None)

                training_stats = checkpoint.get("training_stats", {})
                self.results = training_stats.get("results", [])
                self.game_lengths = training_stats.get("game_lengths", [])
                self.total_games_played = training_stats.get("total_games_played", 0)
                self.total_batches_processed = checkpoint.get("batch_idx", 0)

                self.logger.info(f"Resuming from iteration {self.start_iteration}, epoch {self.current_epoch}, batch {self.batch_idx}.")
            else:
                self.logger.info("No valid checkpoint found. Starting from scratch.")
        else:
            self.logger.info("No checkpoint path or file not found. Starting fresh.")

        self.start_time = time.time()

        # Main training loop
        for iteration in range(self.start_iteration, self.num_iterations):
            if self._is_stopped.is_set():
                break

            iteration_start_time = time.time()
            self.logger.info(f"Starting iteration {iteration + 1}/{self.num_iterations}.")
            self.current_epoch = 1

            # Self-play data collection
            self.model.eval()
            self_play_data, pgn_games = self._generate_self_play_data()
            self.model.train()

            # Save PGN games
            timestamp = int(time.time())
            for idx, game in enumerate(pgn_games, start=1):
                filename = os.path.join(self.self_play_dir, f"iteration_{iteration + 1}_game_{timestamp}_{idx}.pgn")
                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(str(game))
                    self.logger.info(f"Saved self-play game to {filename}")
                except Exception as e:
                    self.logger.error(f"Failed to save PGN for game {idx} in iteration {iteration + 1}: {e}")

            # Train on the collected self-play data
            if len(self_play_data) == 3:
                inputs, policy_targets, value_targets = self_play_data

                # If no data was generated, skip
                if inputs.numel() == 0:
                    self.logger.warning("No self-play data generated this iteration. Skipping training.")
                else:
                    # Build DataLoader from these tensors
                    dataset = TensorDataset(inputs.cpu(), policy_targets.cpu(), value_targets.cpu())
                    data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=(self.device.type == "cuda"), num_workers=min(os.cpu_count(), 8))

                    # Initialize scheduler if needed
                    if self.scheduler is None:
                        total_training_steps = self.num_epochs * len(data_loader)
                        self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, total_steps=total_training_steps, logger=self.logger)

                    # Train for multiple epochs on this data
                    for epoch in range(self.current_epoch, self.num_epochs + 1):
                        if self._is_stopped.is_set():
                            break

                        self.logger.info(f"Epoch {epoch}/{self.num_epochs}, iteration {iteration + 1}.")
                        self.current_epoch = epoch

                        # Train for one epoch
                        train_metrics = train_epoch(model=self.model, data_loader=data_loader, device=self.device, scaler=self.scaler, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, total_epochs=self.num_epochs,
                            skip_batches=self.batch_idx if (epoch == self.current_epoch and self.batch_idx is not None) else 0, accumulation_steps=max(256 // max(self.batch_size, 1), 1), batch_size=self.batch_size,
                            smooth_policy_targets=False, compute_accuracy_flag=False, total_batches_processed=self.total_batches_processed, batch_loss_update_signal=None, batch_accuracy_update_signal=None,
                            progress_update_signal=self.progress_update, time_left_update_signal=self.time_left_update, checkpoint_manager=self.checkpoint_manager, checkpoint_type=self.checkpoint_type,
                            logger=self.logger, is_stopped_event=self._is_stopped, is_paused_event=self._is_paused, start_time=self.start_time, total_steps=self.total_steps)
                        
                        # Update counters
                        self.total_batches_processed = train_metrics["total_batches_processed"]
                        self.batch_idx = None

                        # End of epoch logging
                        self.logger.info(f"Finished epoch {epoch}/{self.num_epochs} in iteration {iteration + 1}.")

            # Possibly checkpoint at iteration level
            if self.save_checkpoints:
                self.checkpoint_manager.save(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=self.current_epoch, batch_idx=self.total_batches_processed, 
                                               iteration=iteration + 1, training_stats={"total_games_played": self.total_games_played, "results": self.results,"game_lengths": self.game_lengths})

            # Print iteration timing
            iter_duration = time.time() - iteration_start_time
            self.logger.info(f"Iteration {iteration + 1} finished in {format_time_left(iter_duration)}.")

            # Emit iteration stats
            if self.stats_update is not None and hasattr(self.stats_update, "emit"):
                self.stats_update.emit({"iteration": iteration + 1, "total_games_played": self.total_games_played})

        # Save final model
        final_dir = os.path.join("models", "saved_models")
        final_path = os.path.join(final_dir, "final_model.pth")
        os.makedirs(final_dir, exist_ok=True)

        self.checkpoint_manager.save_final_model(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                                                  training_stats={"total_games_played": self.total_games_played, "results": self.results, "game_lengths": self.game_lengths}, final_path=final_path)

        self.task_finished.emit()
        self.finished.emit()

    def _generate_self_play_data(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[chess.pgn.Game]]:
        num_processes = max(min(self.num_threads, cpu_count()), 1)

        games_per_process = self.num_games_per_iteration // num_processes
        remainder = self.num_games_per_iteration % num_processes

        manager = Manager()
        stop_event = manager.Event()
        pause_event = manager.Event()

        if self._is_stopped.is_set():
            stop_event.set()
        if not self._is_paused.is_set():
            pause_event.clear()
        else:
            pause_event.set()

        stats_queue = manager.Queue()
        seeds = [self.random_seed + i + int(time.time()) for i in range(num_processes)]

        # Prepare arguments for each subprocess
        self.model_state_dict = { k: v.cpu() for k, v in self.model.state_dict().items() }
        tasks = []
        for i in range(num_processes):
            gpp = games_per_process + (1 if i < remainder else 0)
            tasks.append((self.model_state_dict, self.device.type, self.simulations, self.c_puct, self.temperature, gpp, stop_event, pause_event, seeds[i], stats_queue))

        with Pool(processes=num_processes) as pool:
            results = pool.map(PlayAndCollectWorker.run_process, tasks)

        # Collect stats from queue
        while not stats_queue.empty():
            stat = stats_queue.get()
            if "error" in stat:
                self.logger.error(stat["error"])
                continue

            if self.stats_update and hasattr(self.stats_update, "emit"):
                self.stats_update.emit(stat)

            with self.lock:
                self.total_batches_processed += stat.get("total_games", 0)

            update_progress_time_left(progress_signal=self.progress_update, time_left_signal=self.time_left_update, start_time=self.start_time, current_step=self.total_batches_processed, total_steps=self.total_steps)

        # Combine results from all processes
        all_inputs, all_policy_targets, all_value_targets = [], [], []
        pgn_games_list = []

        for (inp, pol, val, res, g_len, pgns) in results:
            all_inputs.extend(inp)
            all_policy_targets.extend(pol)
            all_value_targets.extend(val)
            self.results.extend(res)
            self.game_lengths.extend(g_len)
            pgn_games_list.extend(pgns)

        # If no data was generated, return empty
        if not all_inputs:
            self.logger.warning("No self-play data generated this iteration.")
            empty_tensor = torch.empty(0, device=self.device)
            return ((empty_tensor, empty_tensor, empty_tensor), [])

        self.total_games_played += self.num_games_per_iteration

        # Convert to Tensors
        inputs_tensor = torch.from_numpy(np.array(all_inputs, dtype=np.float32)).to(self.device)
        policy_targets_tensor = torch.from_numpy(np.array(all_policy_targets, dtype=np.float32)).to(self.device)
        value_targets_tensor = torch.tensor(all_value_targets, dtype=torch.float32, device=self.device)

        return ((inputs_tensor, policy_targets_tensor, value_targets_tensor), pgn_games_list)