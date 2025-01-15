import os
import time
import threading
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import torch.optim as optim
from multiprocessing import Pool, cpu_count, Manager
from PyQt5.QtCore import pyqtSignal
import chess
import chess.pgn
from src.base.base_worker import BaseWorker
from src.models.model import ChessModel
from src.utils.chess_utils import convert_board_to_tensor, get_move_mapping, get_total_moves
from src.utils.common_utils import format_time_left, initialize_optimizer, initialize_random_seeds, initialize_scheduler, update_progress_time_left, wait_if_paused, get_game_result, policy_value_fn, compute_policy_loss, compute_value_loss, compute_total_loss
from src.utils.mcts import MCTS
from src.utils.checkpoint_manager import CheckpointManager

def play_and_collect_wrapper(args: Tuple) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], List[int]]:
    (model_state_dict, device_type, simulations, c_puct, temperature, games_per_process, stop_event, pause_event, seed, stats_queue, move_mapping, total_moves) = args

    initialize_random_seeds(seed)
    device = torch.device(device_type)
    model = ChessModel(total_moves).to(device)

    try:
        model.load_state_dict(model_state_dict)
    except Exception as e:
        stats_queue.put({"error": f"Failed to load state_dict in worker: {str(e)}"})
        return ([], [], [], [], [])

    model.eval()

    inputs_list: List[np.ndarray] = []
    policy_targets_list: List[np.ndarray] = []
    value_targets_list: List[float] = []
    results_list: List[float] = []
    game_lengths_list: List[int] = []
    avg_mcts_visits_list: List[float] = []

    try:
        for _ in range(games_per_process):
            if stop_event.is_set():
                break
            wait_if_paused(pause_event)

            board = chess.Board()
            mcts = MCTS(lambda board: policy_value_fn(board, model, device), c_puct, simulations)
            mcts.set_root_node(board)
            states: List[np.ndarray] = []
            mcts_probs: List[np.ndarray] = []
            current_players: List[bool] = []
            move_count = 0
            max_moves = 200
            total_visits = 0
            num_moves = 0

            while not board.is_game_over() and move_count < max_moves:
                action_probs = mcts.get_move_probs(temperature)
                if not action_probs:
                    break

                moves_list = list(action_probs.keys())
                probs_array = np.array(list(action_probs.values()), dtype=np.float32)
                probs_array /= np.sum(probs_array)
                chosen_move = np.random.choice(moves_list, p=probs_array)

                board_tensor = convert_board_to_tensor(board)
                states.append(board_tensor)

                prob_arr = np.zeros(total_moves, dtype=np.float32)
                for move, prob in action_probs.items():
                    idx = move_mapping.get_index_by_move(move)
                    if idx is not None and 0 <= idx < total_moves:
                        prob_arr[idx] = prob
                mcts_probs.append(prob_arr)

                current_players.append(board.turn)
                board.push(chosen_move)
                mcts.update_with_move(chosen_move)

                if mcts.root:
                    total_visits += mcts.root.n_visits
                    num_moves += 1

                move_count += 1

            result = get_game_result(board)
            if board.is_checkmate():
                last_player = not board.turn
                winners = [result if player == last_player else -result for player in current_players]
            else:
                winners = [0.0 for _ in current_players]

            game_length = len(states)
            visits_avg = (total_visits / num_moves) if num_moves > 0 else 0.0

            inputs_list.extend(states)
            policy_targets_list.extend(mcts_probs)
            value_targets_list.extend(winners)
            results_list.append(result)
            game_lengths_list.append(game_length)
            avg_mcts_visits_list.append(visits_avg)

        total_games = len(results_list)
        wins = results_list.count(1.0)
        losses = results_list.count(-1.0)
        draws = results_list.count(0.0)
        avg_length = (sum(game_lengths_list) / len(game_lengths_list)) if game_lengths_list else 0.0
        avg_visits = (sum(avg_mcts_visits_list) / len(avg_mcts_visits_list)) if avg_mcts_visits_list else 0.0

        stats_queue.put({"total_games": total_games, "wins": wins, "losses": losses, "draws": draws, "avg_game_length": avg_length, "avg_mcts_visits": avg_visits})

    except Exception as e:
        stats_queue.put({"error": f"Exception in play_and_collect_wrapper: {str(e)}"})

    return (inputs_list, policy_targets_list, value_targets_list, results_list, game_lengths_list)

class ReinforcementWorker(BaseWorker):
    stats_update = pyqtSignal(dict)

    def __init__(self, model_path: Optional[str], num_iterations: int, num_games_per_iteration: int, simulations: int, c_puct: float, temperature: float, num_epochs: int, batch_size: int, num_threads: int, save_checkpoints: bool, checkpoint_interval: int, checkpoint_type: str, checkpoint_interval_minutes: int, checkpoint_batch_interval: int, random_seed: int = 42, optimizer_type: str = "adamw", learning_rate: float = 0.0001, weight_decay: float = 1e-4, scheduler_type: str = "cosineannealingwarmrestarts"):
        super().__init__()

        # Checkpoint settings
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_type = checkpoint_type.lower()
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_batch_interval = checkpoint_batch_interval
        self.checkpoint_dir = os.path.join("models", "checkpoints", "reinforcement")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # General settings
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

        # Optimization settings
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type

        # Initialize random seeds and threading lock
        initialize_random_seeds(self.random_seed)
        self.lock = threading.Lock()

        # Logging and state tracking
        self.results: List[float] = []
        self.game_lengths: List[int] = []
        self.total_games_played: int = 0
        self.total_batches_processed: int = 0
        self.current_epoch = 1
        self.batch_idx: Optional[int] = None
        self.start_iteration: int = 0

        # Model and training setup
        self.move_mapping = get_move_mapping()
        self.total_moves = get_total_moves()
        self.model = ChessModel(self.total_moves).to(self.device)
        self.scaler = GradScaler(device="cuda") if self.device.type == "cuda" else GradScaler()
        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay, logger=self.logger)
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir, checkpoint_type=self.checkpoint_type, checkpoint_interval=self.checkpoint_interval, logger=self.logger)

        # Training state
        batches_per_epoch = ((self.num_games_per_iteration // self.batch_size) * self.num_epochs if self.batch_size else (self.num_games_per_iteration // 128) * self.num_epochs)
        self.total_steps = self.num_iterations * self.num_games_per_iteration + self.num_iterations * batches_per_epoch
        self.model_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def run_task(self):
        self.logger.info("Initializing reinforcement worker with model and optimizer.")
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = self.checkpoint_manager.load(self.model_path, self.device, self.model, self.optimizer, self.scheduler)
            if checkpoint:
                self.start_iteration = checkpoint.get('iteration', 0)
                self.current_epoch = checkpoint.get('epoch', 1)
                self.batch_idx = checkpoint.get('batch_idx', None)
                self.results = checkpoint.get('training_stats', {}).get('results', [])
                self.game_lengths = checkpoint.get('training_stats', {}).get('game_lengths', [])
                self.total_games_played = checkpoint.get('training_stats', {}).get('total_games_played', 0)
                self.total_batches_processed = checkpoint.get('batch_idx', 0)
                self.logger.info(f"Resuming from iteration {self.start_iteration}, epoch {self.current_epoch}, batch {self.batch_idx}.")
            else:
                self.logger.info("No valid checkpoint found. Starting from scratch.")
        else:
            self.logger.info("No checkpoint path provided or file does not exist. Starting from scratch.")

        self.start_time = time.time()

        for iteration in range(self.start_iteration, self.num_iterations):
            if self._is_stopped.is_set():
                break

            iteration_start = time.time()
            self.logger.info(f"Starting iteration {iteration + 1}/{self.num_iterations}.")
            self.current_epoch = 1

            # Generate self-play data
            self.model.eval()
            self_play_data = self._generate_self_play_data()
            self.model.train()

            # Train on self-play data
            self._train_on_self_play_data(self_play_data, iteration)

            # Save checkpoint if required (Iteration Checkpoint)
            if self.save_checkpoints and self.checkpoint_type == "iteration":
                checkpoint_data = {
                    'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'iteration': iteration + 1,
                    'training_stats': {'total_games_played': self.total_games_played, 'results': self.results, 'game_lengths': self.game_lengths},
                    'batch_idx': self.total_batches_processed,
                    'epoch': self.current_epoch,
                }
                if self.checkpoint_manager.should_save(iteration=iteration + 1):
                    with self.lock:
                        self.checkpoint_manager.save(checkpoint_data)
                    self.logger.info(f"Checkpoint saved at iteration {iteration + 1}.")

            iteration_time = time.time() - iteration_start
            self.logger.info(f"Iteration {iteration + 1} finished in {format_time_left(iteration_time)}.")

            # Emit statistics
            if self.stats_update.emit:
                self.stats_update.emit({"iteration": iteration + 1, "total_games_played": self.total_games_played})

        # Save the final model if training completed without interruption
        final_dir = os.path.join("models", "saved_models")
        final_path = os.path.join(final_dir, "final_model.pth")
        os.makedirs(final_dir, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "training_stats": {"total_games_played": self.total_games_played, "results": self.results, "game_lengths": self.game_lengths}
        }
        try:
            torch.save(checkpoint, final_path)
            self.logger.info(f"Final model saved at {final_path}")
        except Exception as e:
            self.logger.error(f"Error saving final model: {str(e)}")

        self.task_finished.emit()
        self.finished.emit()

    def _generate_self_play_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_processes = min(self.num_threads, cpu_count())
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
        seeds = [self.random_seed + i for i in range(num_processes)]
        args = []

        for i in range(num_processes):
            gpp = games_per_process + (1 if i < remainder else 0)
            args.append((self.model_state_dict if self.model_state_dict else {k: v.cpu() for k, v in self.model.state_dict().items()}, 
                         self.device.type, self.simulations, self.c_puct, self.temperature, gpp, stop_event, pause_event, seeds[i], stats_queue, self.move_mapping, self.total_moves))

        with Pool(processes=num_processes) as pool:
            results = pool.map(play_and_collect_wrapper, args)

        # Process statistics from workers
        while not stats_queue.empty():
            stat = stats_queue.get()
            if "error" in stat:
                self.logger.error(stat["error"])
                continue
            if self.stats_update.emit:
                self.stats_update.emit(stat)
            with self.lock:
                self.total_batches_processed += stat.get("total_games", 0)
            update_progress_time_left(self.progress_update, self.time_left_update, self.start_time, self.total_batches_processed, self.total_steps)

        # Aggregate results from all processes
        inputs_list: List[np.ndarray] = []
        policy_targets_list: List[np.ndarray] = []
        value_targets_list: List[float] = []

        for res in results:
            inputs_list.extend(res[0])
            policy_targets_list.extend(res[1])
            value_targets_list.extend(res[2])
            self.results.extend(res[3])
            self.game_lengths.extend(res[4])

        total_positions = len(inputs_list)
        if total_positions == 0:
            self.logger.warning("No self-play data generated this iteration. Skipping training.")
            return (torch.empty(0, device=self.device), torch.empty(0, device=self.device), torch.empty(0, device=self.device))

        self.total_games_played += self.num_games_per_iteration

        inputs = torch.from_numpy(np.array(inputs_list, dtype=np.float32)).to(self.device)
        policy_targets = torch.from_numpy(np.array(policy_targets_list, dtype=np.float32)).to(self.device)
        value_targets = torch.tensor(value_targets_list, dtype=torch.float32, device=self.device)

        return inputs, policy_targets, value_targets

    def _train_on_self_play_data(self, self_play_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], iteration: int):
        inputs, policy_targets, value_targets = self_play_data
        if inputs.numel() == 0:
            return

        dataset = TensorDataset(inputs.cpu(), policy_targets.cpu(), value_targets.cpu())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=(self.device.type == "cuda"), num_workers=min(os.cpu_count(), 8))

        try:
            if self.scheduler is None:
                self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, total_steps=self.num_epochs * len(loader), logger=self.logger)
        except ValueError as ve:
            self.logger.error(f"Scheduler initialization error: {str(ve)}")
            self.scheduler = None

        start_epoch = self.current_epoch
        epoch_start = time.time()
        accumulation_steps = max(256 // self.batch_size, 1) if self.batch_size else 1
        accumulate_count = 0

        for epoch in range(start_epoch, self.num_epochs + 1):
            if self._is_stopped.is_set():
                break

            self.logger.info(f"Epoch {epoch}/{self.num_epochs}, iteration {iteration + 1}.")
            self.current_epoch = epoch
            train_iterator = iter(loader)

            if epoch == start_epoch and self.batch_idx is not None:
                skip_batches = self.batch_idx
                if skip_batches >= len(loader):
                    continue
                for _ in range(skip_batches):
                    try:
                        next(train_iterator)
                    except StopIteration:
                        break

            total_loss = 0.0
            local_steps = 0

            for batch_idx, (batch_inputs, batch_policy_targets, batch_value_targets) in enumerate(train_iterator, 1):
                if self._is_stopped.is_set():
                    break

                wait_if_paused(self._is_paused)

                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_policy_targets = batch_policy_targets.to(self.device, non_blocking=True)
                batch_value_targets = batch_value_targets.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type):
                    policy_preds, value_preds = self.model(batch_inputs)
                    policy_loss = compute_policy_loss(policy_preds, batch_policy_targets)
                    value_loss = compute_value_loss(value_preds, batch_value_targets)
                    loss = compute_total_loss(policy_loss, value_loss, self.batch_size)

                self.scaler.scale(loss).backward()
                accumulate_count += 1

                if (accumulate_count % accumulation_steps == 0) or (batch_idx == len(loader)):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    accumulate_count = 0

                if self.scheduler:
                    self.scheduler.step(epoch - 1 + batch_idx / len(loader))

                # Accumulate losses
                total_loss += (policy_loss.item() + value_loss.item()) * batch_inputs.size(0)

                # Clean up to free memory
                del batch_inputs, batch_policy_targets, batch_value_targets, policy_preds, value_preds, loss
                torch.cuda.empty_cache()

                self.batch_idx = batch_idx
                with self.lock:
                    self.total_batches_processed += 1
                    local_steps += 1

                # Emit progress and metrics
                update_progress_time_left(self.progress_update, self.time_left_update, self.start_time, self.total_batches_processed, self.total_steps)

                # Save checkpoints based on type
                if self.save_checkpoints and self.checkpoint_type == "batch":
                    if self.checkpoint_manager.should_save(batch_idx=self.total_batches_processed):
                        self._save_checkpoint(iteration=iteration + 1)

            # Calculate average loss for the epoch
            avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
            duration = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch}/{self.num_epochs} average loss: {avg_loss:.4f} in {format_time_left(duration)}.")

            # Emit statistics
            if self.stats_update.emit:
                self.stats_update.emit({"iteration": iteration + 1, "epoch": epoch, "avg_loss": avg_loss, "total_games_played": self.total_games_played})

            epoch_start = time.time()

    def _save_checkpoint(self, iteration: int):
        with self.lock:
            checkpoint_data = {
                'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'epoch': self.current_epoch,
                'batch_idx': self.total_batches_processed,
                'iteration': iteration,
                'training_stats': {'total_games_played': self.total_games_played, 'results': self.results, 'game_lengths': self.game_lengths},
            }
            self.checkpoint_manager.save(checkpoint_data)
            self.logger.info(f"Checkpoint saved at iteration {iteration}.")