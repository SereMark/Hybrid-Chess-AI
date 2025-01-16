import os
import time
import threading
from typing import List, Tuple, Dict, Optional
import numpy as np
import chess
import chess.pgn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from multiprocessing import Pool, cpu_count, Manager
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.models.model import ChessModel
from src.utils.chess_utils import convert_board_to_tensor, get_move_mapping, get_total_moves
from src.utils.common_utils import format_time_left, initialize_optimizer, initialize_random_seeds, initialize_scheduler, update_progress_time_left, wait_if_paused, get_game_result, policy_value_fn, compute_policy_loss, compute_value_loss, compute_total_loss
from src.utils.mcts import MCTS
from src.utils.checkpoint_manager import CheckpointManager

def play_and_collect_wrapper(args: Tuple) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], List[int], List[chess.pgn.Game]]:
    (model_state_dict, device_type, simulations, c_puct, temperature, games_per_process, stop_event, pause_event, seed, stats_queue, move_mapping, total_moves) = args

    # Initialize seeds and device
    initialize_random_seeds(seed)
    device = torch.device(device_type)

    # Create a fresh model for this subprocess
    model = ChessModel(total_moves).to(device)

    # Safely load model weights
    try:
        model.load_state_dict(model_state_dict)
    except Exception as e:
        stats_queue.put({"error": f"Failed to load state_dict in worker: {str(e)}"})
        return ([], [], [], [], [], [])

    model.eval()

    # Lists for collecting training data
    inputs_list: List[np.ndarray] = []
    policy_targets_list: List[np.ndarray] = []
    value_targets_list: List[float] = []
    results_list: List[float] = []
    game_lengths_list: List[int] = []
    pgn_games_list: List[chess.pgn.Game] = []

    try:
        for _ in range(games_per_process):
            if stop_event.is_set():
                break
            wait_if_paused(pause_event)

            board = chess.Board()
            mcts = MCTS(policy_value_fn=lambda b: policy_value_fn(b, model, device), c_puct=c_puct, n_simulations=simulations)
            mcts.set_root_node(board)

            states: List[np.ndarray] = []
            mcts_probs: List[np.ndarray] = []
            current_players: List[bool] = []

            move_count = 0
            max_moves = 200

            total_visits = 0
            num_moves = 0

            # Create a new PGN Game
            game = chess.pgn.Game()
            game.headers["Event"] = "Reinforcement Self-Play"
            game.headers["Site"] = "Self-Play"
            game.headers["Date"] = time.strftime("%Y.%m.%d")
            game.headers["Round"] = "-"
            game.headers["White"] = "Agent"
            game.headers["Black"] = "Opponent"
            game.headers["Result"] = "*"
            node = game

            # Self-play loop
            while not board.is_game_over() and move_count < max_moves:
                action_probs = mcts.get_move_probs(temperature)
                if not action_probs:
                    break

                moves_list = list(action_probs.keys())
                probs_array = np.array(list(action_probs.values()), dtype=np.float32)
                probs_array /= np.sum(probs_array)
                chosen_move = np.random.choice(moves_list, p=probs_array)

                # Save board state
                board_tensor = convert_board_to_tensor(board)
                states.append(board_tensor)

                # Create probability vector for each possible move
                prob_arr = np.zeros(total_moves, dtype=np.float32)
                for move, prob in action_probs.items():
                    idx = move_mapping.get_index_by_move(move)
                    if idx is not None and 0 <= idx < total_moves:
                        prob_arr[idx] = prob
                mcts_probs.append(prob_arr)

                # Track current player
                current_players.append(board.turn)

                # Make the move
                try:
                    board.push(chosen_move)
                except ValueError:
                    break
                node = node.add_variation(chosen_move)
                mcts.update_with_move(chosen_move)

                if mcts.root:
                    total_visits += mcts.root.n_visits
                    num_moves += 1

                move_count += 1

            # Determine game result
            result = get_game_result(board)
            if board.is_checkmate():
                # last move was by the winning side
                last_player = not board.turn
                winners = [
                    result if player == last_player else -result
                    for player in current_players
                ]
            else:
                # For draws or any non-checkmate end
                winners = [0.0 for _ in current_players]

            game_length = len(states)

            # Set PGN result
            if result > 0:
                game.headers["Result"] = "1-0"
            elif result < 0:
                game.headers["Result"] = "0-1"
            else:
                game.headers["Result"] = "1/2-1/2"

            # Collect for stats
            pgn_games_list.append(game)
            inputs_list.extend(states)
            policy_targets_list.extend(mcts_probs)
            value_targets_list.extend(winners)
            results_list.append(result)
            game_lengths_list.append(game_length)

        # Worker-level stats
        total_games = len(results_list)
        wins = results_list.count(1.0)
        losses = results_list.count(-1.0)
        draws = results_list.count(0.0)
        avg_length = (sum(game_lengths_list) / len(game_lengths_list)) if game_lengths_list else 0.0

        stats_queue.put({
            "total_games": total_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "avg_game_length": avg_length,
        })

    except Exception as e:
        stats_queue.put({"error": f"Exception in play_and_collect_wrapper: {str(e)}"})

    return (inputs_list, policy_targets_list, value_targets_list, results_list, game_lengths_list, pgn_games_list)

class ReinforcementWorker(BaseWorker):
    stats_update = pyqtSignal(dict)

    def __init__(
        self,
        model_path: Optional[str],
        num_iterations: int,
        num_games_per_iteration: int,
        simulations: int,
        c_puct: float,
        temperature: float,
        num_epochs: int,
        batch_size: int,
        num_threads: int,
        save_checkpoints: bool,
        checkpoint_interval: int,
        checkpoint_type: str,
        checkpoint_interval_minutes: int,
        checkpoint_batch_interval: int,
        random_seed: int = 42,
        optimizer_type: str = "adamw",
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosineannealingwarmrestarts"
    ):
        super().__init__()

        # Checkpoint settings
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_type = checkpoint_type.lower()
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_batch_interval = checkpoint_batch_interval
        self.checkpoint_dir = os.path.join("models", "checkpoints", "reinforcement")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # General training settings
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

        # Initialize random seeds, locks
        initialize_random_seeds(self.random_seed)
        self.lock = threading.Lock()

        # Logging / state tracking
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

        # Use GradScaler for mixed precision
        self.scaler = GradScaler()

        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay, logger=self.logger)
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir, checkpoint_type=self.checkpoint_type, checkpoint_interval=self.checkpoint_interval, logger=self.logger)

        # Steps for progress bar
        # Each iteration => self-play games + (epochs * steps) in training
        # Approximate steps for scheduling progress.
        self_play_batches = max(self.num_games_per_iteration // max(self.batch_size, 1), 1)
        self.train_steps_per_iter = self.num_epochs * self_play_batches
        self.total_steps = (self.num_iterations * self.num_games_per_iteration + self.num_iterations * self.train_steps_per_iter)

        # For storing model state dict in subprocess
        self.model_state_dict: Optional[Dict[str, torch.Tensor]] = None

        # Self-play directory
        self.self_play_dir = os.path.join("data", "games", "self-play")
        os.makedirs(self.self_play_dir, exist_ok=True)

    def run_task(self):
        self.logger.info("Initializing reinforcement worker with model and optimizer.")

        # Attempt to load from existing checkpoint
        self._try_load_checkpoint()

        self.start_time = time.time()

        # Main iterative loop
        for iteration in range(self.start_iteration, self.num_iterations):
            if self._is_stopped.is_set():
                break

            iteration_start_time = time.time()
            self.logger.info(f"Starting iteration {iteration + 1}/{self.num_iterations}.")
            self.current_epoch = 1

            self.model.eval()
            self_play_data, pgn_games = self._generate_self_play_data()
            self.model.train()

            self._save_self_play_games(pgn_games, iteration)

            self._train_on_self_play_data(self_play_data, iteration)

            if self.save_checkpoints and self.checkpoint_type == "iteration":
                self._save_checkpoint(iteration=iteration + 1)

            # Log iteration time
            iter_duration = time.time() - iteration_start_time
            self.logger.info(f"Iteration {iteration + 1} finished in {format_time_left(iter_duration)}.")

            # Emit iteration statistics to UI if needed
            if self.stats_update is not None and hasattr(self.stats_update, "emit"):
                self.stats_update.emit({
                    "iteration": iteration + 1,
                    "total_games_played": self.total_games_played
                })

        self._save_final_model()

        self.task_finished.emit()
        self.finished.emit()

    def _try_load_checkpoint(self):
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = self.checkpoint_manager.load(self.model_path, self.device, self.model, self.optimizer, self.scheduler)
            if checkpoint:
                self.start_iteration = checkpoint.get('iteration', 0)
                self.current_epoch = checkpoint.get('epoch', 1)
                self.batch_idx = checkpoint.get('batch_idx', None)
                training_stats = checkpoint.get('training_stats', {})
                self.results = training_stats.get('results', [])
                self.game_lengths = training_stats.get('game_lengths', [])
                self.total_games_played = training_stats.get('total_games_played', 0)
                self.total_batches_processed = checkpoint.get('batch_idx', 0)
                self.logger.info(
                    f"Resuming from iteration {self.start_iteration}, "
                    f"epoch {self.current_epoch}, batch {self.batch_idx}."
                )
            else:
                self.logger.info("No valid checkpoint found. Starting from scratch.")
        else:
            self.logger.info("No checkpoint path provided or file does not exist. Starting from scratch.")

    def _save_checkpoint(self, iteration: int):
        if not self.save_checkpoints:
            return

        # Use lock to ensure thread-safe file writing
        with self.lock:
            checkpoint_data = {
                'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'epoch': self.current_epoch,
                'batch_idx': self.total_batches_processed,
                'iteration': iteration,
                'training_stats': {
                    'total_games_played': self.total_games_played,
                    'results': self.results,
                    'game_lengths': self.game_lengths,
                },
            }
            if self.checkpoint_manager.should_save(iteration=iteration, batch_idx=self.total_batches_processed):
                self.checkpoint_manager.save(checkpoint_data)
                self.logger.info(f"Checkpoint saved at iteration {iteration}.")

    def _save_final_model(self):
        final_dir = os.path.join("models", "saved_models")
        final_path = os.path.join(final_dir, "final_model.pth")
        os.makedirs(final_dir, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "training_stats": {
                "total_games_played": self.total_games_played,
                "results": self.results,
                "game_lengths": self.game_lengths,
            }
        }
        try:
            torch.save(checkpoint, final_path)
            self.logger.info(f"Final model saved at {final_path}")
        except Exception as e:
            self.logger.error(f"Error saving final model: {str(e)}")

    def _generate_self_play_data(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[chess.pgn.Game]]:
        num_processes = min(self.num_threads, cpu_count())
        if num_processes < 1:
            num_processes = 1

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

        # Prepare arguments for each subprocess
        self._prepare_model_state_dict()
        tasks = []
        for i in range(num_processes):
            gpp = games_per_process + (1 if i < remainder else 0)
            tasks.append((
                self.model_state_dict,
                self.device.type,
                self.simulations,
                self.c_puct,
                self.temperature,
                gpp,
                stop_event,
                pause_event,
                seeds[i],
                stats_queue,
                self.move_mapping,
                self.total_moves
            ))

        # Launch parallel self-play
        with Pool(processes=num_processes) as pool:
            # results is a list of:
            #   (inputs, policy_targets, value_targets, results, game_lengths, pgn_games_list)
            results = pool.map(play_and_collect_wrapper, tasks)

        # Consolidate stats from queue
        while not stats_queue.empty():
            stat = stats_queue.get()
            if "error" in stat:
                self.logger.error(stat["error"])
                continue
            if self.stats_update is not None and hasattr(self.stats_update, "emit"):
                self.stats_update.emit(stat)

            with self.lock:
                # total_games from each worker adds to total_batches_processed
                self.total_batches_processed += stat.get("total_games", 0)

            # Update UI progress
            update_progress_time_left(
                progress_signal=self.progress_update,
                time_left_signal=self.time_left_update,
                start_time=self.start_time,
                current_step=self.total_batches_processed,
                total_steps=self.total_steps
            )

        # Aggregate data
        all_inputs: List[np.ndarray] = []
        all_policy_targets: List[np.ndarray] = []
        all_value_targets: List[float] = []
        pgn_games_list: List[chess.pgn.Game] = []

        for (inp, pol, val, res, g_len, pgns) in results:
            all_inputs.extend(inp)
            all_policy_targets.extend(pol)
            all_value_targets.extend(val)
            self.results.extend(res)
            self.game_lengths.extend(g_len)
            pgn_games_list.extend(pgns)

        # If no data is collected, return empty
        if not all_inputs:
            self.logger.warning("No self-play data generated this iteration. Skipping training.")
            empty_tensor = torch.empty(0, device=self.device)
            return ((empty_tensor, empty_tensor, empty_tensor), [])

        self.total_games_played += self.num_games_per_iteration

        # Convert to torch tensors
        inputs_tensor = torch.from_numpy(np.array(all_inputs, dtype=np.float32)).to(self.device)
        policy_targets_tensor = torch.from_numpy(np.array(all_policy_targets, dtype=np.float32)).to(self.device)
        value_targets_tensor = torch.tensor(all_value_targets, dtype=torch.float32, device=self.device)

        return ((inputs_tensor, policy_targets_tensor, value_targets_tensor), pgn_games_list)

    def _prepare_model_state_dict(self):
        # Only do this once per iteration to avoid overhead
        self.model_state_dict = {
            k: v.cpu() for k, v in self.model.state_dict().items()
        }

    def _save_self_play_games(self, pgn_games: List[chess.pgn.Game], iteration: int):
        timestamp = int(time.time())
        for idx, game in enumerate(pgn_games, start=1):
            filename = os.path.join(self.self_play_dir, f"iteration_{iteration + 1}_game_{timestamp}_{idx}.pgn")
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(str(game))
                self.logger.info(f"Saved self-play game to {filename}")
            except Exception as e:
                self.logger.error(f"Failed to save PGN for self-play game {idx} in iteration {iteration + 1}: {e}")

    def _train_on_self_play_data(self, self_play_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], iteration: int):
        if not isinstance(self_play_data, tuple):
            self.logger.error("Invalid self-play data format. Skipping training.")
            return

        if len(self_play_data) != 3:
            self.logger.error("Received incorrectly formatted self-play data.")
            return

        inputs, policy_targets, value_targets = self_play_data
        if inputs.numel() == 0:
            self.logger.warning("Empty training data. Skipping training.")
            return

        dataset = TensorDataset(inputs.cpu(), policy_targets.cpu(), value_targets.cpu())
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=(self.device.type == "cuda"), num_workers=min(os.cpu_count(), 8))

        # Initialize scheduler if not present
        if self.scheduler is None:
            try:
                total_training_steps = self.num_epochs * len(data_loader)
                self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, total_steps=total_training_steps, logger=self.logger)
            except ValueError as ve:
                self.logger.error(f"Scheduler initialization error: {str(ve)}")
                self.scheduler = None

        # Training loop
        start_epoch = self.current_epoch
        accumulation_steps = max(256 // max(self.batch_size, 1), 1)
        accumulate_count = 0

        epoch_start_time = time.time()
        for epoch in range(start_epoch, self.num_epochs + 1):
            if self._is_stopped.is_set():
                break

            self.logger.info(f"Epoch {epoch}/{self.num_epochs}, iteration {iteration + 1}.")
            self.current_epoch = epoch

            # Possibly skip batches if resuming mid-epoch
            train_iterator = iter(data_loader)
            if epoch == start_epoch and self.batch_idx is not None:
                skip_batches = self.batch_idx
                if skip_batches < len(data_loader):
                    for _ in range(skip_batches):
                        try:
                            next(train_iterator)
                        except StopIteration:
                            break

            total_loss = 0.0

            for batch_idx, (batch_inputs, batch_policy, batch_value) in enumerate(train_iterator, 1):
                if self._is_stopped.is_set():
                    break

                wait_if_paused(self._is_paused)

                # Move data to device
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_policy = batch_policy.to(self.device, non_blocking=True)
                batch_value = batch_value.to(self.device, non_blocking=True)

                # Use autocast with an "enabled" flag for broader version compatibility
                with autocast(enabled=(self.device.type == 'cuda')):
                    policy_preds, value_preds = self.model(batch_inputs)
                    # Use the new distribution-based policy loss
                    policy_loss = compute_policy_loss(policy_preds, batch_policy, False)
                    value_loss = compute_value_loss(value_preds, batch_value)
                    loss = compute_total_loss(policy_loss, value_loss, self.batch_size)

                # Gradient scaling
                self.scaler.scale(loss).backward()
                accumulate_count += 1

                # Gradient accumulation
                if (accumulate_count % accumulation_steps == 0) or (batch_idx == len(data_loader)):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    accumulate_count = 0

                # Step scheduler
                if self.scheduler:
                    # Step scheduler by fraction of an epoch
                    self.scheduler.step(epoch - 1 + batch_idx / len(data_loader))

                # Track total loss
                total_loss += (policy_loss.item() + value_loss.item()) * batch_inputs.size(0)

                # Cleanup references
                del batch_inputs, batch_policy, batch_value, policy_preds, value_preds, loss

                self.batch_idx = batch_idx
                with self.lock:
                    self.total_batches_processed += 1

                # Update UI progress
                update_progress_time_left(progress_signal=self.progress_update, time_left_signal=self.time_left_update, start_time=self.start_time, current_step=self.total_batches_processed, total_steps=self.total_steps)

                # Save checkpoint if needed (type == "batch")
                if self.save_checkpoints and self.checkpoint_type == "batch":
                    self._save_checkpoint(iteration=iteration + 1)

            # End of epoch
            avg_loss = total_loss / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0.0
            epoch_duration = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch}/{self.num_epochs} average loss: {avg_loss:.4f} in {format_time_left(epoch_duration)}.")

            # Emit stats
            if self.stats_update is not None and hasattr(self.stats_update, "emit"):
                self.stats_update.emit({
                    "iteration": iteration + 1,
                    "epoch": epoch,
                    "avg_loss": avg_loss,
                    "total_games_played": self.total_games_played
                })

            epoch_start_time = time.time()