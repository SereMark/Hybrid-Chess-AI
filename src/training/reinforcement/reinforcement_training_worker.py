import os, time, numpy as np, torch, chess, torch.nn.functional as F, threading
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from multiprocessing import Pool, cpu_count, Manager
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.models.model import ChessModel
from src.utils.chess_utils import get_total_moves, get_move_mapping, convert_board_to_tensor, get_game_result
from src.utils.common_utils import initialize_random_seeds, format_time_left, wait_if_paused, initialize_model, initialize_optimizer, initialize_scheduler
from src.utils.mcts import MCTS
from src.utils.checkpoint_manager import CheckpointManager

def play_and_collect_wrapper(args):
    model_state_dict, device_type, simulations, c_puct, temperature, games_per_process, stop_event, pause_event, seed, stats_queue, move_mapping, total_moves = args
    initialize_random_seeds(seed)
    device = torch.device(device_type)
    model = ChessModel(num_moves=total_moves)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    inputs_list = []
    policy_targets_list = []
    value_targets_list = []
    results_list = []
    game_lengths_list = []
    avg_mcts_visits_list = []
    @torch.no_grad()
    def policy_value_fn(board):
        board_tensor = convert_board_to_tensor(board)
        board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(device)
        policy_logits, value = model(board_tensor)
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.cpu().item()
        legal_moves = list(board.legal_moves)
        action_probs = {}
        total_legal_prob = 0
        for move in legal_moves:
            move_index = move_mapping.get_index_by_move(move)
            if move_index is not None and move_index < len(policy):
                prob = max(policy[move_index], 1e-8)
                action_probs[move] = prob
                total_legal_prob += prob
            else:
                action_probs[move] = 1e-8
        if total_legal_prob > 0:
            action_probs = {k: v / total_legal_prob for k, v in action_probs.items()}
        else:
            action_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        return action_probs, value
    for _ in range(games_per_process):
        if stop_event.is_set():
            break
        wait_if_paused(pause_event)
        board = chess.Board()
        mcts = MCTS(policy_value_fn, c_puct, simulations)
        mcts.set_root_node(board)
        states, mcts_probs, current_players = [], [], []
        move_count = 0
        max_moves = 200
        total_visits = 0
        num_moves = 0
        while not board.is_game_over() and move_count < max_moves:
            action_probs = mcts.get_move_probs(temperature)
            moves = list(action_probs.keys())
            probs = np.array(list(action_probs.values()))
            probs /= np.sum(probs)
            move = np.random.choice(moves, p=probs)
            board_tensor = convert_board_to_tensor(board)
            states.append(board_tensor)
            prob_array = np.zeros(total_moves, dtype=np.float32)
            for m, p in action_probs.items():
                move_index = move_mapping.get_index_by_move(m)
                if move_index is not None and 0 <= move_index < total_moves:
                    prob_array[move_index] = p
            mcts_probs.append(prob_array)
            current_players.append(board.turn)
            board.push(move)
            mcts.update_with_move(move)
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
        avg_mcts_visits = (total_visits / num_moves if num_moves > 0 else 0)
        inputs_list.extend(states)
        policy_targets_list.extend(mcts_probs)
        value_targets_list.extend(winners)
        results_list.append(result)
        game_lengths_list.append(game_length)
        avg_mcts_visits_list.append(avg_mcts_visits)
    total_games = len(results_list)
    wins = results_list.count(1.0)
    losses = results_list.count(-1.0)
    draws = results_list.count(0.0)
    avg_game_length = (sum(game_lengths_list) / len(game_lengths_list) if game_lengths_list else 0)
    avg_mcts_visits = (sum(avg_mcts_visits_list) / len(avg_mcts_visits_list) if avg_mcts_visits_list else 0)
    stats = {
        "total_games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_game_length": avg_game_length,
        "avg_mcts_visits": avg_mcts_visits
    }
    stats_queue.put(stats)
    return (inputs_list, policy_targets_list, value_targets_list, results_list, game_lengths_list)

class ReinforcementWorker(BaseWorker):
    stats_update = pyqtSignal(dict)
    def __init__(self,model_path,num_iterations,num_games_per_iteration,simulations,c_puct,temperature,num_epochs,batch_size,automatic_batch_size,num_threads,save_checkpoints,checkpoint_interval,checkpoint_type,checkpoint_interval_minutes,checkpoint_batch_interval,checkpoint_path=None,random_seed=42,optimizer_type="adamw",learning_rate=0.0005,weight_decay=2e-4,scheduler_type="cosineannealingwarmrestarts"):
        super().__init__()
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.checkpoint_batch_interval = checkpoint_batch_interval
        self.checkpoint_dir = os.path.join("models", "checkpoints", "reinforcement")
        self.random_seed = random_seed
        self.automatic_batch_size = automatic_batch_size
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.num_workers = num_threads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        initialize_random_seeds(self.random_seed)
        self.model_path = model_path
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.num_epochs = num_epochs
        self.num_threads = num_threads
        self.checkpoint_path = checkpoint_path
        self.stats_fn = self.stats_update.emit
        self.lock = threading.Lock()
        self.total_batches_processed = 0
        self.results = []
        self.game_lengths = []
        self.total_games_played = 0
        self.scaler = GradScaler(device="cuda") if self.device.type == "cuda" else GradScaler()
        self.current_epoch = 1
        self.batch_idx = None
        self.start_iteration = 0
        self.move_mapping = get_move_mapping()
        self.total_moves = get_total_moves()
        self.model, inferred_batch_size = initialize_model(ChessModel, num_moves=self.total_moves, device=self.device, automatic_batch_size=self.automatic_batch_size, logger=self.logger)
        if inferred_batch_size:
            self.batch_size = inferred_batch_size
        self.optimizer = initialize_optimizer(self.model,self.optimizer_type,self.learning_rate,self.weight_decay,logger=self.logger)
        self.scheduler = None
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir,checkpoint_type=self.checkpoint_type,checkpoint_interval=self.checkpoint_interval,logger=self.logger)

    def run_task(self):
        self.logger.info("Initializing reinforcement worker with model and optimizer.")
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = self.checkpoint_manager.load(self.checkpoint_path, self.device, self.model, self.optimizer, self.scheduler)
            if checkpoint:
                self.start_iteration = checkpoint.get("iteration", 0)
                training_stats = checkpoint.get("training_stats", {})
                self.total_games_played = training_stats.get("total_games_played", 0)
                self.results = training_stats.get("results", [])
                self.game_lengths = training_stats.get("game_lengths", [])
                self.logger.info(f"Resuming training from iteration {self.start_iteration}.")
            else:
                self.logger.warning("No valid checkpoint data found. Starting reinforcement training from scratch.")
                self.start_iteration = 0
                self.current_epoch = 1
        else:
            self.logger.info("No checkpoint specified or checkpoint not found. Training from scratch.")
            self.start_iteration = 0
            self.current_epoch = 1
        self.start_time = time.time()
        self.model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for iteration in range(self.start_iteration, self.num_iterations):
            if self._is_stopped.is_set():
                break
            iteration_start_time = time.time()
            self.logger.info(f"Starting iteration {iteration + 1}/{self.num_iterations}.")
            self.current_epoch = 1
            self.model.eval()
            self_play_data = self._generate_self_play_data()
            self.model.train()
            self._train_on_self_play_data(self_play_data, iteration)
            self.batch_idx = None
            if self.save_checkpoints and self.checkpoint_manager.should_save(iteration=iteration + 1):
                checkpoint_data = {
                    "model_state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "iteration": iteration + 1,
                    "training_stats": {
                        "total_games_played": self.total_games_played,
                        "results": self.results,
                        "game_lengths": self.game_lengths
                    }
                }
                self.checkpoint_manager.save(checkpoint_data)
            iteration_time = time.time() - iteration_start_time
            self.logger.info(f"Iteration {iteration + 1} finished in {format_time_left(iteration_time)}.")
        final_model_dir = os.path.join("models", "saved_models")
        final_model_path = os.path.join(final_model_dir, "final_model.pth")
        os.makedirs(final_model_dir, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "training_stats": {
                "total_games_played": self.total_games_played,
                "results": self.results,
                "game_lengths": self.game_lengths
            }
        }
        torch.save(checkpoint, final_model_path)
        self.logger.info("Reinforcement training completed. Final model saved.")
        self.task_finished.emit()
        self.finished.emit()

    def _generate_self_play_data(self):
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
            games = games_per_process + (1 if i < remainder else 0)
            args.append((self.model_state_dict,self.device.type,self.simulations,self.c_puct,self.temperature,games,stop_event,pause_event,seeds[i],stats_queue,self.move_mapping,self.total_moves))
        with Pool(processes=num_processes) as pool:
            results = pool.map(play_and_collect_wrapper, args)
        while not stats_queue.empty():
            stats = stats_queue.get()
            if self.stats_fn:
                self.stats_fn(stats)
        inputs_list = []
        policy_targets_list = []
        value_targets_list = []
        for res in results:
            inputs_list.extend(res[0])
            policy_targets_list.extend(res[1])
            value_targets_list.extend(res[2])
            self.results.extend(res[3])
            self.game_lengths.extend(res[4])
        total_positions = len(inputs_list)
        if total_positions == 0:
            self.logger.warning("No self-play data generated this iteration. Continuing to next iteration.")
            return (torch.empty(0, device=self.device), torch.empty(0, device=self.device), torch.empty(0, device=self.device))
        inputs = torch.from_numpy(np.array(inputs_list, dtype=np.float32)).to(self.device)
        policy_targets = torch.from_numpy(np.array(policy_targets_list, dtype=np.float32)).to(self.device)
        value_targets = torch.tensor(value_targets_list, dtype=torch.float32, device=self.device)
        self.total_games_played += self.num_games_per_iteration
        return inputs, policy_targets, value_targets

    def _train_on_self_play_data(self, self_play_data, iteration):
        inputs, policy_targets, value_targets = self_play_data
        if inputs.numel() == 0:
            return
        dataset = TensorDataset(inputs.cpu(), policy_targets.cpu(), value_targets.cpu())
        loader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True,pin_memory=(self.device.type == "cuda"),num_workers=min(os.cpu_count(), 8))
        total_steps = self.num_epochs * len(loader)
        try:
            self.scheduler = initialize_scheduler(self.optimizer,self.scheduler_type,total_steps=total_steps,logger=self.logger)
        except ValueError as ve:
            self.logger.error(f"Scheduler initialization error: {str(ve)}")
            self.scheduler = None
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.num_epochs + 1):
            if self._is_stopped.is_set():
                break
            self.logger.info(f"Starting epoch {epoch}/{self.num_epochs} for iteration {iteration + 1}.")
            self.current_epoch = epoch
            train_iterator = iter(loader)
            if epoch == start_epoch and self.batch_idx is not None:
                skip_batches = self.batch_idx
                if skip_batches >= len(loader):
                    self.logger.warning("Skip count exceeds total number of batches. Skipping the entire epoch.")
                    continue
                for _ in range(skip_batches):
                    try:
                        next(train_iterator)
                    except StopIteration:
                        break
            total_loss = 0.0
            for batch_idx, (batch_inputs, batch_policy_targets, batch_value_targets) in enumerate(train_iterator, 1):
                if self._is_stopped.is_set():
                    break
                wait_if_paused(self._is_paused)
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_policy_targets = batch_policy_targets.to(self.device, non_blocking=True)
                batch_value_targets = batch_value_targets.to(self.device, non_blocking=True)
                with autocast(device_type=self.device.type):
                    policy_preds, value_preds = self.model(batch_inputs)
                    policy_loss = -(batch_policy_targets * torch.log_softmax(policy_preds, dim=1)).mean()
                    value_loss = F.mse_loss(value_preds.view(-1), batch_value_targets)
                    loss = policy_loss + value_loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                total_loss += loss.item()
                del batch_inputs, batch_policy_targets, batch_value_targets
                torch.cuda.empty_cache()
                self.batch_idx = batch_idx
                if self.scheduler:
                    self.scheduler.step(epoch - 1 + batch_idx / len(loader))
                self.total_batches_processed += 1
                current_progress = min(int((self.total_batches_processed / total_steps) * 100), 100)
                if self.progress_update:
                    self.progress_update.emit(current_progress)
                if self.save_checkpoints and self.checkpoint_manager.should_save(iteration=iteration + 1, batch_idx=self.total_batches_processed):
                    checkpoint_data = {
                        "model_state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                        "epoch": epoch,
                        "batch_idx": self.total_batches_processed,
                        "iteration": iteration + 1,
                        "training_stats": {
                            "total_games_played": self.total_games_played,
                            "results": self.results,
                            "game_lengths": self.game_lengths
                        }
                    }
                    self.checkpoint_manager.save(checkpoint_data)
            avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
            self.logger.info(f"Epoch {epoch}/{self.num_epochs} completed with average loss: {avg_loss:.4f}")