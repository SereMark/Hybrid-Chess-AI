import os, threading, time, numpy as np, torch, torch.nn.functional as F, torch.optim as optim, random
from torch.utils.data import DataLoader, TensorDataset
from src.self_play.self_play import SelfPlay
from src.models.model import ChessModel
from torch.cuda.amp import autocast, GradScaler
from multiprocessing import Pool, cpu_count, Event, Manager
from src.utils.chess_utils import estimate_batch_size


def _play_and_collect_wrapper(args):
    (
        model_state_dict,
        device,
        simulations,
        c_puct,
        temperature,
        games_per_process,
        stop_event,
        pause_event,
        seed,
        stats_queue
    ) = args

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    inputs_list = []
    policy_targets_list = []
    value_targets_list = []
    results_list = []
    game_lengths_list = []

    self_play = SelfPlay(
        model_state_dict=model_state_dict,
        device=device,
        n_simulations=simulations,
        c_puct=c_puct,
        temperature=temperature,
        stats_fn=stats_queue.put
    )
    for _ in range(games_per_process):
        if stop_event.is_set():
            break
        while not pause_event.is_set():
            time.sleep(0.1)
        (
            states,
            mcts_probs,
            winners,
            game_length,
            result,
        ) = self_play.play_game()
        inputs_list.extend(states)
        policy_targets_list.extend(mcts_probs)
        value_targets_list.extend(winners)
        results_list.append(result)
        game_lengths_list.append(game_length)
    return (
        inputs_list,
        policy_targets_list,
        value_targets_list,
        results_list,
        game_lengths_list,
    )


class SelfPlayTrainer:
    def __init__(
        self,
        model_path,
        output_dir,
        num_iterations,
        num_games_per_iteration,
        simulations,
        c_puct,
        temperature,
        num_epochs,
        batch_size,
        automatic_batch_size,
        num_threads,
        stop_event,
        pause_event,
        log_fn=None,
        progress_fn=None,
        time_left_fn=None,
        stats_fn=None,
        checkpoint_path=None,
        random_seed=42
    ):
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
        self.stop_event = stop_event or Event()
        self.pause_event = pause_event or Event()
        self.pause_event.set()
        self.log_fn = log_fn
        self.progress_fn = progress_fn
        self.time_left_fn = time_left_fn
        self.stats_fn = stats_fn
        self.checkpoint_path = checkpoint_path
        self.random_seed = random_seed
        self.start_time = None
        self.total_games_played = 0
        self.results = []
        self.game_lengths = []
        self.lock = threading.Lock()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler(enabled=(self.device == 'cuda'))

    def train(self):
        self._initialize()
        for iteration in range(self.start_iteration, self.num_iterations):
            if self.stop_event.is_set():
                break
            iteration_start_time = time.time()
            if self.log_fn:
                self.log_fn(f"\n=== Iteration {iteration + 1}/{self.num_iterations} ===")
            self.model.eval()
            self_play_data = self._generate_self_play_data()
            self.model.train()
            self._train_on_self_play_data(self_play_data)
            self._save_model(iteration)
            iteration_time = time.time() - iteration_start_time
            if self.log_fn:
                self.log_fn(f"Iteration {iteration + 1} completed in {self._format_time(iteration_time)}")
        self._save_final_model()

    def _initialize(self):
        if self.log_fn:
            self.log_fn("Initializing model and optimizer...")
        self.model = ChessModel().to(self.device)
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.log_fn:
                self.log_fn("Model loaded successfully.")
        else:
            if self.log_fn:
                self.log_fn("Model file not found. Starting from scratch.")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=2e-4)
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_iteration = checkpoint.get('iteration', 0)
            if self.log_fn:
                self.log_fn(f"Resuming from checkpoint at iteration {self.start_iteration}.")
        else:
            self.start_iteration = 0
        if self.automatic_batch_size:
            self.batch_size = estimate_batch_size(self.model, self.device)
            if self.log_fn:
                self.log_fn(f"Automatic batch size estimation: Using batch size {self.batch_size}")
        else:
            if self.log_fn:
                self.log_fn(f"Using manual batch size: {self.batch_size}")
        self.start_time = time.time()

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        self.model_state_dict = self.model.state_dict()

    def _generate_self_play_data(self):
        num_processes = min(self.num_threads, cpu_count())
        games_per_process = self.num_games_per_iteration // num_processes
        manager = Manager()
        stop_event = manager.Event()
        pause_event = manager.Event()
        stats_queue = manager.Queue()
        if self.stop_event.is_set():
            stop_event.set()
        if self.pause_event.is_set():
            pause_event.set()

        seeds = [self.random_seed + i for i in range(num_processes)]

        args = [
            (
                self.model_state_dict,
                self.device,
                self.simulations,
                self.c_puct,
                self.temperature,
                games_per_process,
                stop_event,
                pause_event,
                seeds[i],
                stats_queue
            )
            for i in range(num_processes)
        ]
        with Pool(processes=num_processes) as pool:
            results = pool.map(_play_and_collect_wrapper, args)

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
            return (
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
            )
        inputs = torch.from_numpy(np.array(inputs_list, dtype=np.float32)).to(self.device)
        policy_targets = torch.from_numpy(np.array(policy_targets_list, dtype=np.float32)).to(self.device)
        value_targets = torch.tensor(value_targets_list, dtype=torch.float32, device=self.device)
        self.total_games_played += self.num_games_per_iteration
        return inputs, policy_targets, value_targets

    def _train_on_self_play_data(self, self_play_data):
        inputs, policy_targets, value_targets = self_play_data
        if inputs.numel() == 0:
            return
        dataset = TensorDataset(inputs.cpu(), policy_targets.cpu(), value_targets.cpu())
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(self.device == 'cuda'),
            num_workers=min(os.cpu_count(), 8),
        )
        for epoch in range(self.num_epochs):
            if self.stop_event.is_set():
                break
            while not self.pause_event.is_set():
                time.sleep(0.1)
            total_loss = 0
            for batch_inputs, batch_policy_targets, batch_value_targets in loader:
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_policy_targets = batch_policy_targets.to(self.device, non_blocking=True)
                batch_value_targets = batch_value_targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                with autocast():
                    policy_preds, value_preds = self.model(batch_inputs)
                    policy_loss = -(batch_policy_targets * torch.log_softmax(policy_preds, dim=1)).mean()
                    value_loss = F.mse_loss(value_preds.view(-1), batch_value_targets)
                    loss = policy_loss + value_loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()
                del batch_inputs, batch_policy_targets, batch_value_targets
                torch.cuda.empty_cache()
            avg_loss = total_loss / len(loader)
            if self.log_fn:
                self.log_fn(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

    def _save_model(self, iteration):
        os.makedirs(self.output_dir, exist_ok=True)
        model_save_path = os.path.join(self.output_dir, f'model_iteration_{iteration + 1}.pth')
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': iteration + 1,
            'training_stats': {
                'total_games_played': self.total_games_played,
                'results': self.results,
                'game_lengths': self.game_lengths,
            },
        }
        torch.save(checkpoint, model_save_path)
        if self.log_fn:
            self.log_fn(f"Model saved at iteration {iteration + 1}.")
        self.model_path = model_save_path

    def _save_final_model(self):
        final_model_dir = os.path.join('models', 'saved_models')
        final_model_path = os.path.join(final_model_dir, 'final_model.pth')
        os.makedirs(final_model_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'training_stats': {
                'total_games_played': self.total_games_played,
                'results': self.results,
                'game_lengths': self.game_lengths,
            },
        }
        torch.save(checkpoint, final_model_path)
        if self.log_fn:
            self.log_fn("Final model saved.")

    @staticmethod
    def _format_time(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"