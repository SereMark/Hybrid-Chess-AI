import os, time, chess, torch, chess.pgn, numpy as np
from torch.cuda.amp import GradScaler
from typing import List, Tuple, Dict, Optional
from multiprocessing import Pool, cpu_count, Manager
from torch.utils.data import DataLoader, TensorDataset
from src.models.transformer import TransformerChessModel
from src.utils.checkpoint_manager import CheckpointManager
from src.training.reinforcement.play_and_collect_worker import PlayAndCollectWorker
from src.utils.chess_utils import get_total_moves
from src.utils.train_utils import initialize_optimizer, initialize_random_seeds, initialize_scheduler, train_epoch

class ReinforcementWorker:
    def __init__(self, model_path:Optional[str], num_iterations:int, num_games_per_iteration:int, simulations:int, c_puct:float, temperature:float, num_epochs:int, batch_size:int, num_threads:int, save_checkpoints:bool, checkpoint_interval:int, checkpoint_type:str, random_seed:int=42, optimizer_type:str="adamw", learning_rate:float=0.0001, weight_decay:float=1e-4, scheduler_type:str="cosineannealingwarmrestarts", progress_callback=None, status_callback=None):
        self.model_path = model_path
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_type = checkpoint_type
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        initialize_random_seeds(self.random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerChessModel(get_total_moves()).to(self.device)
        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay)
        self.scheduler = None
        self.scaler = GradScaler()
        self.checkpoint_dir = os.path.join("models", "checkpoints", "reinforcement")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir, checkpoint_type=self.checkpoint_type, checkpoint_interval=self.checkpoint_interval)
        self.results:List[float] = []
        self.game_lengths:List[int] = []
        self.total_games_played = 0
        self.total_batches_processed = 0

    def run(self) -> Dict:
        try:
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = self.checkpoint_manager.load(self.model_path, self.device, self.model, self.optimizer, self.scheduler)
                if checkpoint:
                    self.start_iteration = checkpoint.get("iteration", 0)
                    training_stats = checkpoint.get("training_stats", {})
                    self.results = training_stats.get("results", [])
                    self.game_lengths = training_stats.get("game_lengths", [])
                    self.total_games_played = training_stats.get("total_games_played", 0)
                    self.total_batches_processed = training_stats.get("total_batches_processed", 0)
            else:
                self.start_iteration = 0
            for iteration in range(self.start_iteration, self.num_iterations):
                self.model.eval()
                self.status_callback(f"🔁 Iteration {iteration+1}/{self.num_iterations} 🎮 Generating self-play data...")
                self_play_data, pgn_games = self._generate_self_play_data()
                self.model.train()
                timestamp = int(time.time())
                for idx, game in enumerate(pgn_games, start=1):
                    filename = os.path.join("data", "games", "self-play", f"game_{timestamp}_{idx}.pgn")
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(str(game))
                if len(self_play_data) == 3:
                    inputs, policy_targets, value_targets = self_play_data
                    dataset = TensorDataset(inputs.cpu(), policy_targets.cpu(), value_targets.cpu())
                    data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=(self.device.type=="cuda"), num_workers=min(os.cpu_count(),8))
                    if self.scheduler is None and self.scheduler_type.lower() != 'none':
                        total_training_steps = self.num_epochs * len(data_loader)
                        self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, total_steps=total_training_steps)
                    for epoch in range(1, self.num_epochs +1):
                        train_epoch(model=self.model, data_loader=data_loader, device=self.device, scaler=self.scaler, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, accumulation_steps=max(256 // self.batch_size, 1), batch_size=self.batch_size, smooth_policy_targets=False, compute_accuracy_flag=False, total_batches_processed=self.total_batches_processed, progress_callback=self.progress_callback, status_callback=self.status_callback)
                if self.save_checkpoints:
                    self.checkpoint_manager.save(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=None, iteration=iteration+1, training_stats={"total_games_played":self.total_games_played, "results":self.results, "game_lengths":self.game_lengths, "total_batches_processed":self.total_batches_processed})
            final_model_path = os.path.join("models", "saved_models", "reinforcement_model.pth")
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            self.checkpoint_manager.save_final_model(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, training_stats={"total_games_played":self.total_games_played, "results":self.results, "game_lengths":self.game_lengths}, final_path=final_model_path)
            metrics = {
                "final_model_path": final_model_path,
                "total_iterations": self.num_iterations,
                "total_games_played": self.total_games_played,
                "total_batches_processed": self.total_batches_processed
            }
            return metrics
        except Exception as e:
            self.status_callback(f"Training error: {e}")
            return {}

    def _generate_self_play_data(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[chess.pgn.Game]]:
        num_processes = max(min(self.num_threads, cpu_count()), 1)
        games_per_process = self.num_games_per_iteration // num_processes
        remainder = self.num_games_per_iteration % num_processes
        manager = Manager()
        stats_queue = manager.Queue()
        seeds = [self.random_seed + i + int(time.time()) for i in range(num_processes)]
        self.model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        tasks = [(self.model_state_dict, self.device.type, self.simulations, self.c_puct, self.temperature, games_per_process + (1 if i < remainder else 0), seeds[i], stats_queue) for i in range(num_processes)]
        with Pool(processes=num_processes) as pool:
            results = pool.map(PlayAndCollectWorker.run_process, tasks)
        while not stats_queue.empty():
            stat = stats_queue.get()
            if "error" in stat:
                self.status_callback(f"Subprocess error: {stat['error']}")
                continue
        all_inputs, all_policy_targets, all_value_targets, pgn_games_list = [], [], [], []
        for inp, pol, val, res, g_len, pgns in results:
            all_inputs.extend(inp)
            all_policy_targets.extend(pol)
            all_value_targets.extend(val)
            self.results.extend(res)
            self.game_lengths.extend(g_len)
            pgn_games_list.extend(pgns)
        if not all_inputs:
            return (torch.empty(0, device=self.device), torch.empty(0, device=self.device), torch.empty(0, device=self.device)), []
        self.total_games_played += self.num_games_per_iteration
        inputs_tensor = torch.from_numpy(np.array(all_inputs, dtype=np.float32)).to(self.device)
        policy_targets_tensor = torch.from_numpy(np.array(all_policy_targets, dtype=np.float32)).to(self.device)
        value_targets_tensor = torch.tensor(all_value_targets, dtype=torch.float32, device=self.device)
        return (inputs_tensor, policy_targets_tensor, value_targets_tensor), pgn_games_list