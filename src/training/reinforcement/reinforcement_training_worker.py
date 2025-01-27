import os, time, torch, numpy as np
from torch.cuda.amp import GradScaler
from typing import Optional
from multiprocessing import Pool, cpu_count, Manager
from torch.utils.data import DataLoader, TensorDataset
from src.models.transformer import TransformerChessModel
from src.utils.checkpoint_manager import CheckpointManager
from src.training.reinforcement.play_and_collect_worker import PlayAndCollectWorker
from src.utils.chess_utils import get_total_moves
from src.utils.train_utils import initialize_optimizer, initialize_random_seeds, initialize_scheduler, train_epoch

class ReinforcementWorker:
    def __init__(self, model_path:Optional[str], num_iterations:int, num_games_per_iteration:int, simulations:int, c_puct:float, temperature:float, num_epochs:int, batch_size:int, num_threads:int, checkpoint_interval:int, random_seed:int=42, optimizer_type:str="adamw", learning_rate:float=0.0001, weight_decay:float=1e-4, scheduler_type:str="cosineannealingwarmrestarts", accumulation_steps:int=3, num_workers:int=4, policy_weight:float=1.0, value_weight:float=2.0, progress_callback=None, status_callback=None):
        self.model_path, self.num_iterations, self.num_games_per_iteration = model_path, num_iterations, num_games_per_iteration
        self.simulations, self.c_puct, self.temperature = simulations, c_puct, temperature
        self.num_epochs, self.batch_size, self.num_threads, self.checkpoint_interval = num_epochs, batch_size, num_threads, checkpoint_interval
        self.num_workers, self.random_seed, self.optimizer_type = num_workers, random_seed, optimizer_type
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.scheduler_type, self.progress_callback = scheduler_type, progress_callback
        self.status_callback = status_callback
        self.policy_weight, self.value_weight = policy_weight, value_weight
        initialize_random_seeds(self.random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerChessModel(get_total_moves()).to(self.device)
        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay)
        self.scheduler = None
        self.scaler = GradScaler()
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = os.path.join("models", "checkpoints", "reinforcement")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir, checkpoint_type='iteration', checkpoint_interval=self.checkpoint_interval)
        self.results, self.game_lengths, self.total_games_played = [], [], 0

    def run(self):
        start_iteration = 1
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = self.checkpoint_manager.load(self.model_path, self.device, self.model, self.optimizer, self.scheduler)
            if checkpoint:
                start_iteration = checkpoint.get('iteration', 0) + 1
        for iteration in range(start_iteration, self.num_iterations + 1):
            self.model.eval()
            self.status_callback(f"🔁 Iteration {iteration}/{self.num_iterations} 🎮 Generating self-play data...")
            num_processes = max(min(self.num_threads, cpu_count()), 1)
            games_per_process, remainder = self.num_games_per_iteration // num_processes, self.num_games_per_iteration % num_processes
            _, stats_queue = Manager(), Manager().Queue()
            seeds = [self.random_seed + i + int(time.time()) for i in range(num_processes)]
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            tasks = [(model_state, self.device.type, self.simulations, self.c_puct, self.temperature, games_per_process + (1 if i < remainder else 0), seeds[i], stats_queue) for i in range(num_processes)]
            with Pool(processes=num_processes) as pool:
                results = pool.map(PlayAndCollectWorker.run_process, tasks)
            while not stats_queue.empty():
                stat = stats_queue.get()
                if "error" in stat:
                    self.status_callback(f"Subprocess error: {stat['error']}")
            all_inputs, all_policy, all_value, pgn_games = [], [], [], []
            for inp, pol, val, res, g_len, pgns in results:
                all_inputs.extend(inp)
                all_policy.extend(pol)
                all_value.extend(val)
                self.results.extend(res)
                self.game_lengths.extend(g_len)
                pgn_games.extend(pgns)
            if not all_inputs:
                continue
            self.total_games_played += self.num_games_per_iteration
            inputs_tensor = torch.from_numpy(np.array(all_inputs, dtype=np.float32))
            policy_tensor = torch.from_numpy(np.array(all_policy, dtype=np.float32))
            value_tensor = torch.tensor(all_value, dtype=torch.float32)
            data_loader = DataLoader(TensorDataset(inputs_tensor, policy_tensor, value_tensor), batch_size=self.batch_size, shuffle=True, pin_memory=(self.device.type == "cuda"), num_workers=self.num_workers)
            if not self.scheduler and self.scheduler_type.lower() != 'none':
                self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, total_steps=self.num_epochs * len(data_loader))
            for epoch in range(1, self.num_epochs + 1):
                train_epoch(model=self.model, data_loader=data_loader, device=self.device, scaler=self.scaler, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, accumulation_steps=self.accumulation_steps, batch_size=self.batch_size, smooth_policy_targets=False, compute_accuracy_flag=False, progress_callback=self.progress_callback, status_callback=self.status_callback, policy_weight=self.policy_weight, value_weight=self.value_weight)
            if self.checkpoint_interval and self.checkpoint_interval > 0:
                self.checkpoint_manager.save(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, iteration=iteration)
            for idx, game in enumerate(pgn_games, start=1):
                with open(os.path.join("data", "games", "self-play", f"game_{int(time.time())}_{idx}.pgn"), "w", encoding="utf-8") as f:
                    f.write(str(game))
        final_model_path = os.path.join("models", "saved_models", "reinforcement_model.pth")
        self.checkpoint_manager.save_final_model(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, iteration=self.num_iterations, final_path=final_model_path)
        return True