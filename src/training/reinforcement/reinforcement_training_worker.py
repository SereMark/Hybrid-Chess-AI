import os, time, torch, numpy as np
from torch.cuda.amp import GradScaler
from multiprocessing import Pool, cpu_count, Manager
from torch.utils.data import DataLoader, TensorDataset
from src.models.transformer import TransformerChessModel
from src.utils.checkpoint_manager import CheckpointManager
from src.training.reinforcement.play_and_collect_worker import PlayAndCollectWorker
from src.utils.chess_utils import get_total_moves
from src.utils.train_utils import initialize_optimizer, initialize_scheduler, initialize_random_seeds, train_epoch

class ReinforcementWorker:
    def __init__(self, model_path, num_iterations, num_games_per_iteration, simulations, c_puct, temperature, num_epochs, batch_size, num_threads, checkpoint_interval, random_seed, optimizer_type, learning_rate, weight_decay, scheduler_type, accumulation_steps, num_workers, policy_weight, value_weight, grad_clip, momentum, wandb_flag, progress_callback, status_callback):
        self.model_path, self.num_iterations, self.num_games_per_iteration = model_path, num_iterations, num_games_per_iteration
        self.simulations, self.c_puct, self.temperature = simulations, c_puct, temperature
        self.num_epochs, self.batch_size, self.num_threads = num_epochs, batch_size, num_threads
        self.checkpoint_interval, self.random_seed, self.num_workers = checkpoint_interval, random_seed, num_workers
        self.optimizer_type, self.learning_rate = optimizer_type, learning_rate
        self.weight_decay, self.grad_clip, self.momentum = weight_decay, grad_clip, momentum
        self.scheduler_type, self.wandb = scheduler_type, wandb_flag
        self.progress_callback, self.status_callback = progress_callback, status_callback
        self.policy_weight, self.value_weight = policy_weight, value_weight
        initialize_random_seeds(self.random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerChessModel(get_total_moves()).to(self.device)
        self.optimizer = initialize_optimizer(self.model, self.optimizer_type, self.learning_rate, self.weight_decay, self.momentum)
        self.scheduler = None
        self.scaler = GradScaler()
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = os.path.join("models", "checkpoints", "reinforcement")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir, 'iteration', self.checkpoint_interval)
        self.results, self.game_lengths, self.total_games_played = [], [], 0

    def run(self):
        if self.wandb:
            import wandb
            wandb.init(entity="chess_ai", project="chess_ai_app", config=self.__dict__, reinit=True)
            wandb.watch(self.model, log="all", log_freq=100)
        start_iteration, best_metric, best_iteration, training_start = 1, float('inf'), 0, time.time()
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = self.checkpoint_manager.load(self.model_path, self.device, self.model, self.optimizer, self.scheduler)
            start_iteration = checkpoint.get('iteration', 0) + 1 if checkpoint else 1
        history = []
        for iteration in range(start_iteration, self.num_iterations + 1):
            self.model.eval()
            self.status_callback(f"🔁 Iteration {iteration}/{self.num_iterations} 🎮 Generating self-play data...")
            num_processes = max(min(self.num_threads, cpu_count()), 1)
            games, rem = divmod(self.num_games_per_iteration, num_processes)
            seeds = [self.random_seed + i + int(time.time()) for i in range(num_processes)]
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            tasks = [(model_state, self.device.type, self.simulations, self.c_puct, self.temperature, games + (1 if i < rem else 0), seeds[i], Manager().Queue()) for i in range(num_processes)]
            with Pool(processes=num_processes) as pool:
                results = pool.map(PlayAndCollectWorker.run_process, tasks)
            stats = {}
            for task in tasks:
                while not task[7].empty():
                    stat = task[7].get()
                    for k, v in stat.items():
                        stats[k] = stats.get(k, 0) + v
            if self.wandb:
                history.append([iteration, stats.get("wins",0), stats.get("losses",0), stats.get("draws",0)])
                table = wandb.Table(data=history, columns=["iteration", "wins", "losses", "draws"])
                wandb.log({
                    "iteration": iteration, "total_games": stats.get("total_games",0),
                    "wins": stats.get("wins",0), "losses": stats.get("losses",0),
                    "draws": stats.get("draws",0), "avg_game_length": stats.get("avg_game_length",0),
                    "game_outcomes": wandb.plot.bar(table, "iteration", ["wins", "losses", "draws"], title="Game Outcomes")
                })
                for idx, layer in enumerate(self.model.transformer_encoder.layers[:3]):
                    wandb.log({f"hist/weight_layer_{idx}": wandb.Histogram(layer.self_attn.in_proj_weight.detach().cpu().numpy()), f"hist/grad_layer_{idx}": wandb.Histogram(layer.self_attn.in_proj_weight.grad.detach().cpu().numpy() if layer.self_attn.in_proj_weight.grad is not None else np.zeros_like(layer.self_attn.in_proj_weight.detach().cpu().numpy()))})
                attn = self.model.transformer_encoder.layers[0].self_attn.in_proj_weight
                wandb.log({"attention_mean": attn.mean().item(), "attention_std": attn.std().item()})
            current_metric = stats.get("policy_loss", float('inf')) + stats.get("value_loss", float('inf'))
            if current_metric < best_metric:
                best_metric, best_iteration = current_metric, iteration
                if self.wandb:
                    wandb.run.summary.update({"best_val_loss": best_metric, "best_iteration": best_iteration})
            self.status_callback(f"Iteration {iteration} - Games: {stats.get('total_games',0)}, Wins: {stats.get('wins',0)}, Avg Length: {stats.get('avg_game_length',0):.2f}")
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
            data_loader = DataLoader(TensorDataset(inputs_tensor, policy_tensor, value_tensor), batch_size=self.batch_size, shuffle=True, pin_memory=self.device.type == "cuda", num_workers=self.num_workers)
            self.scheduler = initialize_scheduler(self.optimizer, self.scheduler_type, self.num_epochs * len(data_loader))
            for epoch in range(1, self.num_epochs + 1):
                train_metrics = train_epoch(self.model, data_loader, self.device, self.scaler, self.optimizer, self.scheduler, epoch, self.num_epochs, self.accumulation_steps, self.batch_size, False, False,
                                            self.policy_weight, self.value_weight, self.grad_clip, self.progress_callback, self.status_callback, self.wandb)
                if self.wandb:
                    wandb.log({**train_metrics, "epoch": epoch, "iteration": iteration})
            if self.checkpoint_interval > 0:
                self.checkpoint_manager.save(self.model, self.optimizer, self.scheduler, None, iteration)
            for idx, game in enumerate(pgn_games, 1):
                os.makedirs("data/games/self-play", exist_ok=True)
                with open(f"data/games/self-play/game_{int(time.time())}_{idx}.pgn", "w", encoding="utf-8") as f:
                    f.write(str(game))
        self.checkpoint_manager.save_final_model(self.model, self.optimizer, self.scheduler, None, self.num_iterations, "models/saved_models/reinforcement_model.pth")
        if self.wandb:
            wandb.run.summary.update({"metric": best_metric, "best_iteration": best_iteration, "training_time": time.time() - training_start})
            wandb.finish()
        return True