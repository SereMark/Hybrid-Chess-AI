import os
import time
import torch
import wandb
import chess
import chess.pgn
import numpy as np
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor

from src.model import ChessModel
from src.utils.mcts import MCTS
from src.utils.config import Config
from src.utils.chess import BoardHistory
from src.utils.checkpoint import Checkpoint
from src.utils.chess import board_to_input, get_move_map, get_move_count
from src.utils.train import set_seed, get_optimizer, get_scheduler, get_device, train_epoch

class SelfPlay:
    @staticmethod
    def run_games(model_state, device_type, sims, c_puct, temp, games, seed, channels=32, blocks=4, use_attn=True):
        set_seed(seed)
        
        device = torch.device(device_type)
        
        model = ChessModel(
            get_move_count(),
            ch=channels,
            blocks=blocks,
            use_attn=use_attn
        )
        
        try:
            updated_state = {}
            for k, v in model_state.items():
                if isinstance(v, torch.Tensor):
                    updated_state[k] = v.to(device)
                else:
                    updated_state[k] = v
                    
            model.load_state_dict(updated_state, strict=False)
        except Exception as e:
            print(f"Error loading model state: {e}")
            try:
                model.load_state_dict(model_state, strict=False)
            except Exception as e2:
                print(f"Both loading methods failed: {e2}")
                return [], [], [], {}, []
                
        model.to(device).eval()
        
        inputs, policies, values = [], [], []
        pgns = []
        stats = {
            "wins": 0, "losses": 0, "draws": 0,
            "game_lens": [], "results": []
        }
        
        move_map = get_move_map()
        
        for game_idx in range(games):
            board = chess.Board()
            mcts = MCTS(model, device, c_puct, sims)
            mcts.set_root(board)
            
            board_history = BoardHistory(max_history=7)
            board_history.add_board(board.copy())
            
            states = []
            move_probs = []
            players = []
            move_count = 0
            
            game = chess.pgn.Game()
            game.headers.update({
                "Event": "Self-Play",
                "Site": "Colab",
                "Date": time.strftime("%Y.%m.%d"),
                "Round": str(game_idx + 1),
                "White": "Agent",
                "Black": "Agent",
                "Result": "*"
            })
            
            node = game
            
            while not board.is_game_over() and move_count < 200:
                action_probs = mcts.get_move_probs(temp)
                
                if move_count == 0 and action_probs:
                    moves_list = list(action_probs.keys())
                    noise = np.random.dirichlet([0.3] * len(moves_list))
                    for i, move in enumerate(moves_list):
                        action_probs[move] = 0.75 * action_probs[move] + 0.25 * noise[i]
                
                if not action_probs:
                    break
                
                states.append(board_to_input(board, board_history))
                
                policy_vector = np.zeros(get_move_count(), dtype=np.float32)
                for move, prob in action_probs.items():
                    idx = move_map.idx_by_move(move)
                    if idx is not None and 0 <= idx < get_move_count():
                        policy_vector[idx] = prob
                
                move_probs.append(policy_vector)
                players.append(board.turn)
                
                moves = list(action_probs.keys())
                probs = np.array(list(action_probs.values()), dtype=np.float32)
                probs /= probs.sum()
                
                move = np.random.choice(moves, p=probs)
                
                try:
                    board.push(move)
                    board_history.add_board(board.copy())
                except ValueError:
                    break
                
                node = node.add_variation(move)
                mcts.update_with_move(move)
                move_count += 1
            
            result_map = {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0}
            outcome = result_map.get(board.result(), 0.0)
            
            game.headers["Result"] = '1-0' if outcome > 0 else '0-1' if outcome < 0 else '1/2-1/2'
            
            exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
            pgn_string = game.accept(exporter)
            pgns.append(pgn_string)
            
            rewards = []
            for player in players:
                sign = 1.0 if player == chess.WHITE else -1.0
                rewards.append(outcome * sign)
            
            inputs.extend(states)
            policies.extend(move_probs)
            values.extend(rewards)
            
            stats['game_lens'].append(move_count)
            stats['results'].append(outcome)
            
            if outcome == 1.0:
                stats['wins'] += 1
            elif outcome == -1.0:
                stats['losses'] += 1
            else:
                stats['draws'] += 1
        
        return inputs, policies, values, stats, pgns

class ReinforcementPipeline:
    def __init__(self, config: Config):
        self.config = config
        
        seed = config.get('project.seed', 42)
        set_seed(seed)
        
        self.device_info = get_device()
        self.device = self.device_info["device"]
        self.device_type = self.device_info["type"]
        
        print(f"Using device: {self.device_type}")
        
        channels = config.get('model.channels', 64)
        blocks = config.get('model.blocks', 4)
        use_attention = config.get('model.attention', True)
        
        self.model = ChessModel(
            moves=get_move_count(), 
            ch=channels,
            blocks=blocks,
            use_attn=use_attention
        ).to(self.device)
        
        self.iters = config.get('reinforcement.iters', 10)
        self.games_per_iter = config.get('reinforcement.games_per_iter', 100)
        self.sims_per_move = config.get('reinforcement.sims_per_move', 100)
        self.epochs_per_iter = config.get('reinforcement.epochs_per_iter', 5)
        self.c_puct = config.get('reinforcement.c_puct', 1.4)
        self.temp = config.get('reinforcement.temp', 0.5)
        self.threads = config.get('reinforcement.threads', 4)
        
        self.batch = config.get('data.batch', 128)
        
        self.optimizer_type = config.get('supervised.optimizer', 'adamw')
        self.lr = config.get('supervised.lr', 0.001)
        self.weight_decay = config.get('supervised.weight_decay', 0.0001)
        self.scheduler_type = config.get('supervised.scheduler', 'onecycle')
        self.momentum = config.get('supervised.momentum', 0.9)
        self.accum_steps = config.get('supervised.accum_steps', 1)
        self.policy_weight = config.get('supervised.policy_weight', 1.0)
        self.value_weight = config.get('supervised.value_weight', 1.0)
        self.grad_clip = config.get('supervised.grad_clip', 1.0)
        
        self.optimizer = get_optimizer(
            self.model, self.optimizer_type, self.lr, self.weight_decay, self.momentum
        )
        self.scheduler = None
        
        ckpt_dir = os.path.join(
            config.get('paths.models', 'models'), 
            'checkpoints', 
            'reinforcement'
        )
        ckpt_interval = config.get('training.checkpoint_interval', 5)
        self.ckpt = Checkpoint(ckpt_dir, 'iteration', ckpt_interval)
        
        self.use_amp = (
            self.device_type == "gpu" and 
            config.get('hardware.mixed_precision', True)
        )
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.model_path = None
        self.start_iter = 1
        self.best_metric = float('inf')
        self.best_iter = 0
    
    def setup(self):
        print("Setting up reinforcement learning pipeline...")
        
        try:
            local_model_path = '/content/drive/MyDrive/chess_ai/models/supervised_model.pth'
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            
            if os.path.exists(local_model_path):
                self.model_path = local_model_path
                print(f"Using supervised model: {self.model_path}")
            else:
                print("No supervised model found, starting from scratch")
        except Exception as e:
            print(f"Error accessing model: {e}")
            
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        self.model.load_state_dict(checkpoint['model'], strict=False)
                        if 'optimizer' in checkpoint and self.optimizer:
                            try:
                                self.optimizer.load_state_dict(checkpoint['optimizer'])
                            except Exception as e:
                                print(f"Could not load optimizer state: {e}")
                        if 'scheduler' in checkpoint and self.scheduler:
                            try:
                                self.scheduler.load_state_dict(checkpoint['scheduler'])
                            except Exception as e:
                                print(f"Could not load scheduler state: {e}")
                    else:
                        self.model.load_state_dict(checkpoint, strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                    
                print("Successfully loaded supervised model")
            except Exception as e:
                print(f"Error loading supervised model: {e}, trying fallback method...")
                try:
                    checkpoint = torch.load(self.model_path, map_location='cpu')
                    
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        self.model.load_state_dict(checkpoint['model'], strict=False)
                    else:
                        self.model.load_state_dict(checkpoint, strict=False)
                        
                    print("Successfully loaded supervised model using fallback method")
                except Exception as e2:
                    print(f"All loading methods failed: {e2}")
        
        if self.config.get('wandb.enabled', True):
            try:
                wandb.init(
                    project=self.config.get('wandb.project', 'chess_ai'),
                    name=f"reinforcement_{self.config.mode}_{time.strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "mode": self.config.mode,
                        "iters": self.iters,
                        "games_per_iter": self.games_per_iter,
                        "sims_per_move": self.sims_per_move,
                        "epochs_per_iter": self.epochs_per_iter,
                        "c_puct": self.c_puct,
                        "temp": self.temp,
                        "threads": self.threads,
                        "batch": self.batch,
                        "optimizer": self.optimizer_type,
                        "lr": self.lr,
                        "model_channels": self.config.get('model.channels'),
                        "model_blocks": self.config.get('model.blocks'),
                        "model_attention": self.config.get('model.attention'),
                        "device": self.device_type,
                        "amp": self.use_amp,
                    }
                )
                wandb.watch(self.model, log="all", log_freq=100)
            except Exception as e:
                print(f"Error initializing wandb: {e}")
        
        return True
    
    def run(self):
        if not self.setup():
            return False
        
        start_time = time.time()
        
        games_dir = os.path.join('/content/drive/MyDrive/chess_ai/data/games/self-play')
        os.makedirs(games_dir, exist_ok=True)
        
        try:
            for iteration in range(self.start_iter, self.iters + 1):
                print(f"\nIteration {iteration}/{self.iters}")
                print("=" * 40)
                
                iter_start = time.time()
                
                print("Starting self-play phase...")
                
                model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                
                num_threads = min(self.threads, os.cpu_count() or 4)
                games_per_thread, remainder = divmod(self.games_per_iter, num_threads)
                
                seeds = [
                    self.config.get('project.seed', 42) + iteration * 1000 + i 
                    for i in range(num_threads)
                ]
                
                tasks = []
                for i in range(num_threads):
                    num_games = games_per_thread + (1 if i < remainder else 0)
                    if num_games <= 0:
                        continue
                        
                    tasks.append((
                        model_state,
                        "cuda" if torch.cuda.is_available() else "cpu",
                        self.sims_per_move,
                        self.c_puct,
                        self.temp,
                        num_games,
                        seeds[i],
                        self.config.get('model.channels', 64), 
                        self.config.get('model.blocks', 4),
                        self.config.get('model.attention', True)
                    ))
                
                all_inputs, all_policies, all_values = [], [], []
                all_pgns = []
                stats = {
                    "wins": 0, "losses": 0, "draws": 0,
                    "game_lens": [], "results": []
                }
                
                with ProcessPoolExecutor(max_workers=num_threads) as executor:
                    results = list(executor.map(SelfPlay.run_games, *zip(*tasks)))
                
                for inputs, policies, values, worker_stats, pgns in results:
                    all_inputs.extend(inputs)
                    all_policies.extend(policies)
                    all_values.extend(values)
                    all_pgns.extend(pgns)
                    
                    stats['wins'] += worker_stats.get('wins', 0)
                    stats['losses'] += worker_stats.get('losses', 0)
                    stats['draws'] += worker_stats.get('draws', 0)
                    stats['game_lens'].extend(worker_stats.get('game_lens', []))
                    stats['results'].extend(worker_stats.get('results', []))
                
                stats['total_games'] = len(stats['results'])
                stats['avg_len'] = float(np.mean(stats['game_lens'])) if stats['game_lens'] else 0.0
                
                print(f"Self-play completed: {stats['total_games']} games, "
                      f"{stats['wins']} wins, {stats['losses']} losses, {stats['draws']} draws, "
                      f"avg length: {stats['avg_len']:.1f}")
                
                if wandb.run is not None:
                    wandb.log({
                        "iteration": iteration,
                        "total_games": stats['total_games'],
                        "wins": stats['wins'],
                        "losses": stats['losses'],
                        "draws": stats['draws'],
                        "avg_game_length": stats['avg_len']
                    })
                    
                    if stats['results']:
                        wandb.log({"results_hist": wandb.Histogram(stats['results'])})
                    
                    if stats['game_lens']:
                        wandb.log({"game_len_hist": wandb.Histogram(stats['game_lens'])})
                
                for idx, pgn in enumerate(all_pgns):
                    pgn_path = os.path.join(games_dir, f"game_{iteration}_{idx}.pgn")
                    with open(pgn_path, "w") as f:
                        f.write(pgn)
                
                if all_inputs:
                    print("Starting training phase...")
                    
                    input_tensor = torch.from_numpy(np.array(all_inputs, dtype=np.float32))
                    policy_tensor = torch.from_numpy(np.array(all_policies, dtype=np.float32))
                    value_tensor = torch.tensor(all_values, dtype=torch.float32)
                    
                    dataset = TensorDataset(input_tensor, policy_tensor, value_tensor)
                    
                    workers = self.config.get('hardware.workers', 2)
                    pin_memory = self.config.get('hardware.pin_memory', True)
                    prefetch = self.config.get('hardware.prefetch', 2)
                    
                    dataloader = DataLoader(
                        dataset,
                        batch_size=self.batch,
                        shuffle=True,
                        num_workers=workers,
                        pin_memory=pin_memory,
                        persistent_workers=(workers > 0),
                        prefetch_factor=prefetch if workers > 0 else None
                    )
                    
                    total_steps = self.epochs_per_iter * len(dataloader)
                    
                    if self.scheduler is None:
                        self.scheduler = get_scheduler(
                            self.optimizer, self.scheduler_type, total_steps
                        )
                    
                    best_combined_loss = float('inf')
                    
                    for epoch in range(1, self.epochs_per_iter + 1):
                        print(f"Training epoch {epoch}/{self.epochs_per_iter}")
                        
                        train_metrics = train_epoch(
                            self.model,
                            dataloader,
                            self.device_info,
                            self.optimizer,
                            self.policy_weight,
                            self.value_weight,
                            self.accum_steps,
                            self.grad_clip,
                            self.scheduler,
                            compute_accuracy=False,
                            log_interval=10
                        )
                        
                        combined_loss = (
                            self.policy_weight * train_metrics["policy_loss"] + 
                            self.value_weight * train_metrics["value_loss"]
                        )
                        
                        if wandb.run is not None:
                            wandb.log({
                                "iteration": iteration,
                                "epoch": epoch,
                                "train/policy_loss": train_metrics["policy_loss"],
                                "train/value_loss": train_metrics["value_loss"],
                                "train/combined_loss": combined_loss,
                                "learning_rate": self.scheduler.get_last_lr()[0]
                            })
                        
                        if combined_loss < best_combined_loss:
                            best_combined_loss = combined_loss
                    
                    if best_combined_loss < self.best_metric:
                        self.best_metric = best_combined_loss
                        self.best_iter = iteration
                        
                        self.ckpt.save(
                            self.model, self.optimizer, self.scheduler, 
                            iteration, tag="best"
                        )
                        
                        print(f"New best model with loss: {best_combined_loss:.4f}")
                
                if self.ckpt.interval > 0 and iteration % self.ckpt.interval == 0:
                    self.ckpt.save(
                        self.model, self.optimizer, self.scheduler, iteration
                    )
                
                iter_time = time.time() - iter_start
                print(f"Iteration {iteration} completed in {iter_time:.2f}s")
                
                if wandb.run is not None:
                    wandb.log({
                        "iteration": iteration,
                        "iteration_time": iter_time
                    })
            
            final_path = os.path.join('/content/drive/MyDrive/chess_ai/models', 'reinforcement_model.pth')
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            
            self.ckpt.save(
                self.model, self.optimizer, self.scheduler, 
                self.iters, path=final_path
            )
            
            print(f"Saved final model to: {final_path}")
            
            total_time = time.time() - start_time
            print(f"Reinforcement learning completed in {total_time:.2f}s")
            print(f"Best iteration: {self.best_iter} with metric: {self.best_metric:.4f}")
            
            if wandb.run is not None:
                wandb.run.summary.update({
                    "best_metric": self.best_metric,
                    "best_iteration": self.best_iter,
                    "total_time": total_time
                })
                wandb.finish()
            
            return True
            
        except Exception as e:
            print(f"Error during reinforcement learning: {e}")
            if wandb.run is not None:
                wandb.finish()
            return False