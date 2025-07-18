import torch
from config import get_config
from experience_buffer import ExperienceBuffer
from game import create_training_batch
from move_encoder import MoveEncoder
from parallel_game_manager import ParallelGameManager
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(self, model, device: str, learning_rate: float = 0.001):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.move_encoder = MoveEncoder()

        self.value_loss_fn = nn.MSELoss()

        self.losses = []
        self.games_played = 0
        self.training_steps = 0

        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 20
        self.min_improvement = 0.001

        self.experience_buffer = ExperienceBuffer(max_size=50000, device=device)

    def _get_gpu_memory_info(self):
        if self.device == "cuda" and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                "allocated": memory_allocated,
                "reserved": memory_reserved,
                "total": memory_total,
                "free": memory_total - memory_reserved
            }
        return None

    def _check_early_stopping(self, current_loss):
        if current_loss < self.best_loss - self.min_improvement:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stopping_patience

    def train_on_batch(self, batch):
        if batch is None:
            return None

        board_tensors = batch["board_tensors"]
        target_values = batch["target_values"].unsqueeze(1)
        target_policies = batch["target_policies"]

        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(board_tensors)
        predicted_values = outputs["value"]
        predicted_policies = outputs["policy"]

        value_loss = self.value_loss_fn(predicted_values, target_values)
        log_policies = torch.log(predicted_policies + 1e-8)
        policy_loss = -(target_policies * log_policies).sum(dim=1).mean()
        total_loss = value_loss + policy_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
        }
        self.losses.append(loss_dict)
        self.training_steps += 1

        return loss_dict

    def generate_games(self, num_games=3, mcts_simulations=25):
        parallel_manager = ParallelGameManager(
            model=self.model,
            move_encoder=self.move_encoder,
            device=self.device,
            num_parallel_games=num_games,
            batch_size=16,
            num_simulations=mcts_simulations,
        )

        games = parallel_manager.play_games()
        self.games_played += len(games)
        return games

    def train_iteration(self, num_games=None, mcts_simulations=None):
        num_games = num_games or get_config("training", "games_per_iteration") or 3
        mcts_simulations = mcts_simulations or get_config("mcts", "simulations") or 25

        games = self.generate_games(num_games, mcts_simulations)

        self.experience_buffer.add_batch(games)

        if self.experience_buffer.size > 0:
            batch = self.experience_buffer.sample()
            loss_dict = self.train_on_batch(batch)
            if loss_dict:
                self.scheduler.step(loss_dict['total_loss'])
                should_stop = self._check_early_stopping(loss_dict['total_loss'])
                loss_dict['early_stop'] = should_stop
                gpu_memory = self._get_gpu_memory_info()
                if gpu_memory:
                    loss_dict['gpu_memory'] = gpu_memory
            return loss_dict
        else:
            batch = create_training_batch(games, self.device)
            loss_dict = self.train_on_batch(batch)
            if loss_dict:
                self.scheduler.step(loss_dict['total_loss'])
                should_stop = self._check_early_stopping(loss_dict['total_loss'])
                loss_dict['early_stop'] = should_stop
                gpu_memory = self._get_gpu_memory_info()
                if gpu_memory:
                    loss_dict['gpu_memory'] = gpu_memory
            return loss_dict

    def save_model(self, filepath):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "games_played": self.games_played,
            "training_steps": self.training_steps,
            "losses": self.losses,
        }
        torch.save(checkpoint, filepath)
