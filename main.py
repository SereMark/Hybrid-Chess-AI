#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch
from config import config
from game import Game, get_temp, sample_move
from mcts import MCTS
from model import BOARD_SIZE, ChessModel, MoveEncoder
from torch import nn, optim

move_encoder = MoveEncoder()


def create_batch(games: list[Game], device: str, move_encoder) -> dict[str, torch.Tensor] | None:
    all_data = []
    for game in games:
        all_data.extend(game.get_training_data())

    if not all_data:
        return None

    board_tensors = []
    target_values = []
    target_policies = []

    for data in all_data:
        board_tensors.append(data["board_tensor"])
        value = max(-1.0, min(1.0, float(data["value"])))
        target_values.append(value)
        policy_vector = torch.zeros(config.MOVE_SPACE_SIZE, dtype=torch.float32)
        for move, prob in data["move_probs"].items():
            if prob > 0:
                idx = move_encoder.encode_move(move)
                if 0 <= idx < config.MOVE_SPACE_SIZE:
                    policy_vector[idx] = max(0.0, float(prob))
        target_policies.append(policy_vector)

    return {
        "board_tensors": torch.stack(board_tensors).to(device),
        "target_values": torch.tensor(target_values, dtype=torch.float32).to(device),
        "target_policies": torch.stack(target_policies).to(device),
    }


class Buffer:
    def __init__(self, max_size: int = config.BUFFER_SIZE, device: str = "cuda"):
        self.max_size = max_size
        self.device = device
        self.size = 0
        self.pos = 0

        board_encoding_size = (
            BOARD_SIZE * BOARD_SIZE * config.ENCODING_CHANNELS
        )

        self.boards = np.zeros((max_size, board_encoding_size), dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.policies = np.zeros((max_size, config.MOVE_SPACE_SIZE), dtype=np.float32)

    def add_batch(self, games):
        if not games:
            return 0

        positions_added = 0
        for game in games:
            training_data = game.get_training_data()
            for position in training_data:
                board_tensor = position["board_tensor"]
                value = position["value"]
                move_probs = position["move_probs"]

                board_flat = board_tensor.cpu().numpy().flatten()
                self.boards[self.pos] = board_flat

                self.values[self.pos] = value

                policy_vector = np.zeros(self.policies.shape[1], dtype=np.float32)
                for move, prob in move_probs.items():
                    idx = move_encoder.encode_move(move)
                    if 0 <= idx < len(policy_vector):
                        policy_vector[idx] = prob
                self.policies[self.pos] = policy_vector

                self.pos = (self.pos + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)
                positions_added += 1

        return positions_added

    def sample(self, batch_size: int | None = None):
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        if self.size == 0:
            return None

        batch_size = min(batch_size, self.size)
        indices = np.random.default_rng().choice(self.size, batch_size, replace=False)

        return {
            "board_tensors": torch.from_numpy(self.boards[indices]).to(self.device),
            "target_values": torch.from_numpy(self.values[indices]).to(self.device),
            "target_policies": torch.from_numpy(self.policies[indices]).to(self.device),
        }


class GameManager:
    def __init__(
        self, model, move_encoder, device: str, num_parallel_games: int | None = None
    ):
        self.model = model
        self.move_encoder = move_encoder
        self.device = device
        self.num_parallel_games = num_parallel_games or config.PARALLEL_GAMES
        self.max_moves_per_game = config.MAX_MOVES_PER_GAME
        self.mcts = MCTS(model, move_encoder, device)

    def _get_active(self, games, completed_games, move_counts):
        active_indices = []
        active_boards = []

        for i, game in enumerate(games):
            if i not in completed_games:
                if game.is_game_over() or move_counts[i] >= self.max_moves_per_game:
                    completed_games.append(i)
                else:
                    active_indices.append(i)
                    active_boards.append(game.board)

        return active_indices, active_boards

    def _fallback_probs(self, board):
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return {move: 1.0 / len(legal_moves) for move in legal_moves}
        return None

    def _play_move(self, game, board, move_probs, move_counts, game_idx):
        board_tensor = self.model.encode_board(board)
        game.add_position(board_tensor, move_probs, None)

        temperature = get_temp(move_counts[game_idx])
        move = sample_move(move_probs, temperature)

        if game.history:
            game.history[-1]["move_played"] = move

        return game.make_move(move)

    def play_games(self):
        games = [Game() for _ in range(self.num_parallel_games)]
        move_counts = [0] * self.num_parallel_games
        completed_games = []

        while len(completed_games) < self.num_parallel_games:
            active_indices, active_boards = self._get_active(games, completed_games, move_counts)

            if not active_indices:
                break

            move_probs_batch = self.mcts.search_batch(active_boards)

            for idx, game_idx in enumerate(active_indices):
                game = games[game_idx]
                board = active_boards[idx]
                move_probs = move_probs_batch[idx]

                if not move_probs:
                    move_probs = self._fallback_probs(board)
                    if not move_probs:
                        completed_games.append(game_idx)
                        continue

                if self._play_move(game, board, move_probs, move_counts, game_idx):
                    move_counts[game_idx] += 1
                else:
                    completed_games.append(game_idx)

        return games


class Trainer:
    def __init__(self, model, device: str, learning_rate: float | None = None):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate or config.LEARNING_RATE
        )
        self.value_loss_fn = nn.MSELoss()
        self.games_played = 0
        self.training_steps = 0
        self.buffer = Buffer(max_size=config.BUFFER_SIZE, device=device)

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
        policy_loss = (
            -(target_policies * torch.log(predicted_policies + config.POLICY_EPSILON)).sum(dim=1).mean()
        )
        total_loss = value_loss + policy_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1
        return {
            "total_loss": float(total_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
        }

    def generate_games(self, num_games=None):
        num_games = num_games or config.GAMES_PER_ITERATION

        manager = GameManager(
            model=self.model,
            move_encoder=move_encoder,
            device=self.device,
            num_parallel_games=num_games,
        )

        games = manager.play_games()
        self.games_played += len(games)
        return games

    def train_iteration(self, num_games=None):
        games = self.generate_games(num_games)
        self.buffer.add_batch(games)

        if self.buffer.size > config.BATCH_SIZE:
            batch = self.buffer.sample()
        else:
            batch = create_batch(games, self.device, move_encoder)

        return self.train_on_batch(batch)

    def save_model(self, filepath):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "games_played": self.games_played,
            "training_steps": self.training_steps,
        }
        torch.save(checkpoint, filepath)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(
        f"Training: {config.ITERATIONS} iterations, {config.GAMES_PER_ITERATION} games each"
    )
    print(
        f"Settings: lr={config.LEARNING_RATE}, batch_size={config.BATCH_SIZE}, mcts_sims={config.MCTS_SIMULATIONS}"
    )

    save_dir = Path("./checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    model = ChessModel(device=device)
    trainer = Trainer(model, device)

    print("Starting training...")

    for iteration in range(1, config.ITERATIONS + 1):
        print(f"\nIteration {iteration}/{config.ITERATIONS}")
        loss_dict = trainer.train_iteration()

        if loss_dict:
            metrics = [f"Loss: {loss_dict['total_loss']:.4f}"]
            metrics.append(f"Value: {loss_dict['value_loss']:.4f}")
            metrics.append(f"Policy: {loss_dict['policy_loss']:.4f}")
            print(" | ".join(metrics))

        if iteration % 10 == 0:
            save_path = save_dir / f"model_iteration_{iteration}.pt"
            trainer.save_model(save_path)
            print(f"Saved: {save_path}")

    final_path = save_dir / "model_final.pt"
    trainer.save_model(final_path)
    print(f"\nTraining complete! Final model: {final_path}")


if __name__ == "__main__":
    main()
