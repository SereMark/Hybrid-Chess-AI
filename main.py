#!/usr/bin/env python3

from pathlib import Path

import torch
from config import get_config
from model import ChessModel
from training import Trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    iterations = get_config("training", "iterations")
    games_per_iteration = get_config("training", "games_per_iteration")
    learning_rate = get_config("training", "learning_rate")
    batch_size = get_config("training", "batch_size")
    mcts_simulations = get_config("mcts", "simulations")

    print(f"Training: {iterations} iterations, {games_per_iteration} games each")
    print(f"Settings: lr={learning_rate}, batch_size={batch_size}, mcts_sims={mcts_simulations}")

    save_dir = Path("./checkpoints")
    save_dir.mkdir(exist_ok=True)

    model = ChessModel(device=device)
    trainer = Trainer(model, device, learning_rate)

    print("Starting training...")

    for iteration in range(1, iterations + 1):
        print(f"\nIteration {iteration}/{iterations}")

        loss_dict = trainer.train_iteration(
            num_games=games_per_iteration, mcts_simulations=mcts_simulations
        )

        if loss_dict:
            metrics = [f"Loss: {loss_dict['total_loss']:.4f}"]

            if 'value_loss' in loss_dict and 'policy_loss' in loss_dict:
                metrics.append(f"Value: {loss_dict['value_loss']:.4f}")
                metrics.append(f"Policy: {loss_dict['policy_loss']:.4f}")

            current_lr = trainer.optimizer.param_groups[0]['lr']
            metrics.append(f"LR: {current_lr:.6f}")

            patience = trainer.patience_counter
            metrics.append(f"Patience: {patience}/{trainer.early_stopping_patience}")

            print(" | ".join(metrics))

            if 'gpu_memory' in loss_dict:
                gpu_mem = loss_dict['gpu_memory']
                print(f"GPU Memory: {gpu_mem['allocated']:.1f}GB allocated, {gpu_mem['free']:.1f}GB free")

            if loss_dict.get('early_stop', False):
                print("Early stopping triggered - training stopped")
                break

        if iteration % 10 == 0:
            save_path = save_dir / f"model_iteration_{iteration}.pt"
            trainer.save_model(save_path)
            print(f"Saved: {save_path}")

    final_path = save_dir / "model_final.pt"
    trainer.save_model(final_path)
    print(f"\nTraining complete! Final model: {final_path}")


if __name__ == "__main__":
    main()
