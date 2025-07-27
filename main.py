import time
from pathlib import Path

import torch
from config import BUFFER_SIZE, GB_BYTES, ITERATIONS, SAVE_EVERY
from trainer import ChessTrainer
from utils import get_system_stats


def main():
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    save_dir = Path("./checkpoints")
    save_dir.mkdir(exist_ok=True)

    trainer = ChessTrainer(device)
    print(f"Training on {device}")
    print(f"Model: {sum(p.numel() for p in trainer.model.parameters()):,} parameters")

    for i in range(1, ITERATIONS + 1):
        metrics = trainer.iteration()

        if metrics:
            sys_stats = get_system_stats()
            gpu_mem = torch.cuda.memory_allocated() / GB_BYTES if cuda_available else 0

            elapsed = time.time() - trainer.training_start_time
            avg_time = sum(trainer.iteration_times) / len(trainer.iteration_times)
            eta_min = (ITERATIONS - i) * avg_time / 60
            games_per_sec = trainer.games_played / elapsed if elapsed > 0 else 0
            completion_rate = metrics["completed"] / metrics["total"] * 100

            loss = metrics.get("loss", 0)
            value_loss = metrics.get("value_loss", 0)
            policy_loss = metrics.get("policy_loss", 0)
            lr = metrics.get("learning_rate", 0)

            print(
                f"[{i:3d}/{ITERATIONS}] Loss: {loss:.4f} (V:{value_loss:.3f} P:{policy_loss:.3f}) LR: {lr:.6f}"
            )
            print(
                f"  Games: {metrics['completed']}/{metrics['total']} ({completion_rate:.0f}%) "
                f"W:{metrics['wins']} L:{metrics['losses']} D:{metrics['draws']} Moves: {metrics['avg_moves']:.1f}"
            )
            print(
                f"  System: CPU {sys_stats['cpu_percent']:.1f}% RAM {sys_stats['ram_used_gb']:.1f}/{sys_stats['ram_total_gb']:.1f}GB "
                f"GPU {gpu_mem:.1f}GB Load {sys_stats['load_avg']:.2f}"
            )
            print(
                f"  Speed: {metrics['time']:.1f}s/iter {games_per_sec:.1f} games/s "
                f"Buffer: {metrics['buffer']}/{BUFFER_SIZE} ETA: {eta_min:.0f}m\n"
            )

        if i % SAVE_EVERY == 0:
            trainer.save(save_dir / f"model_{i}.pt")

    trainer.save(save_dir / "model_final.pt")
    print(f"\nTraining complete! Games: {trainer.games_played:,}")


if __name__ == "__main__":
    main()
