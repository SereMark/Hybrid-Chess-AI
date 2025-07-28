import time
from pathlib import Path

import torch
from config import (
    BUFFER_SIZE,
    GB_BYTES,
    ITERATIONS,
    SAVE_EVERY,
)
from trainer import ChessTrainer


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
            gpu_mem_alloc = (
                torch.cuda.memory_allocated() / GB_BYTES if cuda_available else 0
            )

            elapsed = time.time() - trainer.training_start_time
            avg_time = sum(trainer.iteration_times) / len(trainer.iteration_times)
            eta_hours = (ITERATIONS - i) * avg_time / 3600
            games_per_sec = trainer.games_played / elapsed if elapsed > 0 else 0
            games_per_hour = games_per_sec * 3600

            loss = metrics.get("loss", 0)
            value_loss = metrics.get("value_loss", 0)
            policy_loss = metrics.get("policy_loss", 0)
            lr = metrics.get("learning_rate", 0)
            grad_norm = metrics.get("grad_norm", 0)

            completed = metrics["completed"]
            total = metrics["total"]
            move_limit = metrics.get("move_limit", 0)

            self_play_time = metrics.get("self_play_time", 0)
            train_time = metrics.get("train_time", 0)

            buffer_size = trainer.buffer_size
            buffer_pct = (buffer_size / BUFFER_SIZE) * 100

            print(f"[{i:3d}/{ITERATIONS}] Iteration {i} ({i / ITERATIONS * 100:.1f}%) - {elapsed / 60:.1f}m elapsed")
            print(f"  Loss: {loss:.3f} (Value: {value_loss:.3f}, Policy: {policy_loss:.3f}) | LR: {lr:.1e} | Grad: {grad_norm:.3f}")
            print(f"  Games: {total} total, {completed} completed, {move_limit} hit limits | Speed: {games_per_sec:.3f}/s ({games_per_hour:.0f}/hour)")
            print(f"  Timing: Self-play {self_play_time:.0f}s, Training {train_time:.1f}s | GPU: {gpu_mem_alloc:.2f}GB | Buffer: {buffer_pct:.0f}% | ETA: {eta_hours:.1f}h\n")

        if i % SAVE_EVERY == 0:
            trainer.save(save_dir / f"model_{i}.pt")

    trainer.save(save_dir / "model_final.pt")
    print(f"\nTraining complete! Games: {trainer.games_played:,}")


if __name__ == "__main__":
    main()
