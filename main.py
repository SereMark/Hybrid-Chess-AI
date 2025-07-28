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
            elapsed = time.time() - trainer.training_start_time
            avg_time = sum(trainer.iteration_times) / len(trainer.iteration_times)
            games_per_sec = trainer.games_played / elapsed if elapsed > 0 else 0

            print(
                f"[{i:3d}/{ITERATIONS}] Iteration {i} ({i / ITERATIONS * 100:.1f}%) - {elapsed / 60:.1f}m elapsed"
            )
            print(
                f"  Loss: {metrics.get('loss', 0):.3f} (Value: {metrics.get('value_loss', 0):.3f}, Policy: {metrics.get('policy_loss', 0):.3f}) | LR: {metrics.get('learning_rate', 0):.1e} | Grad: {metrics.get('grad_norm', 0):.3f}"
            )
            print(
                f"  Games: {metrics['total']} total, {metrics['completed']} completed, {metrics.get('move_limit', 0)} hit limits | Speed: {games_per_sec:.3f}/s ({games_per_sec * 3600:.0f}/hour)"
            )
            print(
                f"  Timing: Self-play {metrics.get('self_play_time', 0):.0f}s, Training {metrics.get('train_time', 0):.1f}s | GPU: {torch.cuda.memory_allocated() / GB_BYTES if cuda_available else 0:.2f}GB | Buffer: {trainer.buffer_size / BUFFER_SIZE * 100:.0f}% | ETA: {(ITERATIONS - i) * avg_time / 3600:.1f}h\n"
            )

        if i % SAVE_EVERY == 0:
            trainer.save(save_dir / f"model_{i}.pt")

    trainer.save(save_dir / "model_final.pt")
    print(f"\nTraining complete! Games: {trainer.games_played:,}")


if __name__ == "__main__":
    main()
