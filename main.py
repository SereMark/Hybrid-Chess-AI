import time
from pathlib import Path

import torch
from config import (
    BATCH_SIZE,
    BUFFER_SIZE,
    CACHE_SIZE,
    GB_BYTES,
    GRADIENT_ACCUMULATION,
    ITERATIONS,
    SAVE_EVERY,
)
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
            gpu_mem_cached = (
                torch.cuda.memory_cached() / GB_BYTES if cuda_available else 0
            )
            gpu_mem_reserved = (
                torch.cuda.memory_reserved() / GB_BYTES if cuda_available else 0
            )
            detailed_state = trainer.get_detailed_state()

            elapsed = time.time() - trainer.training_start_time
            avg_time = sum(trainer.iteration_times) / len(trainer.iteration_times)
            eta_min = (ITERATIONS - i) * avg_time / 60
            eta_hours = eta_min / 60
            games_per_sec = trainer.games_played / elapsed if elapsed > 0 else 0
            completion_rate = metrics["completed"] / metrics["total"] * 100

            loss = metrics.get("loss", 0)
            value_loss = metrics.get("value_loss", 0)
            policy_loss = metrics.get("policy_loss", 0)
            lr = metrics.get("learning_rate", 0)
            grad_norm = metrics.get("grad_norm", 0)
            actual_batch_size = metrics.get("actual_batch_size", 0)
            optimizer_stepped = metrics.get("optimizer_stepped", False)

            current_time = trainer.iteration_times[-1] if trainer.iteration_times else 0
            if len(trainer.iteration_times) >= 2:
                time_trend = trainer.iteration_times[-1] - trainer.iteration_times[-2]
            else:
                time_trend = 0

            loss_ratio = value_loss / policy_loss if policy_loss > 0 else 0

            print(f"{'=' * 90}")
            print(
                f"[{i:3d}/{ITERATIONS}] ITERATION {i} - {elapsed / 3600:.1f}h elapsed - Progress: {i / ITERATIONS * 100:.1f}%"
            )
            print(f"{'=' * 90}")

            print("TRAINING METRICS:")
            print(
                f"   Loss: {loss:.6f} (V:{value_loss:.4f} P:{policy_loss:.4f}) V/P Ratio: {loss_ratio:.2f}"
            )
            print(f"   Grad Norm: {grad_norm:.4f} | LR: {lr:.8f}")
            print(
                f"   Batch: {actual_batch_size}/{BATCH_SIZE} | Accumulation: {detailed_state['accumulation_progress_pct']:.0f}% ({metrics.get('accumulation_step', 0)}/{GRADIENT_ACCUMULATION})"
            )
            print(f"   Optimizer Step: {'OK' if optimizer_stepped else 'PENDING'}")

            print("GAME ANALYSIS:")
            print(
                f"   Completed: {metrics['completed']}/{metrics['total']} ({completion_rate:.1f}%)"
            )
            print(
                f"   Results: W:{metrics['wins']} L:{metrics['losses']} D:{metrics['draws']} | Resigned: {metrics.get('resigned', 0)} | Hit Limit: {metrics.get('move_limit', 0)}"
            )
            print(
                f"   Moves: Avg:{metrics['avg_moves']:.1f} Min:{metrics.get('min_moves', 0)} Max:{metrics.get('max_moves', 0)}"
            )
            print(
                f"   Temp Transitions: {metrics.get('temp_transitions', 0)}/{metrics['total']} | Total Games: {detailed_state['games_played']:,}"
            )

            print("BUFFER & CACHE:")
            print(
                f"   Buffer: {detailed_state['buffer_size']:,}/{BUFFER_SIZE:,} ({detailed_state['buffer_usage_pct']:.1f}%) | Pos: {detailed_state['buffer_position']:,}"
            )
            print(
                f"   Cache: {detailed_state['cache_size']:,}/{CACHE_SIZE:,} ({detailed_state['cache_usage_pct']:.1f}%)"
            )

            print("PERFORMANCE:")
            print(
                f"   Time: {current_time:.1f}s (Δ{time_trend:+.1f}s) | Avg: {avg_time:.1f}s"
            )
            print(
                f"   Speed: {games_per_sec:.2f} games/s | {trainer.games_played / elapsed * 3600:.0f} games/hour"
            )
            print(
                f"   ETA: {eta_hours:.1f}h ({eta_min:.0f}m) | Iterations: {detailed_state['total_iterations']}"
            )

            print("SYSTEM STATUS:")
            print(
                f"   CPU: {sys_stats['cpu_percent']:.1f}% | Load: {sys_stats['load_avg']:.2f}"
            )
            print(
                f"   RAM: {sys_stats['ram_used_gb']:.1f}/{sys_stats['ram_total_gb']:.1f}GB ({sys_stats['ram_used_gb'] / sys_stats['ram_total_gb'] * 100:.1f}%)"
            )
            print(
                f"   GPU Memory: {gpu_mem:.2f}GB alloc | {gpu_mem_cached:.2f}GB cached | {gpu_mem_reserved:.2f}GB reserved"
            )

            print("TRAINING HEALTH:")
            if completion_rate > 0:
                print(f"   Game Completion: OK {completion_rate:.1f}% natural endings")
            else:
                print(
                    "   Game Completion: WARNING  No natural endings (all hit limits/resigned)"
                )

            if loss < 2.0:
                print(f"   Convergence: OK Excellent ({loss:.3f})")
            elif loss < 3.0:
                print(f"   Convergence: OK Good ({loss:.3f})")
            elif loss < 4.0:
                print(f"   Convergence: WARNING  Moderate ({loss:.3f})")
            else:
                print(f"   Convergence: HIGH Learning phase ({loss:.3f})")

            if grad_norm < 1.0:
                print(f"   Gradients: OK Stable ({grad_norm:.3f})")
            elif grad_norm < 5.0:
                print(f"   Gradients: WARNING  Moderate ({grad_norm:.3f})")
            else:
                print(f"   Gradients: HIGH High ({grad_norm:.3f})")

            print(f"{'=' * 90}\n")

        if i % SAVE_EVERY == 0:
            trainer.save(save_dir / f"model_{i}.pt")

    trainer.save(save_dir / "model_final.pt")
    print(f"\nTraining complete! Games: {trainer.games_played:,}")


if __name__ == "__main__":
    main()
