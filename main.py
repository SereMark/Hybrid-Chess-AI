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
            gpu_mem_alloc = (
                torch.cuda.memory_allocated() / GB_BYTES if cuda_available else 0
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
            time_trend = (
                (trainer.iteration_times[-1] - trainer.iteration_times[-2])
                if len(trainer.iteration_times) >= 2
                else 0
            )
            loss_ratio = value_loss / policy_loss if policy_loss > 0 else 0
            ram_usage_pct = sys_stats["ram_used_gb"] / sys_stats["ram_total_gb"] * 100

            print(f"{'=' * 100}")
            print(
                f"[{i:3d}/{ITERATIONS}] ITERATION {i} - {elapsed / 3600:.1f}h elapsed - Progress: {i / ITERATIONS * 100:.1f}%"
            )
            print(f"{'=' * 100}")

            print("TRAINING:")
            print(
                f"  Loss: {loss:.6f} (Value: {value_loss:.4f} | Policy: {policy_loss:.4f} | Ratio: {loss_ratio:.2f})"
            )
            print(
                f"  Optimization: Grad {grad_norm:.4f} | LR {lr:.2e} | Step {'✓' if optimizer_stepped else '○'}"
            )
            print(
                f"  Batch: {actual_batch_size}/{BATCH_SIZE} | Accumulation: {detailed_state['accumulation_progress_pct']:.0f}% ({metrics.get('accumulation_step', 0)}/{GRADIENT_ACCUMULATION})"
            )

            print("GAMES:")
            print(
                f"  Completion: {metrics['completed']}/{metrics['total']} ({completion_rate:.1f}%) | Total: {detailed_state['games_played']:,}"
            )
            print(
                f"  Outcomes: W{metrics['wins']} L{metrics['losses']} D{metrics['draws']} | Resigned: {metrics.get('resigned', 0)} | Limits: {metrics.get('move_limit', 0)}"
            )
            print(
                f"  Moves: {metrics['avg_moves']:.1f} avg | {metrics.get('min_moves', 0)}-{metrics.get('max_moves', 0)} range | Temp: {metrics.get('temp_transitions', 0)}"
            )

            print("PERFORMANCE:")
            print(
                f"  Timing: {current_time:.1f}s (Δ{time_trend:+.1f}s) | Avg: {avg_time:.1f}s | Trend: {detailed_state['time_trend_per_iter']:+.2f}s/iter"
            )
            print(
                f"  Breakdown: Self-play {metrics.get('self_play_time', 0):.1f}s ({metrics.get('self_play_pct', 0):.0f}%) | Training {metrics.get('train_time', 0):.1f}s ({metrics.get('train_pct', 0):.0f}%)"
            )
            print(
                f"  Speed: {games_per_sec:.2f} games/s | {trainer.games_played / max(elapsed, 0.001) * 3600:.0f}/hour | ETA: {eta_hours:.1f}h"
            )

            print("MCTS:")
            print(
                f"  Search: {detailed_state.get('mcts_searches_performed', 0)} runs | {detailed_state.get('mcts_total_simulations', 0)} sims"
            )
            print(
                f"  Nodes: {detailed_state.get('mcts_nodes_expanded', 0):,} expanded | Terminal: {detailed_state.get('mcts_terminal_hit_rate', 0):.1f}%"
            )
            print(f"  Model calls: {detailed_state.get('mcts_model_forward_calls', 0)}")

            print("MODEL:")
            print(
                f"  Forward: {detailed_state.get('model_forward_calls', 0)} passes | Cache: {detailed_state.get('model_cache_hit_rate', 0):.1f}% hit rate"
            )
            print(
                f"  Cache: {detailed_state.get('model_cache_utilization', 0):.1f}% utilization"
            )

            print("MEMORY:")
            print(
                f"  Buffer: {detailed_state['buffer_size']:,}/{BUFFER_SIZE:,} ({detailed_state['buffer_usage_pct']:.1f}%) | Pos: {detailed_state['buffer_position']:,}"
            )
            print(
                f"  Cache: {detailed_state['cache_size']:,}/{CACHE_SIZE:,} ({detailed_state['cache_usage_pct']:.1f}%)"
            )
            print(
                f"  System: CPU {sys_stats['cpu_percent']:.1f}% | RAM {sys_stats['ram_used_gb']:.1f}/{sys_stats['ram_total_gb']:.1f}GB ({ram_usage_pct:.1f}%)"
            )
            print(
                f"  GPU: {gpu_mem_alloc:.2f}GB allocated | {gpu_mem_reserved:.2f}GB reserved"
            )

            print("HEALTH:")
            completion_status = "✓ Natural" if completion_rate > 0 else "⚠ Forced"
            if loss < 2.0:
                convergence_status = "✓ Excellent"
            elif loss < 3.0:
                convergence_status = "✓ Good"
            elif loss < 4.0:
                convergence_status = "⚠ Moderate"
            else:
                convergence_status = "○ Learning"

            if grad_norm < 1.0:
                gradient_status = "✓ Stable"
            elif grad_norm < 5.0:
                gradient_status = "⚠ Moderate"
            else:
                gradient_status = "○ High"

            print(
                f"  Completion: {completion_status} ({completion_rate:.1f}%) | Convergence: {convergence_status} ({loss:.3f}) | Gradients: {gradient_status} ({grad_norm:.3f})"
            )

            print(f"{'=' * 100}\n")

        if i % SAVE_EVERY == 0:
            trainer.save(save_dir / f"model_{i}.pt")

    trainer.save(save_dir / "model_final.pt")
    print(f"\nTraining complete! Games: {trainer.games_played:,}")


if __name__ == "__main__":
    main()
