import time
from pathlib import Path

import torch
from config import config
from model import ChessModel, MoveEncoder
from trainer import Trainer
from utils import (
    ConsoleMetricsLogger,
    get_system_status,
    monitor_gpu_memory,
    setup_logging,
)

if __name__ == "__main__":
    logger = setup_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    system_status = get_system_status()
    gpu_info = monitor_gpu_memory()

    save_dir = Path("./checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        logger.info(
            f"GPU Memory: {gpu_info['current_memory_gb']:.2f}GB used, "
            f"utilization: {gpu_info['memory_utilization']:.1f}%"
        )
    else:
        logger.warning("No GPU available, using CPU")

    trainer: Trainer | None = None
    console_logger = ConsoleMetricsLogger()
    try:
        move_encoder = MoveEncoder()
        model = ChessModel(device=device)
        trainer = Trainer(model, device, move_encoder, console_logger)

        logger.info("Starting training loop...")

        hyperparams = {
            "iterations": config.training.iterations,
            "buffer_size": config.training.buffer_size,
            "learning_rate": config.optimizer.learning_rate,
            "batch_size": config.optimizer.batch_size,
            "games_per_iteration": config.game.games_per_iteration,
            "mcts_simulations": config.mcts.simulations,
            "max_moves_per_game": config.game.max_moves_per_game,
        }
        console_logger.log_params(hyperparams)

        sample_input = torch.zeros(1, config.board_encoding_size, device=device)
        console_logger.log_graph(model, sample_input)

        start_time = time.perf_counter()

        for iteration in range(1, config.training.iterations + 1):
            iteration_start = time.perf_counter()

            progress_percent = (iteration / config.training.iterations) * 100
            elapsed_time = time.perf_counter() - start_time
            if iteration > 1:
                avg_iteration_time = elapsed_time / (iteration - 1)
                remaining_iterations = config.training.iterations - iteration
                eta_minutes = (remaining_iterations * avg_iteration_time) / 60
                eta_str = f"ETA: {eta_minutes:.1f}min"
            else:
                eta_str = "ETA: calculating..."

            logger.info(
                f"Iteration {iteration}/{config.training.iterations} ({progress_percent:.1f}%) | {eta_str}"
            )

            if iteration % config.training.gpu_check_every == 1:
                system_status = get_system_status()
                if system_status["gpu_available"]:
                    logger.info(
                        f"GPU Status: {system_status['gpu_memory_gb']:.2f}GB used"
                    )
                    console_logger.log_gpu(
                        system_status["gpu_memory_gb"],
                        system_status["gpu_utilization"],
                        iteration,
                    )

            loss_dict = trainer.train_step(iteration=iteration)

            if loss_dict:
                iteration_time = time.perf_counter() - iteration_start

                console_logger.log_iteration(
                    loss_dict,
                    {
                        "iteration_time": iteration_time,
                        "games_played": loss_dict.get("games_played", 0),
                        "buffer_size": loss_dict.get("buffer_size", 0),
                        "training_steps": loss_dict.get("training_steps", 0),
                        "avg_game_length": loss_dict.get("avg_game_length", 0),
                        "games_completed": loss_dict.get("games_completed", 0),
                        "games_timeout": loss_dict.get("games_timeout", 0),
                    },
                    iteration,
                )

                if iteration % config.training.save_every == 0:
                    console_logger.log_model(trainer.model, iteration)

            else:
                logger.warning(f"No loss computed for iteration {iteration}")

            if iteration % config.training.save_every == 0:
                save_path = save_dir / f"model_iteration_{iteration}.pt"
                trainer.save_model(save_path)

                system_status = get_system_status()
                if system_status["gpu_available"]:
                    logger.info(
                        f"Checkpoint saved | GPU: {system_status['gpu_memory_gb']:.2f}GB"
                    )

        total_training_time = time.perf_counter() - start_time
        final_path = save_dir / "model_final.pt"
        trainer.save_model(final_path)

        console_logger.log_milestone(
            "Training completed successfully",
            config.training.iterations,
            {"total_time_minutes": total_training_time / 60},
        )

        final_metrics = {}
        if trainer.losses:
            final_metrics["final_loss"] = trainer.losses[-1]
            initial_loss = (
                trainer.losses[0] if len(trainer.losses) > 1 else trainer.losses[-1]
            )
            if initial_loss != 0:
                final_metrics["loss_improvement_percent"] = (
                    (initial_loss - trainer.losses[-1]) / initial_loss
                ) * config.system.percentage_multiplier
        final_metrics["total_games_played"] = trainer.games_played
        final_metrics["total_training_time_hours"] = total_training_time / 3600

        console_logger.log_params(hyperparams, final_metrics)
        console_logger.close()

        logger.info("Training complete")
        logger.info(f"Total time: {total_training_time / 60:.1f} minutes")
        logger.info(
            f"Final: {trainer.games_played} games, {trainer.training_steps} steps"
        )

        if trainer.losses:
            final_loss = trainer.losses[-1]
            initial_loss = trainer.losses[0] if len(trainer.losses) > 1 else final_loss
            improvement = (
                ((initial_loss - final_loss) / initial_loss)
                * config.system.percentage_multiplier
                if initial_loss != 0
                else 0.0
            )
            logger.info(
                f"Loss improvement: {improvement:.1f}% ({initial_loss:.4f} → {final_loss:.4f})"
            )

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        if trainer is not None:
            backup_path = save_dir / "model_emergency.pt"
            trainer.save_model(backup_path)
            logger.info(f"Emergency checkpoint saved to {backup_path}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
