from typing import NoReturn, List, Dict, Any, Optional, Union
import logging
import sys
import time
import os
from pathlib import Path
import torch
import argparse
import chess
import yaml
logger = logging.getLogger(__name__)

try:
    with open("config.yaml", 'r') as f:
        _config = yaml.safe_load(f)
    CONFIG = _config.get('defaults', {})
except FileNotFoundError:
    CONFIG = {}
except Exception:
    CONFIG = {}


def get_config(section: str, key: str, preset: str = None) -> Any:
    if preset and preset in _config:
        preset_config = _config[preset]
        if section in preset_config and key in preset_config[section]:
            return preset_config[section][key]
    return CONFIG.get(section, {}).get(key)


def get_preset_config(preset: str) -> Dict[str, Any]:
    if preset not in _config:
        raise ValueError(f"Preset '{preset}' not found")
    result = CONFIG.copy()
    for section, values in _config[preset].items():
        if section in result and isinstance(values, dict):
            result[section].update(values)
        else:
            result[section] = values
    return result


def apply_preset_config(args, preset: str) -> None:
    config = get_preset_config(preset)
    training = config['training']
    system = config['system']
    args.iterations = training['iterations']
    args.games_per_iteration = training['games_per_iteration']
    args.mcts_simulations = config['mcts']['simulations']
    args.learning_rate = training['learning_rate']
    args.save_every = training['save_every']
    args.save_dir = system['checkpoint_dir']
    args.max_moves_per_game = training['max_moves_per_game']
    args.config_preset = preset


def setup_logging(preset: str = None, log_level: str = None):
    if log_level:
        level = getattr(logging, log_level.upper())
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().setLevel(level)


def get_mcts_simulations(iteration: int, total_iterations: int, base_sims: int, preset: str) -> int:
    schedule = get_config('mcts', 'schedule', preset)
    if not schedule:
        return base_sims
    progress = iteration / total_iterations
    ramp_threshold = schedule.get('ramp_iterations', 500) / total_iterations
    if progress <= ramp_threshold:
        start_sims = schedule.get('start_simulations', base_sims)
        end_sims = schedule.get('end_simulations', base_sims)
        ramp_progress = progress / ramp_threshold
        return int(start_sims + (end_sims - start_sims) * ramp_progress)
    return base_sims


def main() -> NoReturn:
    from model import ChessModel, MoveEncoder
    from mcts import MCTS
    from training import Trainer
    from game import play_self_play_game
    defaults = CONFIG
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--preset', choices=['debug', 'fast', 'optimal', 'champion'])
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default=defaults['system']['device'])
    parser.add_argument('--iterations', type=int, default=defaults['training']['iterations'])
    parser.add_argument('--games-per-iteration', type=int, default=defaults['training']['games_per_iteration'])
    parser.add_argument('--mcts-simulations', type=int, default=defaults['mcts']['simulations'])
    parser.add_argument('--learning-rate', type=float, default=defaults['training']['learning_rate'])
    parser.add_argument('--save-every', type=int, default=defaults['training']['save_every'])
    parser.add_argument('--load-model', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=defaults['system']['checkpoint_dir'])
    args = parser.parse_args()

    if args.preset:
        apply_preset_config(args, args.preset)
        setup_logging(args.preset, args.log_level)
        system_config = get_preset_config(args.preset)['system']
        if 'expected_time' in system_config:
            logger.info(f"Preset: {args.preset} ({system_config['expected_time']})")
        if 'target_strength' in system_config:
            logger.info(f"Target: {system_config['target_strength']}")
    else:
        setup_logging(log_level=args.log_level)
        args.config_preset = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        logger.debug(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    else:
        logger.info("Using CPU (CUDA not available)")
        logger.warning("Training will be slower without GPU")

    try:
        if args.iterations <= 0:
            raise ValueError("Iterations must be positive")
        if args.games_per_iteration <= 0:
            raise ValueError("Games per iteration must be positive")
        if args.mcts_simulations <= 0:
            raise ValueError("MCTS simulations must be positive")
        if args.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        logger.info("Chess AI Training")
        logger.info("Configuration:")
        logger.info(f"  Device: {args.device}")
        logger.info(f"  Iterations: {args.iterations}")
        logger.info(f"  Games per iteration: {args.games_per_iteration}")
        logger.info(f"  MCTS simulations: {args.mcts_simulations}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Save directory: {args.save_dir}")
        logger.debug(f"Args: {vars(args)}")
        logger.info("-" * 60)

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Save directory created: {save_dir.absolute()}")

        logger.debug("Initializing model...")
        model_start = time.time()
        model = ChessModel(device=args.device)
        model_init_time = time.time() - model_start
        logger.debug(f"Model init: {model_init_time:.2f}s")

        logger.debug("Initializing trainer...")
        trainer_start = time.time()
        trainer = Trainer(model, model.device, args.learning_rate)
        trainer_init_time = time.time() - trainer_start
        logger.debug(f"Trainer init: {trainer_init_time:.2f}s")

        if args.load_model:
            if os.path.exists(args.load_model):
                load_start = time.time()
                trainer.load_model(args.load_model)
                load_time = time.time() - load_start
                logger.info(f"Resumed training from {args.load_model} ({load_time:.2f}s)")
                logger.info(f"Previous progress: {trainer.games_played} games, {trainer.training_steps} steps")
                logger.debug(f"Checkpoint file size: {os.path.getsize(args.load_model) / 1024**2:.1f}MB")
            else:
                logger.warning(f"Model file {args.load_model} not found, starting fresh")

        logger.info(f"Starting {args.iterations} iterations...")
        total_games_target = args.iterations * args.games_per_iteration
        logger.info(f"Target: {total_games_target} total games")
        logger.debug(f"Estimated total MCTS simulations: {total_games_target * args.mcts_simulations * 50:,}")

        if args.preset:
            system_config = get_preset_config(args.preset)['system']
            if 'memory_usage' in system_config:
                logger.info(f"Memory: {system_config['memory_usage']}")
            if 'batch_size_hint' in system_config:
                logger.debug(f"Batch size hint: {system_config['batch_size_hint']}")

        for iteration in range(args.iterations):
            iteration_num = iteration + 1
            if iteration_num % 10 == 1 or iteration_num <= 3:
                logger.info(f"\nITERATION {iteration_num}/{args.iterations}")
                progress_pct = (iteration_num / args.iterations) * 100
                logger.debug(f"Progress: {progress_pct:.1f}% complete")
            start_time = time.time()

            try:
                logger.debug(f"Starting iteration {iteration_num}...")
                mcts_sims = args.mcts_simulations
                if args.preset:
                    mcts_sims = get_mcts_simulations(
                        iteration_num, args.iterations, args.mcts_simulations, args.preset
                    )
                loss_dict = trainer.train_iteration(
                    num_games=args.games_per_iteration,
                    mcts_simulations=mcts_sims
                )
                iteration_time = time.time() - start_time

                if iteration_time > 20.0:
                    logger.warning(f"Iteration time: {iteration_time:.1f}s (target <10s)")
                elif iteration_time > 10.0:
                    logger.debug(f"Iteration time: {iteration_time:.1f}s")

                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    memory_pct = (memory_used / memory_total) * 100
                    logger.debug(f"GPU memory: {memory_used:.1f}GB/{memory_total:.1f}GB ({memory_pct:.1f}%)")
                    if memory_pct > 80:
                        logger.warning(f"GPU memory usage: {memory_pct:.1f}%")

                if iteration_num <= 3 or iteration_num % 10 == 0:
                    logger.info(f"Iteration {iteration_num}: {iteration_time:.1f}s, {trainer.get_stats()}")
                    games_per_sec = args.games_per_iteration / iteration_time
                    logger.info(f"Performance: {games_per_sec:.1f} games/sec")
                    if games_per_sec < 0.1:
                        logger.warning(f"Slow game generation: {games_per_sec:.2f} games/sec")
                    remaining_iterations = args.iterations - iteration_num
                    estimated_time_remaining = remaining_iterations * iteration_time
                    eta_hours = estimated_time_remaining / 3600
                    logger.debug(f"ETA: {eta_hours:.1f} hours remaining")

                if iteration_num % args.save_every == 0:
                    save_filename = get_config('system', 'model_iteration_filename', args.config_preset).format(iteration_num)
                    save_path = save_dir / save_filename
                    save_start = time.time()
                    try:
                        trainer.save_model(save_path)
                        save_time = time.time() - save_start
                        file_size = os.path.getsize(save_path) / 1024**2
                        logger.info(f"Saved: {save_path} ({file_size:.1f}MB, {save_time:.2f}s)")
                        logger.debug(f"Checkpoint includes {len(trainer.losses)} loss records")
                    except Exception as e:
                        logger.warning(f"Checkpoint save failed: {e}")
                        try:
                            import shutil
                            free_space = shutil.disk_usage(save_dir).free / 1024**3
                            logger.debug(f"Available disk space: {free_space:.1f}GB")
                            if free_space < 1.0:
                                logger.warning(f"Disk space: {free_space:.1f}GB remaining")
                        except:
                            pass

                if iteration_num % 25 == 0:
                    recent_losses = trainer.losses[-10:] if len(trainer.losses) >= 10 else trainer.losses
                    if recent_losses:
                        avg_loss = sum(l['total_loss'] for l in recent_losses) / len(recent_losses)
                        logger.info(f"Progress: {iteration_num} iterations, loss: {avg_loss:.4f}")
                        if len(trainer.losses) >= 20:
                            older_losses = trainer.losses[-20:-10]
                            older_avg = sum(l['total_loss'] for l in older_losses) / len(older_losses)
                            trend = avg_loss - older_avg
                            if trend > 0.5:
                                logger.warning(f"Loss trend: +{trend:.3f} over last 10 iterations")
                            elif trend < -0.1:
                                logger.debug(f"Loss trend: {trend:.3f} over last 10 iterations")

            except Exception as e:
                logger.error(f"Iteration {iteration_num} failed: {e}")
                logger.debug(f"Exception type: {type(e).__name__}")
                if "CUDA out of memory" in str(e):
                    if torch.cuda.is_available():
                        memory_info = torch.cuda.memory_summary()
                        logger.debug(f"GPU memory summary:\n{memory_info}")
                    logger.critical("GPU memory exhausted, stopping")
                    break
                elif "iteration_time" in locals() and iteration_time > 30.0:
                    logger.error(f"Iteration timeout ({iteration_time:.1f}s), skipping")
                    logger.warning("Consider reducing parameters")
                logger.debug("Continuing with next iteration...")
                continue

        total_games = trainer.games_played
        total_steps = trainer.training_steps
        final_loss = trainer.losses[-1]['total_loss'] if trainer.losses else 0.0
        logger.info(f"Complete: {total_games} games, {total_steps} steps")
        logger.info(f"Final loss: {final_loss:.4f}")
        logger.debug(f"Average games per iteration: {total_games / args.iterations:.1f}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        sys.exit(1)

    finally:
        try:
            logger.debug("Saving final...")
            final_save_path = save_dir / get_config('system', 'model_filename')
            save_start = time.time()
            trainer.save_model(final_save_path)
            save_time = time.time() - save_start
            file_size = os.path.getsize(final_save_path) / 1024**2
            logger.info(f"Final saved: {final_save_path} ({file_size:.1f}MB, {save_time:.2f}s)")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
            logger.debug(f"Save error type: {type(e).__name__}")

    logger.info("Training completed")
    logger.debug("Cleaning up...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")
    sys.exit(0)


if __name__ == "__main__":
    main()
