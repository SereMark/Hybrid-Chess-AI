from typing import List, Dict, Optional, Any
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from main import get_config
from model import ChessModel, MoveEncoder
from mcts import MCTS
from game import Game, play_self_play_game, create_training_batch
from parallel_game_manager import ParallelGameManager
from experience_buffer import ExperienceBuffer

logger = logging.getLogger(__name__)
NUMERICAL_STABILITY_EPSILON = 1e-8
RECENT_LOSSES_WINDOW = 10


class Trainer:
    def __init__(self, 
                 model: ChessModel, 
                 device: str, 
                 learning_rate: float = None,
                 use_experience_replay: bool = True,
                 buffer_size: int = None,
                 use_mixed_precision: bool = True) -> None:
        if learning_rate is None:
            learning_rate = get_config('training', 'learning_rate')
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.move_encoder = MoveEncoder()
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        self.losses: List[Dict[str, float]] = []
        self.games_played: int = 0
        self.training_steps: int = 0
        self.total_positions_trained = 0
        self.failed_games = 0
        
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        if self.use_mixed_precision:
            try:
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("Mixed precision training enabled (FP16 with Tensor Cores)")
            except Exception as e:
                logger.warning(f"Mixed precision initialization failed: {e}")
                self.use_mixed_precision = False
                self.scaler = None
        else:
            self.scaler = None
        
        self.use_experience_replay = use_experience_replay
        if self.use_experience_replay:
            if buffer_size is None:
                buffer_size = 50000
                exp_replay_config = get_config('training', 'experience_replay')
                if exp_replay_config and isinstance(exp_replay_config, dict):
                    buffer_size = exp_replay_config.get('buffer_size', 50000)
                    if not exp_replay_config.get('enabled', True):
                        self.use_experience_replay = False
                        self.experience_buffer = None
                        logger.info("Experience replay disabled by configuration")
                        return
            self.experience_buffer = ExperienceBuffer(max_size=buffer_size, device=device)
            logger.info(f"Experience replay enabled with buffer size {buffer_size}")
        else:
            self.experience_buffer = None
            
        logger.info(f"Trainer ready: lr={learning_rate} on {device}")
        logger.debug(f"Optimizer: {type(self.optimizer).__name__}")
        logger.debug(f"Loss functions: {type(self.policy_loss_fn).__name__}, {type(self.value_loss_fn).__name__}")

    def train_on_batch(self, batch: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, float]]:
        if batch is None:
            logger.warning("Cannot train on None batch")
            return None
        try:
            if logger.isEnabledFor(logging.DEBUG):
                train_start = time.time()
            board_tensors = batch['board_tensors']
            target_values = batch['target_values'].unsqueeze(1)
            target_policies = batch['target_policies']
            batch_size = board_tensors.size(0)
            self.total_positions_trained += batch_size
            logger.debug(f"Training batch: {batch_size} positions")
            logger.debug(f"Batch shapes - boards: {board_tensors.shape}, values: {target_values.shape}, policies: {target_policies.shape}")
            self.model.train()
            
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    if logger.isEnabledFor(logging.DEBUG):
                        forward_start = time.time()
                    outputs = self.model(board_tensors)
                    if logger.isEnabledFor(logging.DEBUG):
                        forward_time = time.time() - forward_start
                        logger.debug(f"Forward pass (FP16): {forward_time * 1000:.2f}ms")
                    predicted_values = outputs['value']
                    predicted_policies = outputs['policy']
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        loss_start = time.time()
                    value_loss = self.value_loss_fn(predicted_values, target_values)
                    log_policies = torch.log(predicted_policies + NUMERICAL_STABILITY_EPSILON)
                    policy_loss = -(target_policies * log_policies).sum(dim=1).mean()
                    
                total_loss = value_loss.float() + policy_loss.float()
                
                if logger.isEnabledFor(logging.DEBUG):
                    loss_time = time.time() - loss_start
                    logger.debug(f"Loss computation (mixed precision): {loss_time * 1000:.2f}ms")
                    logger.debug(f"Loss components - value: {value_loss.item():.4f}, policy: {policy_loss.item():.4f}")
                
                if logger.isEnabledFor(logging.DEBUG):
                    backward_start = time.time()
                self.scaler.scale(total_loss).backward()
                
                self.scaler.unscale_(self.optimizer)
                
                grad_clip_norm = get_config('training', 'gradient_clip_max_norm')
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Gradient norm (unscaled): {grad_norm:.4f}")
                    if grad_norm >= grad_clip_norm:
                        logger.debug("Gradients were clipped")
                    if grad_norm > 5.0:
                        logger.warning(f"Large gradient norm: {grad_norm:.4f}")
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if logger.isEnabledFor(logging.DEBUG):
                    backward_time = time.time() - backward_start
                    logger.debug(f"Backward pass + optimization (mixed precision): {backward_time * 1000:.2f}ms")
                    
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    forward_start = time.time()
                outputs = self.model(board_tensors)
                if logger.isEnabledFor(logging.DEBUG):
                    forward_time = time.time() - forward_start
                    logger.debug(f"Forward pass: {forward_time * 1000:.2f}ms")
                predicted_values = outputs['value']
                predicted_policies = outputs['policy']
                if logger.isEnabledFor(logging.DEBUG):
                    loss_start = time.time()
                value_loss = self.value_loss_fn(predicted_values, target_values)
                log_policies = torch.log(predicted_policies + NUMERICAL_STABILITY_EPSILON)
                policy_loss = -(target_policies * log_policies).sum(dim=1).mean()
                total_loss = value_loss + policy_loss
                if logger.isEnabledFor(logging.DEBUG):
                    loss_time = time.time() - loss_start
                    logger.debug(f"Loss computation: {loss_time * 1000:.2f}ms")
                    logger.debug(f"Loss components - value: {value_loss.item():.4f}, policy: {policy_loss.item():.4f}")
                if logger.isEnabledFor(logging.DEBUG):
                    backward_start = time.time()
                total_loss.backward()
                if logger.isEnabledFor(logging.DEBUG):
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                    logger.debug(f"Gradient norm before clipping: {grad_norm_before:.4f}")
                    if grad_norm_before > 5.0:
                        logger.warning(f"Large gradient norm: {grad_norm_before:.4f} (may indicate gradient explosion)")
                grad_clip_norm = get_config('training', 'gradient_clip_max_norm')
                grad_norm_after = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Gradient norm after clipping: {grad_norm_after:.4f}")
                    if grad_norm_after >= grad_clip_norm:
                        logger.debug("Gradients were clipped")
                self.optimizer.step()
                if logger.isEnabledFor(logging.DEBUG):
                    backward_time = time.time() - backward_start
                    logger.debug(f"Backward pass + optimization: {backward_time * 1000:.2f}ms")
                    
            if total_loss.item() > 10.0:
                logger.warning(f"High total loss detected: {total_loss.item():.4f}")
                if value_loss.item() > 5.0:
                    logger.warning(f"High value loss component: {value_loss.item():.4f}")
                if policy_loss.item() > 5.0:
                    logger.warning(f"High policy loss component: {policy_loss.item():.4f}")
                    
            loss_dict = {
                'total_loss': total_loss.item(),
                'value_loss': value_loss.item(),
                'policy_loss': policy_loss.item()
            }
            self.losses.append(loss_dict)
            self.training_steps += 1
            if logger.isEnabledFor(logging.DEBUG):
                total_train_time = time.time() - train_start
                throughput = batch_size / total_train_time
                logger.debug(f"Total training step: {total_train_time * 1000:.2f}ms ({throughput:.1f} positions/sec)")
            return loss_dict
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
            if "CUDA out of memory" in str(e):
                logger.warning(f"GPU memory exhausted during training (batch size: {batch_size if 'batch_size' in locals() else 'unknown'})")
            raise RuntimeError(f"Training failed at step {self.training_steps}") from e

    def get_progressive_simulations(self, base_simulations: int) -> int:
        if self.games_played < 1000:
            return max(25, base_simulations // 4)
        elif self.games_played < 5000:
            return max(50, base_simulations // 2)
        elif self.games_played < 20000:
            return max(100, int(base_simulations * 0.75))
        else:
            return base_simulations
    
    def generate_games(self, 
                       num_games: int = None, 
                       mcts_simulations: int = None,
                       use_parallel: bool = True,
                       use_progressive_simulations: bool = True) -> List[Game]:
        if num_games is None:
            num_games = get_config('training', 'games_per_iteration')
        if mcts_simulations is None:
            mcts_simulations = get_config('mcts', 'simulations')
        if num_games <= 0:
            raise ValueError(f"num_games must be positive, got {num_games}")
        if mcts_simulations <= 0:
            raise ValueError(f"mcts_simulations must be positive, got {mcts_simulations}")
            
        if use_progressive_simulations:
            original_simulations = mcts_simulations
            mcts_simulations = self.get_progressive_simulations(mcts_simulations)
            if mcts_simulations != original_simulations:
                logger.info(f"Progressive MCTS: {mcts_simulations} simulations "
                            f"(scaled from {original_simulations}, {self.games_played} games played)")
                
        if logger.isEnabledFor(logging.DEBUG):
            gen_start = time.time()
            mode = "parallel" if use_parallel else "sequential"
            logger.debug(f"Generating {num_games} games ({mode} mode) with {mcts_simulations} MCTS simulations each")
        
        if use_parallel:
            batch_size = get_config('batch', 'mcts_batch_size') or 16
            parallel_games = get_config('batch', 'parallel_games') or 8
            
            if num_games != parallel_games:
                logger.debug(f"Adjusting parallel games from {parallel_games} to {num_games}")
            
            parallel_manager = ParallelGameManager(
                model=self.model,
                move_encoder=self.move_encoder,
                device=self.device,
                num_parallel_games=num_games,
                batch_size=batch_size,
                num_simulations=mcts_simulations
            )
            
            games = parallel_manager.play_games()
            
            self.games_played += len(games)
            
            total_moves = sum(len(game.moves_played) for game in games)
            avg_moves = total_moves / len(games) if games else 0
            
            game_outcomes = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, 'timeout': 0}
            for game in games:
                if game.is_game_over():
                    result = game.board.result()
                    game_outcomes[result] = game_outcomes.get(result, 0) + 1
                else:
                    game_outcomes['timeout'] += 1
                    
            if logger.isEnabledFor(logging.DEBUG):
                gen_time = time.time() - gen_start
                games_per_sec = len(games) / gen_time if gen_time > 0 else 0
                logger.debug(f"Parallel game generation completed in {gen_time:.2f}s ({games_per_sec:.2f} games/sec)")
                logger.debug(f"Average game length: {avg_moves:.1f} moves")
                logger.debug(f"Game outcomes: {game_outcomes}")
                
            logger.info(f"Generated {len(games)} games in parallel (avg {avg_moves:.1f} moves)")
            return games
            
        else:
            mcts = MCTS(self.model, self.move_encoder, self.device, mcts_simulations)
            games = []
            completed = 0
            failed = 0
            total_moves = 0
            game_outcomes = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, 'timeout': 0}
            for i in range(num_games):
                try:
                    if logger.isEnabledFor(logging.DEBUG) and (i + 1) % max(1, num_games // 4) == 0:
                        logger.debug(f"Game generation progress: {i + 1}/{num_games}")
                    game_start = time.time()
                    game = play_self_play_game(
                        model=self.model,
                        move_encoder=self.move_encoder,
                        mcts=mcts,
                        device=self.device,
                        max_moves=get_config('training', 'max_moves_per_game')
                    )
                    game_time = time.time() - game_start
                    games.append(game)
                    self.games_played += 1
                    completed += 1
                    moves_played = len(game.moves_played)
                    total_moves += moves_played
                    if game.is_game_over():
                        result = game.board.result()
                        game_outcomes[result] = game_outcomes.get(result, 0) + 1
                    else:
                        game_outcomes['timeout'] += 1
                    if logger.isEnabledFor(logging.DEBUG):
                        outcome = game.board.result() if game.is_game_over() else 'timeout'
                        logger.debug(f"Game {i + 1} completed: {moves_played} moves, {outcome}, {game_time:.2f}s")
                        if game_time > 10.0:
                            logger.warning(f"Slow game generation: {game_time:.1f}s for game {i + 1}")
                except Exception as e:
                    failed += 1
                    self.failed_games += 1
                    logger.warning(f"Game {i + 1} failed: {e}")
                    logger.debug(f"Game failure details: {type(e).__name__}")
                    continue
            if completed < num_games:
                failure_rate = failed / num_games * 100
                if failure_rate > 20:
                    logger.warning(f"High game failure rate: {failure_rate:.1f}% ({failed}/{num_games})")
                else:
                    logger.debug(f"Game generation incomplete: {completed}/{num_games} games ({failure_rate:.1f}% failed)")
            if completed == 0:
                logger.error("Failed to generate any games")
                raise RuntimeError("Failed to generate any games")
            avg_moves = total_moves / completed if completed > 0 else 0
            if logger.isEnabledFor(logging.DEBUG):
                gen_time = time.time() - gen_start
                games_per_sec = completed / gen_time if gen_time > 0 else 0
                logger.debug(f"Sequential game generation completed in {gen_time:.2f}s ({games_per_sec:.2f} games/sec)")
                logger.debug(f"Average game length: {avg_moves:.1f} moves")
                logger.debug(f"Game outcomes: {game_outcomes}")
            logger.info(f"Generated {completed} games sequentially (avg {avg_moves:.1f} moves, {failed} failed)")
            return games

    def train_iteration(self, 
                        num_games: int = None, 
                        mcts_simulations: int = None) -> Optional[Dict[str, float]]:
        if num_games is None:
            num_games = get_config('training', 'games_per_iteration')
        if mcts_simulations is None:
            mcts_simulations = get_config('mcts', 'simulations')
        iteration_num = self.training_steps + 1
        try:
            if logger.isEnabledFor(logging.DEBUG):
                iter_start = time.time()
                logger.debug(f"Starting training iteration {iteration_num}")
            logger.debug("Phase 1: Generating games...")
            game_start_time = time.time()
            games = self.generate_games(num_games, mcts_simulations)
            game_time = time.time() - game_start_time
            
            if self.use_experience_replay and self.experience_buffer:
                positions_added = self.experience_buffer.add_batch(games)
                logger.debug(f"Added {positions_added} positions to experience buffer")
                buffer_stats = self.experience_buffer.get_stats()
                logger.debug(f"Buffer stats: {buffer_stats['size']} positions, "
                            f"{buffer_stats['utilization']:.1%} full")
            
            if self.use_experience_replay and self.experience_buffer and self.experience_buffer.size > 0:
                logger.debug("Phase 2: Training on experience replay buffer...")
                train_start_time = time.time()
                
                epochs = 4
                exp_replay_config = get_config('training', 'experience_replay')
                if exp_replay_config and isinstance(exp_replay_config, dict):
                    epochs = exp_replay_config.get('epochs_per_iteration', 4)
                
                total_losses = []
                batch_size = 256
                
                for epoch in range(epochs):
                    if epoch == 0:
                        batch = self.experience_buffer.sample_recent(batch_size, recent_fraction=0.7)
                    else:
                        batch = self.experience_buffer.sample(batch_size)
                    
                    if batch:
                        loss_dict = self.train_on_batch(batch)
                        if loss_dict:
                            total_losses.append(loss_dict)
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Epoch {epoch + 1}/{epochs}: loss {loss_dict['total_loss']:.4f}")
                
                train_time = time.time() - train_start_time
                
                if total_losses:
                    avg_loss_dict = {
                        'total_loss': np.mean([l['total_loss'] for l in total_losses]),
                        'value_loss': np.mean([l['value_loss'] for l in total_losses]),
                        'policy_loss': np.mean([l['policy_loss'] for l in total_losses])
                    }
                    positions = batch_size * len(total_losses)
                else:
                    avg_loss_dict = None
                    positions = 0
                    
                loss_dict = avg_loss_dict
                
            else:
                logger.debug("Phase 2: Creating training batch...")
                batch_start_time = time.time()
                batch = create_training_batch(games, self.device)
                batch_time = time.time() - batch_start_time
                if batch is None:
                    logger.error("No training data from games")
                    logger.warning("Iteration failed: no valid training positions generated")
                    return None
                positions = len(batch['board_tensors'])
                logger.debug(f"Training batch created: {positions} positions from {len(games)} games")
                logger.debug(f"Batch creation time: {batch_time * 1000:.1f}ms")
                
                logger.debug("Phase 3: Training on batch...")
                train_start_time = time.time()
                loss_dict = self.train_on_batch(batch)
                train_time = time.time() - train_start_time
                
            total_time = time.time() - iter_start
            
            if loss_dict:
                logger.info(f"Iteration {iteration_num}: {positions} positions, {total_time:.1f}s total, loss {loss_dict['total_loss']:.4f}")
                if logger.isEnabledFor(logging.DEBUG):
                    game_pct = game_time / total_time * 100
                    train_pct = train_time / total_time * 100
                    logger.debug(f"Time breakdown - games: {game_time:.2f}s ({game_pct:.1f}%), "
                                f"training: {train_time:.2f}s ({train_pct:.1f}%)")
                if total_time > 15.0:
                    logger.warning(f"Slow iteration: {total_time:.1f}s (target <10s)")
                    if game_time > 10.0:
                        logger.warning(f"Game generation bottleneck: {game_time:.1f}s")
                    if train_time > 5.0:
                        logger.warning(f"Training bottleneck: {train_time:.1f}s")
                if loss_dict['total_loss'] > 20.0:
                    logger.warning(f"Very high loss detected: {loss_dict['total_loss']:.4f}")
                elif loss_dict['total_loss'] > 10.0:
                    logger.warning(f"High loss detected: {loss_dict['total_loss']:.4f}")
                positions_per_sec = positions / total_time if total_time > 0 else 0
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Training throughput: {positions_per_sec:.1f} positions/sec")
                    if positions_per_sec < 10:
                        logger.warning(f"Low training throughput: {positions_per_sec:.1f} positions/sec")
            return loss_dict
        except Exception as e:
            logger.error(f"Training iteration {iteration_num} failed after {self.games_played} total games: {e}")
            logger.debug(f"Iteration failure details: {type(e).__name__}")
            if "memory" in str(e).lower():
                logger.warning("Memory-related failure - consider reducing batch size or MCTS simulations")
            elif "cuda" in str(e).lower():
                logger.warning("CUDA-related failure - check GPU status and memory")
            raise RuntimeError(f"Training iteration {iteration_num} failed") from e

    def save_model(self, filepath: str) -> None:
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'games_played': self.games_played,
                'training_steps': self.training_steps,
                'losses': self.losses
            }
            torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Model save failed to {filepath} (step {self.training_steps}): {e}")
            raise RuntimeError(f"Model save failed") from e

    def load_model(self, filepath: str) -> None:
        try:
            checkpoint = torch.load(filepath, 
                                    map_location=self.device, 
                                    weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.games_played = checkpoint.get('games_played', 0)
            self.training_steps = checkpoint.get('training_steps', 0)
            self.losses = checkpoint.get('losses', [])
            logger.info(f"Model loaded from {filepath}")
            logger.info(f"Restored: {self.games_played} games, {self.training_steps} steps")
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Model load failed from {filepath}: {e}. Check file permissions and disk space.")
            raise RuntimeError(f"Model loading failed") from e

    def get_stats(self) -> str:
        if not self.losses:
            return "No training data yet"
        recent_window = min(RECENT_LOSSES_WINDOW, len(self.losses))
        recent_losses = self.losses[-recent_window:]
        avg_total = np.mean([l['total_loss'] for l in recent_losses])
        avg_value = np.mean([l['value_loss'] for l in recent_losses])
        avg_policy = np.mean([l['policy_loss'] for l in recent_losses])
        return (f"Games: {self.games_played}, Steps: {self.training_steps}, "
                f"Recent Avg Loss: {avg_total:.4f} "
                f"(Value: {avg_value:.4f}, Policy: {avg_policy:.4f})")
