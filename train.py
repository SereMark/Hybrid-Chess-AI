
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import psutil
from model import ChessNet
from selfplay import SelfPlayEngine
from encoder import PositionEncoder
from config import Config


class AlphaZeroTrainer:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model = ChessNet().to(self.device)
        
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=Config.LEARNING_RATE_INIT,
            momentum=Config.MOMENTUM,
            weight_decay=Config.WEIGHT_DECAY,
            nesterov=True
        )
        
        milestones = [step[0] for step in Config.LEARNING_RATE_SCHEDULE]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.1
        )
        
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        self.selfplay_engine = SelfPlayEngine(self.model, self.device)
        self.encoder = PositionEncoder()
        
        self.iteration = 0
        self.total_games = 0
        self.start_time = time.time()
        
    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def get_memory_info(self):
        if self.device.type == 'cuda':
            return {
                'gpu_memory_used': torch.cuda.memory_allocated(self.device) / 1024**3,
                'gpu_memory_total': torch.cuda.memory_reserved(self.device) / 1024**3
            }
        return {'gpu_memory_used': 0, 'gpu_memory_total': 0}
    
    def train_step(self, batch_data):
        states, policies, values = batch_data
        
        x = torch.from_numpy(self.encoder.encode_batch(states)).to(self.device)
        pi_target = torch.from_numpy(np.array(policies)).to(self.device)
        v_target = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        self.model.train()
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            pi_pred, v_pred = self.model(x)
            
            policy_loss = F.kl_div(F.log_softmax(pi_pred, dim=1), pi_target, reduction='batchmean')
            value_loss = F.mse_loss(v_pred, v_target)
            total_loss = Config.POLICY_WEIGHT * policy_loss + Config.VALUE_WEIGHT * value_loss
        
        self.optimizer.zero_grad()
        
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP)
            self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def training_iteration(self):
        iteration_stats = {}
        total_iter_start = time.time()
        
        elapsed_total = time.time() - self.start_time
        if self.iteration > 0:
            eta_total = elapsed_total * (Config.ITERATIONS - self.iteration) / self.iteration
        else:
            eta_total = 0
        
        mem_info = self.get_memory_info()
        
        print(f"\n[Iteration {self.iteration}/{Config.ITERATIONS}] | "
              f"ETA: {self.format_time(eta_total)} | "
              f"GPU: {mem_info['gpu_memory_used']:.1f}/{mem_info['gpu_memory_total']:.1f}GB")
        
        selfplay_start = time.time()
        game_stats = self.selfplay_engine.play_games(Config.GAMES_PER_ITERATION)
        self.total_games += game_stats['games']
        selfplay_elapsed = time.time() - selfplay_start
        
        games_per_min = game_stats['games'] / (selfplay_elapsed / 60) if selfplay_elapsed > 0 else 0
        moves_per_sec = game_stats['moves'] / selfplay_elapsed if selfplay_elapsed > 0 else 0
        avg_game_length = game_stats['moves'] / game_stats['games'] if game_stats['games'] > 0 else 0
        
        win_pct = game_stats['white_wins'] / game_stats['games'] * 100 if game_stats['games'] > 0 else 0
        draw_pct = game_stats['draws'] / game_stats['games'] * 100 if game_stats['games'] > 0 else 0
        loss_pct = game_stats['black_wins'] / game_stats['games'] * 100 if game_stats['games'] > 0 else 0
        
        print(f"Self-play: {game_stats['games']} games | {games_per_min:.1f} games/min | "
              f"{moves_per_sec/1000:.1f}K moves/sec | avg {avg_game_length:.1f} moves")
        print(f"Results:   W:{win_pct:.1f}% D:{draw_pct:.1f}% L:{loss_pct:.1f}% | "
              f"Time: {self.format_time(selfplay_elapsed)}")
        
        iteration_stats.update(game_stats)
        iteration_stats['selfplay_time'] = selfplay_elapsed
        iteration_stats['games_per_min'] = games_per_min
        iteration_stats['moves_per_sec'] = moves_per_sec
        
        train_start = time.time()
        training_losses = []
        for step in range(Config.TRAIN_STEPS_PER_ITER):
            batch = self.selfplay_engine.generate_batch(Config.BATCH_SIZE)
            if batch:
                losses = self.train_step(batch)
                training_losses.append(losses)
            
            if step % 100 == 0 and step > 0:
                print(f"  Training step {step}/{Config.TRAIN_STEPS_PER_ITER}", end="\r", flush=True)
        
        train_elapsed = time.time() - train_start
        self.scheduler.step()
        
        if training_losses:
            avg_policy_loss = np.mean([loss[0] for loss in training_losses])
            avg_value_loss = np.mean([loss[1] for loss in training_losses])
            current_lr = self.scheduler.get_last_lr()[0]
            buffer_size = self.selfplay_engine.get_buffer_size()
            buffer_pct = (buffer_size / Config.BUFFER_SIZE) * 100
            
            batches_per_sec = len(training_losses) / train_elapsed if train_elapsed > 0 else 0
            
            print(f"Training:  {len(training_losses)} steps | {batches_per_sec:.1f} batches/sec | "
                  f"Time: {self.format_time(train_elapsed)}")
            print(f"Loss:      Policy {avg_policy_loss:.4f} | Value {avg_value_loss:.4f} | "
                  f"LR {current_lr:.2e} | Buffer {buffer_pct:.0f}%")
            
            iteration_stats.update({
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'learning_rate': current_lr,
                'buffer_size': buffer_size,
                'buffer_percent': buffer_pct,
                'training_time': train_elapsed,
                'batches_per_sec': batches_per_sec,
                'total_iteration_time': time.time() - total_iter_start
            })
        
        return iteration_stats
    
    def save_checkpoint(self, filepath=None):
        if filepath is None:
            filepath = f"checkpoint_{self.iteration}.pt"
        
        checkpoint = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_time': time.time() - self.start_time,
        }
        
        torch.save(checkpoint, filepath)
        checkpoint_size = os.path.getsize(filepath) / 1024**2
        print(f"Checkpoint: Saved {filepath} | {checkpoint_size:.1f}MB | "
              f"Games: {self.total_games:,} | Time: {self.format_time(time.time() - self.start_time)}")
    
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {filepath} (iteration {self.iteration})")
    
    def train(self):
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        mem_info = self.get_memory_info()
        
        print("Starting training...")
        print(f"Device: {self.device} | Memory: {mem_info['gpu_memory_total']:.1f}GB")
        print(f"Model: {total_params:.1f}M parameters | {Config.BLOCKS} blocks x {Config.CHANNELS} channels")
        print(f"Training: {Config.ITERATIONS:,} iterations | {Config.GAMES_PER_ITERATION:,} games/iter")
        print(f"Expected: {Config.ITERATIONS * Config.GAMES_PER_ITERATION:,} total games")
        
        for iteration in range(1, Config.ITERATIONS + 1):
            self.iteration = iteration
            stats = self.training_iteration()
            
            if iteration % Config.CHECKPOINT_FREQ == 0:
                self.save_checkpoint()
        
        total_time = time.time() - self.start_time
        avg_time_per_iter = total_time / Config.ITERATIONS
        
        print("\nTraining finished!")
        print(f"Total time: {self.format_time(total_time)}")
        print(f"Total games: {self.total_games:,}")
        print(f"Avg time/iter: {self.format_time(avg_time_per_iter)}")
        print(f"Games/hour: {self.total_games / (total_time / 3600):.0f}")


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    
    trainer = AlphaZeroTrainer()
    trainer.train()
