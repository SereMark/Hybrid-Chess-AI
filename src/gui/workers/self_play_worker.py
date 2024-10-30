import os, torch, threading, time, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PyQt5.QtCore import QObject, pyqtSignal
from src.self_play.self_play import SelfPlay
from src.models.model import ChessModel
from torch.cuda.amp import GradScaler, autocast


class SelfPlayWorker(QObject):
    log_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    time_left_update = pyqtSignal(str)
    stats_update = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, model_path, output_dir, num_iterations, num_games_per_iteration,
                 simulations, c_puct, temperature, num_epochs, batch_size, automatic_batch_size,
                 num_threads, stop_event):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.automatic_batch_size = automatic_batch_size
        self.num_threads = num_threads
        self.stop_event = stop_event
        self.start_time = None
        self.total_games_played = 0
        self.results = []
        self.game_lengths = []
        self.lock = threading.Lock()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = GradScaler(enabled=(self.device == 'cuda'))

    def run(self):
        self.log_update.emit("=== Starting Self-Play Training ===")
        self.log_update.emit(f"Device: {self.device}")
        if self.device == 'cuda':
            self.log_update.emit(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            available_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            available_memory_gb = available_memory / 1024**3
            self.log_update.emit(f"Available GPU memory: {available_memory_gb:.2f} GB")

        self.log_update.emit(f"\nLoading model from {self.model_path}")
        try:
            model = ChessModel()
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            self.log_update.emit("Model loaded successfully")
        except Exception as e:
            self.log_update.emit(f"Error loading model: {str(e)}")
            self.finished.emit()
            return

        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=2e-4)
        self.log_update.emit("Optimizer initialized with lr=0.0005, weight_decay=2e-4")

        if self.automatic_batch_size:
            old_batch_size = self.batch_size
            self.batch_size = self.estimate_batch_size(model)
            self.log_update.emit(f"Automatic batch size estimation: {old_batch_size} → {self.batch_size}")
        else:
            self.log_update.emit(f"Using manual batch size: {self.batch_size}")

        self.start_time = time.time()
        self.log_update.emit(f"\nTraining started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        for iteration in range(self.num_iterations):
            if self.stop_event.is_set():
                self.log_update.emit("\nTraining stopped by user")
                break

            iteration_start_time = time.time()
            self.log_update.emit(f"\n=== Iteration {iteration + 1}/{self.num_iterations} ===")

            self.log_update.emit("\nStarting self-play phase...")
            model.eval()
            self_play_data = self.generate_self_play_data(model)
            self.log_update.emit(f"Self-play completed - Generated {len(self_play_data[0])} positions")

            self.log_update.emit("\nStarting training phase...")
            model.train()
            self.train_on_self_play_data(model, optimizer, self_play_data)

            self.save_model(model, iteration)

            iteration_time = time.time() - iteration_start_time
            self.log_update.emit(f"\nIteration completed in {self.format_time_left(iteration_time)}")

            if self.device == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                self.log_update.emit(f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")

            total_time = time.time() - self.start_time
            self.log_update.emit(f"\n=== Training Complete ===")
            self.log_update.emit(f"Total time: {self.format_time_left(total_time)}")
            self.log_update.emit(f"Total games played: {self.total_games_played}")

            try:
                final_model_path = os.path.join('models', 'saved_models', 'final_model.pth')
                os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'training_stats': {
                        'total_games_played': self.total_games_played,
                        'total_training_time': total_time,
                        'win_rate': self.results.count(1) / len(self.results) if self.results else 0,
                        'draw_rate': self.results.count(0) / len(self.results) if self.results else 0,
                        'average_game_length': sum(self.game_lengths) / len(self.game_lengths) if self.game_lengths else 0,
                        'final_iteration': self.num_iterations
                    }
                }
                
                torch.save(checkpoint, final_model_path)
                file_size_mb = os.path.getsize(final_model_path) / (1024**2)
                self.log_update.emit("\n=== Final Model Saved ===")
                self.log_update.emit(f"Path: {final_model_path}")
                self.log_update.emit(f"File size: {file_size_mb:.1f}MB")
                self.log_update.emit(f"Games played: {self.total_games_played}")
                
                if self.results:
                    win_rate = self.results.count(1) / len(self.results)
                    draw_rate = self.results.count(0) / len(self.results)
                    loss_rate = self.results.count(-1) / len(self.results)
                    self.log_update.emit("\n=== Final Statistics ===")
                    self.log_update.emit(f"Win rate:  {win_rate:.1%}")
                    self.log_update.emit(f"Draw rate: {draw_rate:.1%}")
                    self.log_update.emit(f"Loss rate: {loss_rate:.1%}")
                
                if self.game_lengths:
                    avg_length = sum(self.game_lengths) / len(self.game_lengths)
                    min_length = min(self.game_lengths)
                    max_length = max(self.game_lengths)
                    self.log_update.emit(f"\nGame Lengths:")
                    self.log_update.emit(f"Average: {avg_length:.1f} moves")
                    self.log_update.emit(f"Minimum: {min_length} moves")
                    self.log_update.emit(f"Maximum: {max_length} moves")
                    
            except Exception as e:
                self.log_update.emit(f"\nError saving final model: {str(e)}")

            self.log_update.emit("\n" + "="*50)
            self.finished.emit()

    def estimate_batch_size(self, model):
        self.log_update.emit("\n=== Batch Size Estimation ===")
        if self.device == 'cpu':
            self.log_update.emit("CPU detected - Using default batch size 32")
            return 32

        try:
            torch.cuda.empty_cache()
            device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            max_mem = device_properties.total_memory
            self.log_update.emit(f"Total GPU Memory: {max_mem / (1024**3):.2f} GB")
            self.log_update.emit("Estimating memory requirements...")

            model.eval()
            test_input = torch.randn(2, 20, 8, 8, device=self.device)

            with autocast('cuda', enabled=(self.device == 'cuda')):
                with torch.no_grad():
                    _ = model(test_input)

            torch.cuda.synchronize()
            baseline_mem = torch.cuda.memory_allocated()
            mem_per_sample = baseline_mem / 2
            self.log_update.emit(f"Memory per sample (baseline): {mem_per_sample / 1024**2:.1f} MB")

            per_sample_mem = (5 * 1024**2) + mem_per_sample
            available_mem = max_mem * 0.5
            estimated_batch_size = max(int(available_mem / per_sample_mem), 2)
            estimated_batch_size = min(estimated_batch_size, 1024)

            self.log_update.emit(f"Estimated optimal batch size: {estimated_batch_size}")
            model.train()
            return estimated_batch_size

        except Exception as e:
            self.log_update.emit(f"Error during batch size estimation: {str(e)}")
            self.log_update.emit("Falling back to default batch size 32")
            return 32

    def generate_self_play_data(self, model):
        inputs_list = []
        policy_targets_list = []
        value_targets_list = []

        self.total_games_played_iteration = 0
        self.results_iteration = []
        self.game_lengths_iteration = []

        self.log_update.emit(f"\nStarting {self.num_threads} self-play threads")
        threads = []
        for thread_id in range(self.num_threads):
            t = threading.Thread(
                target=self.play_and_collect,
                args=(model, inputs_list, policy_targets_list, value_targets_list, thread_id)
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        self.log_update.emit(f"\nSelf-play complete - Processing collected data")
        total_positions = len(inputs_list)
        if total_positions == 0:
            self.log_update.emit("No data collected from self-play.")
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device), torch.empty(0, device=self.device)

        inputs = torch.tensor(inputs_list, dtype=torch.float32, device=self.device)
        policy_targets = torch.tensor(policy_targets_list, dtype=torch.float32, device=self.device)
        value_targets = torch.tensor(value_targets_list, dtype=torch.float32, device=self.device)
        self.log_update.emit(f"Collected {total_positions} positions")

        with self.lock:
            self.results.extend(self.results_iteration)
            self.game_lengths.extend(self.game_lengths_iteration)
            self.total_games_played += self.total_games_played_iteration

        return inputs, policy_targets, value_targets

    def play_and_collect(self, model, inputs_list, policy_targets_list, value_targets_list, thread_id):
        try:
            self.log_update.emit(f"Thread {thread_id}: Initializing self-play agent...")
            self_play = SelfPlay(model, self.device, self.simulations, self.c_puct, self.temperature)
            self_play.set_logger(self.log_update)
            games_per_thread = self.num_games_per_iteration // self.num_threads
            if thread_id < self.num_games_per_iteration % self.num_threads:
                games_per_thread += 1

            thread_start_time = time.time()
            self.log_update.emit(f"Thread {thread_id}: Starting {games_per_thread} games...")

            for game_num in range(games_per_thread):
                if self.stop_event.is_set():
                    self.log_update.emit(f"Thread {thread_id}: Stopped by user")
                    break

                game_start_time = time.time()
                self.log_update.emit(f"Thread {thread_id}: Starting game {game_num + 1}")

                try:
                    states, mcts_probs, winners, game_length, result = self_play.play_game()
                    game_time = time.time() - game_start_time

                    self.log_update.emit(
                        f"Thread {thread_id}: Game {game_num + 1} completed: "
                        f"Length: {game_length} moves, "
                        f"Result: {'Win' if result == 1.0 else 'Loss' if result == -1.0 else 'Draw'}, "
                        f"Positions: {len(states)}, "
                        f"Time: {game_time:.1f}s"
                    )

                    with self.lock:
                        inputs_list.extend(states)
                        policy_targets_list.extend(mcts_probs)
                        value_targets_list.extend(winners)
                        
                        self.total_games_played_iteration += 1
                        self.results_iteration.append(result)
                        self.game_lengths_iteration.append(game_length)
                        
                        self.total_games_played += 1
                        self.results.append(result)
                        self.game_lengths.append(game_length)
                
                except Exception as game_error:
                    self.log_update.emit(f"Thread {thread_id}: Error in game {game_num + 1}: {str(game_error)}")
                    continue

            thread_time = time.time() - thread_start_time
            self.log_update.emit(
                f"Thread {thread_id} complete - "
                f"{self.total_games_played_iteration} games in {thread_time:.1f}s "
                f"({thread_time / max(1, self.total_games_played_iteration):.1f}s per game)"
            )
            
            self.emit_stats_update()
            self.emit_progress()

        except Exception as thread_error:
            self.log_update.emit(f"Thread {thread_id}: Critical error: {str(thread_error)}")
            raise

    def train_on_self_play_data(self, model, optimizer, self_play_data):
        inputs, policy_targets, value_targets = self_play_data
        if inputs.numel() == 0:
            self.log_update.emit("No data to train on. Skipping training phase.")
            return

        inputs = inputs.cpu()
        policy_targets = policy_targets.cpu()
        value_targets = value_targets.cpu()

        dataset = TensorDataset(inputs, policy_targets, value_targets)
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=(self.device == 'cuda')
        )

        self.log_update.emit(f"\nTraining on {len(dataset)} positions")
        self.log_update.emit(f"Batch size: {self.batch_size}, Batches per epoch: {len(loader)}")

        best_loss = float('inf')
        no_improvement_count = 0

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0

            for batch_idx, (batch_inputs, batch_policy_targets, batch_value_targets) in enumerate(loader):
                optimizer.zero_grad()
                
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_policy_targets = batch_policy_targets.to(self.device, non_blocking=True)
                batch_value_targets = batch_value_targets.to(self.device, non_blocking=True)

                with autocast(enabled=(self.device == 'cuda')):
                    policy_preds, value_preds = model(batch_inputs)

                    policy_loss = -torch.mean(torch.sum(batch_policy_targets * F.log_softmax(policy_preds, dim=1), dim=1))
                    value_loss = F.mse_loss(value_preds.view(-1), batch_value_targets)
                    loss = policy_loss + value_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

                if (batch_idx + 1) % max(1, len(loader) // 10) == 0:
                    self.log_update.emit(
                        f"Epoch {epoch + 1}/{self.num_epochs} - "
                        f"Batch {batch_idx + 1}/{len(loader)} - "
                        f"Loss: {loss.item():.4f}"
                    )

                if self.device == 'cuda':
                    del batch_inputs, batch_policy_targets, batch_value_targets
                    torch.cuda.empty_cache()

            avg_loss = total_loss / len(loader)
            avg_policy_loss = total_policy_loss / len(loader)
            avg_value_loss = total_value_loss / len(loader)
            epoch_time = time.time() - epoch_start_time

            self.log_update.emit(
                f"\nEpoch {epoch + 1}/{self.num_epochs} Summary:"
                f"\n  Average Loss: {avg_loss:.4f}"
                f"\n  Policy Loss: {avg_policy_loss:.4f}"
                f"\n  Value Loss: {avg_value_loss:.4f}"
                f"\n  Time: {self.format_time_left(epoch_time)}"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improvement_count = 0
                self.log_update.emit(f"  New best loss: {best_loss:.4f}")
            else:
                no_improvement_count += 1
                self.log_update.emit(f"  No improvement for {no_improvement_count} epochs")

            if self.device == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                self.log_update.emit(
                    f"  GPU Memory Status:"
                    f"\n    Allocated: {memory_allocated:.1f}MB"
                    f"\n    Reserved: {memory_reserved:.1f}MB"
                )

            if no_improvement_count >= 5:
                self.log_update.emit("  Early stopping triggered.")
                break

    def save_model(self, model, iteration):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            model_save_path = os.path.join(self.output_dir, f'model_iteration_{iteration + 1}.pth')

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'iteration': iteration + 1,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'training_stats': {
                    'total_games_played': self.total_games_played,
                    'win_rate': self.results.count(1) / len(self.results) if self.results else 0,
                    'average_game_length': sum(self.game_lengths) / len(self.game_lengths) if self.game_lengths else 0,
                }
            }

            torch.save(checkpoint, model_save_path)
            file_size_mb = os.path.getsize(model_save_path) / (1024**2)
            self.log_update.emit(f"\nModel checkpoint saved:")
            self.log_update.emit(f"  Path: {model_save_path}")
            self.log_update.emit(f"  Iteration: {iteration + 1}")
            self.log_update.emit(f"  Games played: {self.total_games_played}")
            self.log_update.emit(f"  File size: {file_size_mb:.1f}MB")

        except Exception as e:
            self.log_update.emit(f"\nError saving model: {str(e)}")

    def emit_stats_update(self):
        with self.lock:
            if self.game_lengths:
                avg_game_length = sum(self.game_lengths) / len(self.game_lengths)
                min_game_length = min(self.game_lengths)
                max_game_length = max(self.game_lengths)

                wins = self.results.count(1)
                losses = self.results.count(-1)
                draws = self.results.count(0)
                total_games = len(self.results)

                elapsed_time = time.time() - self.start_time
                games_per_second = self.total_games_played / elapsed_time if elapsed_time > 0 else 0

                stats = {
                    'games_played': self.total_games_played,
                    'wins': wins,
                    'losses': losses,
                    'draws': draws,
                    'win_rate': wins / total_games if total_games > 0 else 0,
                    'draw_rate': draws / total_games if total_games > 0 else 0,
                    'loss_rate': losses / total_games if total_games > 0 else 0,
                    'average_game_length': avg_game_length,
                    'min_game_length': min_game_length,
                    'max_game_length': max_game_length,
                    'games_per_second': games_per_second
                }

                self.stats_update.emit(stats)

    def emit_progress(self):
        with self.lock:
            total_games = self.num_iterations * self.num_games_per_iteration
            progress = min(int((self.total_games_played / total_games) * 100), 100)
            self.progress_update.emit(progress)

            elapsed_time = time.time() - self.start_time
            games_per_second = self.total_games_played / elapsed_time if elapsed_time > 0 else 0

            if self.total_games_played > 0:
                time_per_game = elapsed_time / self.total_games_played
                estimated_total_time = time_per_game * total_games
                time_left = estimated_total_time - elapsed_time
                time_left_str = self.format_time_left(time_left)
                
                self.time_left_update.emit(
                    f"Time left: {time_left_str} ({games_per_second:.2f} games/sec)"
                )
            else:
                self.time_left_update.emit("Time Left: Calculating...")

    @staticmethod
    def format_time_left(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"