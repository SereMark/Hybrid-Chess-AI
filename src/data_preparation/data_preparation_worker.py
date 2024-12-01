from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
import os, glob, time, numpy as np, h5py, chess.pgn, io, threading
from collections import defaultdict
from src.utils.chess_utils import get_move_mapping, convert_board_to_tensor, flip_board, flip_move
from src.utils.common_utils import format_time_left, log_message, should_stop, wait_if_paused

class DataPreparationWorker(BaseWorker):
    stats_update = pyqtSignal(dict)

    def __init__(self, raw_data_dir, processed_data_dir, max_games, min_elo, batch_size):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.max_games = max_games
        self.min_elo = min_elo
        self.batch_size = batch_size
        self.positions = defaultdict(
            lambda: defaultdict(lambda: {'win': 0, 'draw': 0, 'loss': 0, 'eco': '', 'name': ''})
        )
        self.game_counter = 0
        self.start_time = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.total_samples = 0
        self.total_games_processed = 0
        self.total_moves_processed = 0
        self.game_results_counter = {1.0: 0, -1.0: 0, 0.0: 0}
        self.game_length_bins = np.arange(0, 200, 5)
        self.game_length_histogram = np.zeros(len(self.game_length_bins) - 1, dtype=int)
        self.player_rating_bins = np.arange(1000, 3000, 50)
        self.player_rating_histogram = np.zeros(len(self.player_rating_bins) - 1, dtype=int)
        self.batch_inputs = []
        self.batch_policy_targets = []
        self.batch_value_targets = []
        self.current_dataset_size = 0
        self.move_mapping = get_move_mapping()

    def run_task(self):
        self.start_time = time.time()
        try:
            total_estimated_games = self._estimate_total_games()
            pgn_files = glob.glob(os.path.join(self.raw_data_dir, '*.pgn'))
            if not pgn_files:
                log_message(f"No PGN files found in {self.raw_data_dir}", self.log_update)
                return
            log_message(f"Found {len(pgn_files)} PGN files to process", self.log_update)
            os.makedirs(self.processed_data_dir, exist_ok=True)
            h5_file_path = os.path.join(self.processed_data_dir, 'dataset.h5')
            with h5py.File(h5_file_path, 'w') as h5_file:
                self._initialize_h5_datasets(h5_file)
                log_message("Initialized H5 datasets successfully", self.log_update)
                for filename in pgn_files:
                    if should_stop(self.stop_event):
                        log_message("Stopping processing due to stop event", self.log_update)
                        break
                    wait_if_paused(self.pause_event)
                    log_message(f"Processing file: {filename}", self.log_update)
                    with open(filename, 'r', errors='ignore') as f:
                        while True:
                            if should_stop(self.stop_event):
                                log_message("Stopping processing due to stop event", self.log_update)
                                break
                            wait_if_paused(self.pause_event)
                            game = chess.pgn.read_game(f)
                            if game is None:
                                break
                            game_str = str(game)
                            result = self._process_game(game_str)
                            if result is None:
                                continue
                            self._process_data_entry(result, h5_file)
                            self.total_games_processed += 1
                            if self.total_games_processed % 100 == 0:
                                self._update_progress_and_time_left(total_estimated_games)
                                self._emit_stats()
                            if self.total_games_processed >= self.max_games:
                                log_message("Reached maximum number of games", self.log_update)
                                break
                if self.batch_inputs:
                    log_message(f"Writing final batch of {len(self.batch_inputs)} samples", self.log_update)
                    self._write_batch_to_h5(h5_file)
            if not self.stop_event.is_set():
                log_message("Splitting dataset into training and validation sets...", self.log_update)
                self._split_dataset()
            else:
                log_message("Data preparation stopped by user request.", self.log_update)
        except Exception as e:
            log_message(f"Critical error during data processing: {str(e)}", self.log_update)
            raise

    def _initialize_h5_datasets(self, h5_file):
        self.h5_inputs = h5_file.create_dataset(
            'inputs', shape=(0, 20, 8, 8), maxshape=(None, 20, 8, 8),
            dtype=np.float32, compression='gzip'
        )
        self.h5_policy_targets = h5_file.create_dataset(
            'policy_targets', shape=(0,), maxshape=(None,),
            dtype=np.int64, compression='gzip'
        )
        self.h5_value_targets = h5_file.create_dataset(
            'value_targets', shape=(0,), maxshape=(None,),
            dtype=np.float32, compression='gzip'
        )

    def _process_data_entry(self, data, h5_file):
        inputs = data['inputs']
        policy_targets = data['policy_targets']
        value_targets = data['value_targets']
        game_length = data['game_length']
        avg_rating = data['avg_rating']
        game_result = data['game_result']
        num_new_samples = len(inputs)
        if num_new_samples > 0:
            self.total_samples += num_new_samples
            self.total_moves_processed += num_new_samples
            self.game_results_counter[game_result] += 1
            self._update_histograms(game_length, avg_rating)
            self.batch_inputs.extend(inputs)
            self.batch_policy_targets.extend(policy_targets)
            self.batch_value_targets.extend(value_targets)
            if len(self.batch_inputs) >= self.batch_size:
                self._write_batch_to_h5(h5_file)

    def _write_batch_to_h5(self, h5_file):
        try:
            if not self.batch_inputs:
                log_message("Warning: Attempted to write empty batch", self.log_update)
                return
            batch_size = len(self.batch_inputs)
            start_idx = self.current_dataset_size
            end_idx = self.current_dataset_size + batch_size
            self.h5_inputs.resize((end_idx, 20, 8, 8))
            self.h5_policy_targets.resize((end_idx,))
            self.h5_value_targets.resize((end_idx,))
            self.h5_inputs[start_idx:end_idx] = np.array(self.batch_inputs, dtype=np.float32)
            self.h5_policy_targets[start_idx:end_idx] = np.array(self.batch_policy_targets, dtype=np.int64)
            self.h5_value_targets[start_idx:end_idx] = np.array(self.batch_value_targets, dtype=np.float32)
            self.current_dataset_size += batch_size
            self.batch_inputs.clear()
            self.batch_policy_targets.clear()
            self.batch_value_targets.clear()
        except Exception as e:
            log_message(f"Error writing batch to H5: {str(e)}", self.log_update)
            raise

    def _update_histograms(self, game_length, avg_rating):
        length_idx = np.digitize(game_length, self.game_length_bins) - 1
        if 0 <= length_idx < len(self.game_length_histogram):
            self.game_length_histogram[length_idx] += 1
        if avg_rating:
            rating_idx = np.digitize(avg_rating, self.player_rating_bins) - 1
            if 0 <= rating_idx < len(self.player_rating_histogram):
                self.player_rating_histogram[rating_idx] += 1

    def _estimate_total_games(self):
        try:
            total_games = 0
            avg_game_size = 5000
            pgn_files = glob.glob(os.path.join(self.raw_data_dir, '*.pgn'))
            for filename in pgn_files:
                file_size = os.path.getsize(filename)
                estimated_games = file_size // avg_game_size
                total_games += estimated_games
            estimated_total_games = min(total_games, self.max_games)
            return estimated_total_games
        except Exception as e:
            log_message(f"Error estimating total games: {str(e)}", self.log_update)
            return self.max_games

    def _update_progress_and_time_left(self, total_estimated_games):
        if self.progress_update:
            progress_percentage = int((self.total_games_processed / total_estimated_games) * 100)
            self.progress_update.emit(progress_percentage)
        if self.time_left_update:
            elapsed_time = time.time() - self.start_time
            if self.total_games_processed > 0:
                estimated_total_time = (elapsed_time / self.total_games_processed) * total_estimated_games
                time_left = estimated_total_time - elapsed_time
                time_left = max(0, time_left)
                time_left_str = format_time_left(time_left)
                self.time_left_update.emit(time_left_str)
            else:
                self.time_left_update.emit("Calculating...")

    def _emit_stats(self):
        if self.stats_update:
            stats = {
                'total_games_processed': self.total_games_processed,
                'total_moves_processed': self.total_moves_processed,
                'game_results_counter': self.game_results_counter.copy(),
                'game_length_bins': self.game_length_bins.tolist(),
                'game_length_histogram': self.game_length_histogram.tolist(),
                'player_rating_bins': self.player_rating_bins.tolist(),
                'player_rating_histogram': self.player_rating_histogram.tolist()
            }
            self.stats_update.emit(stats)

    def _split_dataset(self):
        try:
            output_dir = self.processed_data_dir
            h5_file_path = os.path.join(output_dir, 'dataset.h5')
            train_indices_path = os.path.join(output_dir, 'train_indices.npy')
            val_indices_path = os.path.join(output_dir, 'val_indices.npy')
            test_indices_path = os.path.join(output_dir, 'test_indices.npy')
            with h5py.File(h5_file_path, 'r') as h5_file:
                num_samples = h5_file['inputs'].shape[0]
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                train_end = int(num_samples * 0.8)
                val_end = int(num_samples * 0.9)
                train_indices = indices[:train_end]
                val_indices = indices[train_end:val_end]
                test_indices = indices[val_end:]
                np.save(train_indices_path, train_indices)
                np.save(val_indices_path, val_indices)
                np.save(test_indices_path, test_indices)
            log_message("Dataset split into training, validation, and test sets.", self.log_update)
        except Exception as e:
            log_message(f"Error during dataset splitting: {e}", self.log_update)

    def _process_game(self, game_str):
        try:
            game = chess.pgn.read_game(io.StringIO(game_str))
            if game is None:
                log_message("Skipped a game: Unable to parse game.", self.log_update)
                return None
            headers = game.headers
            white_elo_str = headers.get('WhiteElo')
            black_elo_str = headers.get('BlackElo')
            if white_elo_str is None or black_elo_str is None:
                log_message("Skipped a game: Missing WhiteElo or BlackElo.", self.log_update)
                return None
            try:
                white_elo = int(white_elo_str)
                black_elo = int(black_elo_str)
            except ValueError:
                log_message("Skipped a game: Non-integer ELO value.", self.log_update)
                return None
            if white_elo < self.min_elo or black_elo < self.min_elo:
                return None
            avg_rating = (white_elo + black_elo) / 2
            result = headers.get('Result', '*')
            if result == '1-0':
                game_result = 1.0
            elif result == '0-1':
                game_result = -1.0
            elif result == '1/2-1/2':
                game_result = 0.0
            else:
                log_message("Skipped a game: Unrecognized result format.", self.log_update)
                return None
            inputs = []
            policy_targets = []
            value_targets = []
            game_length = 0
            board = game.board()
            moves = list(game.mainline_moves())
            game_length = len(moves)
            for move in moves:
                current_tensor = convert_board_to_tensor(board)
                move_idx = self.move_mapping.get_index_by_move(move)
                if move_idx is None:
                    log_message(f"Skipped a move: Move '{move}' not in MOVE_MAPPING.", self.log_update)
                    board.push(move)
                    continue
                inputs.append(current_tensor)
                policy_targets.append(move_idx)
                value_target = game_result if board.turn == chess.WHITE else -game_result
                value_targets.append(value_target)
                flipped_board = flip_board(board)
                flipped_move = flip_move(move)
                flipped_move_idx = self.move_mapping.get_index_by_move(flipped_move)
                if flipped_move_idx is not None:
                    flipped_tensor = convert_board_to_tensor(flipped_board)
                    inputs.append(flipped_tensor)
                    policy_targets.append(flipped_move_idx)
                    flipped_value_target = -value_target
                    value_targets.append(flipped_value_target)
                else:
                    log_message(f"Skipped a flipped move: Move '{flipped_move}' not in MOVE_MAPPING.", self.log_update)
                board.push(move)
            if not inputs:
                log_message("Skipped a game: No valid moves found.", self.log_update)
                return None
            return {
                'inputs': inputs,
                'policy_targets': policy_targets,
                'value_targets': value_targets,
                'game_length': game_length,
                'avg_rating': avg_rating,
                'game_result': game_result
            }
        except Exception as e:
            log_message(f"Error processing game: {str(e)}", self.log_update)
            return None

    def stop(self):
        super().stop()
        self.stop_event.set()
        self.pause_event.set()

    def pause(self):
        self.pause_event.clear()
        log_message("Processing paused.", self.log_update)

    def resume(self):
        self.pause_event.set()
        log_message("Processing resumed.", self.log_update)