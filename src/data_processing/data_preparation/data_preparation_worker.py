import os
import io
import time
from collections import defaultdict
import chess.pgn
import h5py
import numpy as np
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.utils.chess_utils import convert_board_to_tensor, flip_board, flip_move, get_move_mapping
from src.utils.common_utils import estimate_total_games, update_progress_time_left, wait_if_paused, parse_game_result

class DataPreparationWorker(BaseWorker):
    stats_update = pyqtSignal(dict)

    def __init__(self, raw_pgn_file: str, max_games: int, min_elo: int, batch_size: int):
        super().__init__()
        self.raw_pgn_file = raw_pgn_file
        self.max_games = max_games
        self.min_elo = min_elo
        self.batch_size = batch_size

        self.positions = defaultdict(lambda: defaultdict(lambda: {"win": 0, "draw": 0, "loss": 0, "eco": "", "name": ""}))
        self.game_counter = 0
        self.start_time = None
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
        self.output_dir = os.path.abspath(os.path.join("data", "processed"))
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)

    def run_task(self):
        self.start_time = time.time()
        try:
            if not os.path.isfile(self.raw_pgn_file):
                self.logger.warning(f"No PGN file found at {self.raw_pgn_file}. Aborting data preparation.")
                return

            total_estimated_games = estimate_total_games(file_paths=self.raw_pgn_file, avg_game_size=5000, max_games=self.max_games, logger=self.logger)

            h5_path = os.path.join(self.output_dir, "dataset.h5")

            with h5py.File(h5_path, "w") as h5_file:
                self.h5_inputs = h5_file.create_dataset("inputs", shape=(0, 20, 8, 8), maxshape=(None, 20, 8, 8), dtype=np.float32, compression="gzip")
                self.h5_policy_targets = h5_file.create_dataset("policy_targets", shape=(0,), maxshape=(None,), dtype=np.int64, compression="gzip")
                self.h5_value_targets = h5_file.create_dataset("value_targets", shape=(0,), maxshape=(None,), dtype=np.float32, compression="gzip")

                file_games_count = 0
                fsize = os.path.getsize(self.raw_pgn_file)
                self.logger.info(f"Processing PGN file: {os.path.basename(self.raw_pgn_file)} (~{fsize} bytes).")

                with open(self.raw_pgn_file, "r", errors="ignore") as f:
                    while (self.total_games_processed < self.max_games and not self._is_stopped.is_set()):
                        wait_if_paused(self._is_paused)
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        game_str = str(game)
                        result = self._process_game(game_str)
                        if result is None:
                            continue
                        self._process_data_entry(result, h5_file)
                        self.total_games_processed += 1
                        file_games_count += 1

                        if self.total_games_processed % 100 == 0:
                            update_progress_time_left(self.progress_update, self.time_left_update, self.start_time, self.total_games_processed, total_estimated_games)
                            self._emit_stats()

                # Write any remaining data in batches
                if self.batch_inputs:
                    self._write_batch_to_h5()

            if not self._is_stopped.is_set():
                self._split_dataset()

        except Exception as e:
            self.logger.error(f"Critical error in data preparation: {str(e)}")
            raise

    def _process_game(self, game_str: str):
        try:
            game = chess.pgn.read_game(io.StringIO(game_str))
            if game is None:
                return None

            headers = game.headers
            white_elo_str = headers.get("WhiteElo")
            black_elo_str = headers.get("BlackElo")
            if white_elo_str is None or black_elo_str is None:
                return None

            try:
                white_elo = int(white_elo_str)
                black_elo = int(black_elo_str)
            except ValueError:
                return None

            if white_elo < self.min_elo or black_elo < self.min_elo:
                return None

            avg_rating = (white_elo + black_elo) / 2
            result = headers.get("Result", "*")
            game_result = parse_game_result(result)
            if game_result is None:
                return None

            board = game.board()
            moves = list(game.mainline_moves())
            inputs, policy_targets, value_targets = self._extract_move_data(board, moves, game_result)

            if not inputs:
                return None

            return {
                "inputs": inputs,
                "policy_targets": policy_targets,
                "value_targets": value_targets,
                "game_length": len(moves),
                "avg_rating": avg_rating,
                "game_result": game_result,
            }

        except Exception as e:
            self.logger.error(f"Error processing game entry: {str(e)}")
            return None

    def _extract_move_data(self, board, moves, game_result):
        inputs = []
        policy_targets = []
        value_targets = []

        for move in moves:
            current_tensor = convert_board_to_tensor(board)
            move_idx = self.move_mapping.get_index_by_move(move)
            if move_idx is None:
                board.push(move)
                continue

            inputs.append(current_tensor)
            value_target = game_result if board.turn == chess.WHITE else -game_result
            value_targets.append(value_target)
            policy_targets.append(move_idx)

            # Handle board flipping for data augmentation
            flipped_board = flip_board(board)
            flipped_move = flip_move(move)
            flipped_move_idx = self.move_mapping.get_index_by_move(flipped_move)
            if flipped_move_idx is not None:
                flipped_tensor = convert_board_to_tensor(flipped_board)
                inputs.append(flipped_tensor)
                policy_targets.append(flipped_move_idx)
                flipped_value_target = -value_target
                value_targets.append(flipped_value_target)

            board.push(move)

        return inputs, policy_targets, value_targets

    def _process_data_entry(self, data: dict, h5_file):
        inputs = data["inputs"]
        policy_targets = data["policy_targets"]
        value_targets = data["value_targets"]
        game_length = data["game_length"]
        avg_rating = data["avg_rating"]
        game_result = data["game_result"]

        num_new_samples = len(inputs)
        if num_new_samples == 0:
            return

        self.total_samples += num_new_samples
        self.total_moves_processed += num_new_samples
        self.game_results_counter[game_result] += 1
        self._update_histograms(game_length, avg_rating)

        self.batch_inputs.extend(inputs)
        self.batch_policy_targets.extend(policy_targets)
        self.batch_value_targets.extend(value_targets)

        if len(self.batch_inputs) >= self.batch_size:
            self._write_batch_to_h5()

    def _write_batch_to_h5(self):
        try:
            batch_size = len(self.batch_inputs)
            start_idx = self.current_dataset_size
            end_idx = self.current_dataset_size + batch_size

            # Resize datasets to accommodate new data
            self.h5_inputs.resize((end_idx, 20, 8, 8))
            self.h5_policy_targets.resize((end_idx,))
            self.h5_value_targets.resize((end_idx,))

            # Write data
            self.h5_inputs[start_idx:end_idx] = np.array(self.batch_inputs, dtype=np.float32)
            self.h5_policy_targets[start_idx:end_idx] = np.array(self.batch_policy_targets, dtype=np.int64)
            self.h5_value_targets[start_idx:end_idx] = np.array(self.batch_value_targets, dtype=np.float32)

            # Update dataset size
            self.current_dataset_size += batch_size

            # Clear batches
            self.batch_inputs.clear()
            self.batch_policy_targets.clear()
            self.batch_value_targets.clear()

        except Exception as e:
            self.logger.error(f"Error writing batch to HDF5: {str(e)}")

    def _update_histograms(self, game_length: int, avg_rating: float):
        # Update game length histogram
        length_idx = np.digitize(game_length, self.game_length_bins) - 1
        if 0 <= length_idx < len(self.game_length_histogram):
            self.game_length_histogram[length_idx] += 1

        # Update player rating histogram
        if avg_rating:
            rating_idx = np.digitize(avg_rating, self.player_rating_bins) - 1
            if 0 <= rating_idx < len(self.player_rating_histogram):
                self.player_rating_histogram[rating_idx] += 1

    def _emit_stats(self):
        if self.stats_update:
            stats = {
                "total_games_processed": self.total_games_processed,
                "total_moves_processed": self.total_moves_processed,
                "game_results_counter": self.game_results_counter.copy(),
                "game_length_bins": self.game_length_bins.tolist(),
                "game_length_histogram": self.game_length_histogram.tolist(),
                "player_rating_bins": self.player_rating_bins.tolist(),
                "player_rating_histogram": self.player_rating_histogram.tolist(),
            }
            self.stats_update.emit(stats)

    def _split_dataset(self):
        try:
            h5_path = os.path.join(self.output_dir, "dataset.h5")
            train_indices_path = os.path.join(self.output_dir, "train_indices.npy")
            val_indices_path = os.path.join(self.output_dir, "val_indices.npy")
            test_indices_path = os.path.join(self.output_dir, "test_indices.npy")

            with h5py.File(h5_path, "r") as h5_file:
                num_samples = h5_file["inputs"].shape[0]
                if num_samples == 0:
                    self.logger.warning("No samples to split in the dataset.")
                    return

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

            self.logger.info("Split dataset into train/val/test sets.")

        except Exception as e:
            self.logger.error(f"Error splitting dataset: {str(e)}")