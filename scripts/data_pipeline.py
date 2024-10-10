import chess.pgn
import os
import numpy as np
import h5py
import glob
import torch
from multiprocessing import Pool, cpu_count
from src.neural_network.train import INDEX_MAPPING, TOTAL_MOVES

def convert_board_to_tensor(board):
    planes = np.zeros((20, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color = piece.color
        row = square // 8
        col = square % 8
        plane_idx = piece_type - 1
        if color == chess.BLACK:
            plane_idx += 6
        planes[plane_idx, row, col] = 1
    castling = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]
    for i, right in enumerate(castling):
        if right:
            planes[12 + i, :, :] = 1
    if board.ep_square is not None:
        row = board.ep_square // 8
        col = board.ep_square % 8
        planes[16, row, col] = 1
    planes[17, :, :] = board.halfmove_clock / 100.0
    planes[18, :, :] = board.fullmove_number / 100.0
    planes[19, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    if board.turn == chess.BLACK:
        planes = planes[:, ::-1, ::-1].copy()
    return torch.tensor(planes, dtype=torch.float32)

def process_game(game):
    board = game.board()
    data = []
    result = game.headers.get('Result', None)
    if result == '1-0':
        game_result = 1.0
    elif result == '0-1':
        game_result = -1.0
    elif result == '1/2-1/2':
        game_result = 0.0
    else:
        return data

    for move in game.mainline_moves():
        input_tensor = convert_board_to_tensor(board).numpy()
        try:
            move_index = INDEX_MAPPING [move]
        except KeyError:
            move_index = (move.from_square * 64 + move.to_square) % TOTAL_MOVES
        data.append((input_tensor, move_index, game_result))
        board.push(move)

    return data

class DataProcessor:
    def __init__(self, raw_data_dir, processed_data_dir, max_games=None):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.max_games = max_games
        self.total_samples = 0
        self.total_games_processed = 0

    def estimate_game_count(self, filename):
        file_size = os.path.getsize(filename)
        avg_game_size = 5000
        estimated_games = file_size // avg_game_size
        return estimated_games

    def process_pgn_files(self, progress_callback=None, log_callback=None):
        pgn_files = glob.glob(os.path.join(self.raw_data_dir, '*.pgn'))
        os.makedirs(self.processed_data_dir, exist_ok=True)

        h5_file_path = os.path.join(self.processed_data_dir, 'dataset.h5')
        h5_file = h5py.File(h5_file_path, 'w')
        max_shape = (None, 20, 8, 8)
        inputs_dataset = h5_file.create_dataset('inputs', shape=(0, 20, 8, 8), maxshape=max_shape, dtype='float32', chunks=True)
        policy_targets_dataset = h5_file.create_dataset('policy_targets', shape=(0,), maxshape=(None,), dtype='int64', chunks=True)
        value_targets_dataset = h5_file.create_dataset('value_targets', shape=(0,), maxshape=(None,), dtype='float32', chunks=True)

        pool = Pool(processes=cpu_count())

        try:
            for filename in pgn_files:
                if self.max_games is not None and self.total_games_processed >= self.max_games:
                    break

                total_games = self.estimate_game_count(filename)
                if log_callback:
                    log_callback(f"Estimated total games in {filename}: {total_games}")

                with open(filename, 'r', errors='ignore') as f:
                    games_batch = []

                    if self.max_games is not None:
                        remaining_games = self.max_games - self.total_games_processed
                        games_to_process = min(total_games, remaining_games)
                    else:
                        games_to_process = total_games

                    for _ in range(games_to_process):
                        try:
                            game = chess.pgn.read_game(f)
                            if game is None:
                                break
                            games_batch.append(game)
                            self.total_games_processed += 1
                        except Exception as e:
                            if log_callback:
                                log_callback(f"Error reading game: {e}")
                            continue

                        if len(games_batch) >= 1000:
                            self._process_batch(games_batch, inputs_dataset, policy_targets_dataset, value_targets_dataset, pool, progress_callback, log_callback)
                            games_batch = []

                    if len(games_batch) > 0:
                        self._process_batch(games_batch, inputs_dataset, policy_targets_dataset, value_targets_dataset, pool, progress_callback, log_callback)

        finally:
            pool.close()
            pool.join()
            h5_file.close()

            if log_callback:
                log_callback(f"Total samples collected: {self.total_samples}")

    def _process_batch(self, games_batch, inputs_dataset, policy_targets_dataset, value_targets_dataset, pool, progress_callback, log_callback):
        try:
            results = pool.map(process_game, games_batch)
        except Exception as e:
            if log_callback:
                log_callback(f"Error processing batch: {e}")
            return

        batch_inputs = []
        batch_policy_targets = []
        batch_value_targets = []
        for data in results:
            if len(data) == 0:
                continue
            for input_tensor, policy_target, value_target in data:
                batch_inputs.append(input_tensor)
                batch_policy_targets.append(policy_target)
                batch_value_targets.append(value_target)

        num_new_samples = len(batch_inputs)
        if num_new_samples > 0:
            start_index = self.total_samples
            self.total_samples += num_new_samples

            inputs_dataset.resize((self.total_samples, 20, 8, 8))
            policy_targets_dataset.resize((self.total_samples,))
            value_targets_dataset.resize((self.total_samples,))

            inputs_dataset[start_index:self.total_samples] = batch_inputs
            policy_targets_dataset[start_index:self.total_samples] = batch_policy_targets
            value_targets_dataset[start_index:self.total_samples] = batch_value_targets

            if progress_callback and self.max_games:
                progress_percentage = int((self.total_games_processed / self.max_games) * 100)
                progress_callback(progress_percentage)

            if log_callback:
                log_callback(f"Processed {self.total_games_processed} games, Total samples: {self.total_samples}")

def split_dataset(processed_data_dir, log_callback=None):
    output_dir = processed_data_dir
    h5_file_path = os.path.join(output_dir, 'dataset.h5')
    train_indices_path = os.path.join(output_dir, 'train_indices.npy')
    val_indices_path = os.path.join(output_dir, 'val_indices.npy')

    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            num_samples = h5_file['inputs'].shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            split_point = int(num_samples * 0.9)
            train_indices = indices[:split_point]
            val_indices = indices[split_point:]

            np.save(train_indices_path, train_indices)
            np.save(val_indices_path, val_indices)

        if log_callback:
            log_callback("Dataset split into training and validation sets.")
    except Exception as e:
        if log_callback:
            log_callback(f"Error during dataset splitting: {e}")