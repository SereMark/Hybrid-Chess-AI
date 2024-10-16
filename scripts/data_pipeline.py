import chess.pgn
import os
import numpy as np
import h5py
import glob
from multiprocessing import Pool, cpu_count
import time

MOVE_MAPPING = {}
INDEX_MAPPING = {}

def initialize_move_mappings():
    index = 0
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            move = chess.Move(from_sq, to_sq)
            MOVE_MAPPING[index] = move
            INDEX_MAPPING[move] = index
            index += 1
            if chess.square_rank(to_sq) in [0, 7]:
                for promo in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    move = chess.Move(from_sq, to_sq, promotion=promo)
                    MOVE_MAPPING[index] = move
                    INDEX_MAPPING[move] = index
                    index += 1
    return index

TOTAL_MOVES = initialize_move_mappings()

def convert_board_to_tensor(board):
    planes = np.zeros((20, 8, 8), dtype=np.float32)

    piece_type_indices = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    for square, piece in board.piece_map().items():
        plane_idx = piece_type_indices[(piece.piece_type, piece.color)]
        row = square // 8
        col = square % 8
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

    return planes

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
        return None

    move_count = 0
    move_frequencies = {}
    for move in game.mainline_moves():
        input_tensor = convert_board_to_tensor(board).astype(np.float32)
        try:
            move_index = INDEX_MAPPING[move]
            move_san = board.san(move)
            move_frequencies[move_san] = move_frequencies.get(move_san, 0) + 1
        except KeyError:
            board.push(move)
            continue
        data.append((input_tensor, move_index, game_result))
        board.push(move)
        move_count += 1

    game_length = move_count
    white_rating = int(game.headers.get('WhiteElo', 0))
    black_rating = int(game.headers.get('BlackElo', 0))
    avg_rating = (white_rating + black_rating) / 2 if white_rating and black_rating else None

    return data, move_frequencies, game_length, avg_rating

class DataProcessor:
    def __init__(self, raw_data_dir, processed_data_dir, max_games=None, progress_callback=None, log_callback=None,
                 stats_callback=None, time_left_callback=None, stop_callback=None, pause_callback=None):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.max_games = max_games
        self.total_samples = 0
        self.total_games_processed = 0
        self.total_moves_processed = 0
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.stats_callback = stats_callback
        self.time_left_callback = time_left_callback
        self.stop_callback = stop_callback or (lambda: False)
        self.pause_callback = pause_callback or (lambda: False)
        self.game_results_counter = {1.0: 0, -1.0: 0, 0.0: 0}
        self.move_frequencies = {}
        self.game_lengths = []
        self.player_ratings = []
        self.start_time = None

    def estimate_game_count(self, filename):
        file_size = os.path.getsize(filename)
        avg_game_size = 5000
        estimated_games = file_size // avg_game_size
        return estimated_games

    def process_pgn_files(self):
        pgn_files = glob.glob(os.path.join(self.raw_data_dir, '*.pgn'))
        os.makedirs(self.processed_data_dir, exist_ok=True)

        h5_file_path = os.path.join(self.processed_data_dir, 'dataset.h5')
        h5_file = h5py.File(h5_file_path, 'w')
        max_shape = (None, 20, 8, 8)
        inputs_dataset = h5_file.create_dataset('inputs', shape=(0, 20, 8, 8),
                                                maxshape=max_shape, dtype='float32', chunks=True)
        policy_targets_dataset = h5_file.create_dataset('policy_targets', shape=(0,),
                                                        maxshape=(None,), dtype='int64', chunks=True)
        value_targets_dataset = h5_file.create_dataset('value_targets', shape=(0,),
                                                       maxshape=(None,), dtype='float32', chunks=True)

        pool = Pool(processes=cpu_count())

        self.start_time = time.time()
        total_estimated_games = self.max_games if self.max_games else 0
        for filename in pgn_files:
            estimated_games_in_file = self.estimate_game_count(filename)
            total_estimated_games += estimated_games_in_file

        try:
            for filename in pgn_files:
                if self.max_games is not None and self.total_games_processed >= self.max_games:
                    break

                total_games = self.estimate_game_count(filename)
                if self.log_callback:
                    self.log_callback(f"Estimated total games in {filename}: {total_games}")

                with open(filename, 'r', errors='ignore') as f:
                    games_batch = []

                    if self.max_games is not None:
                        remaining_games = self.max_games - self.total_games_processed
                        games_to_process = min(total_games, remaining_games)
                    else:
                        games_to_process = total_games

                    game_counter = 0
                    while game_counter < games_to_process:
                        if self.stop_callback():
                            if self.log_callback:
                                self.log_callback("Data preparation stopped by user.")
                            return
                        while self.pause_callback():
                            if self.log_callback:
                                self.log_callback("Data preparation paused.")
                            time.sleep(0.5)
                        try:
                            game = chess.pgn.read_game(f)
                            if game is None:
                                break
                            games_batch.append(game)
                            self.total_games_processed += 1
                            game_counter += 1
                        except Exception as e:
                            if self.log_callback:
                                self.log_callback(f"Error reading game: {e}")
                            continue

                        if len(games_batch) >= 1000:
                            self._process_batch(games_batch, inputs_dataset, policy_targets_dataset,
                                                value_targets_dataset, pool)
                            games_batch = []

                            self._update_progress_and_time_left()

                    if len(games_batch) > 0:
                        self._process_batch(games_batch, inputs_dataset, policy_targets_dataset,
                                            value_targets_dataset, pool)
                        self._update_progress_and_time_left()

        finally:
            pool.close()
            pool.join()
            h5_file.close()

            if self.log_callback:
                self.log_callback(f"Total samples collected: {self.total_samples}")

    def _update_progress_and_time_left(self):
        if self.progress_callback and self.max_games:
            progress_percentage = int((self.total_games_processed / self.max_games) * 100)
            self.progress_callback(progress_percentage)

        if self.time_left_callback and self.max_games:
            elapsed_time = time.time() - self.start_time
            estimated_total_time = (elapsed_time / self.total_games_processed) * self.max_games
            time_left = estimated_total_time - elapsed_time
            time_left_str = time.strftime('%H:%M:%S', time.gmtime(time_left))
            self.time_left_callback(time_left_str)

    def _process_batch(self, games_batch, inputs_dataset, policy_targets_dataset, value_targets_dataset, pool):
        try:
            results = pool.map(process_game, games_batch)
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"Error processing batch: {e}")
            return

        batch_inputs = []
        batch_policy_targets = []
        batch_value_targets = []
        batch_move_freq = {}
        batch_game_lengths = []
        batch_player_ratings = []
        for result in results:
            if not result:
                continue
            data, move_frequencies, game_length, avg_rating = result
            if len(data) == 0:
                continue
            for input_tensor, policy_target, value_target in data:
                batch_inputs.append(input_tensor)
                batch_policy_targets.append(policy_target)
                batch_value_targets.append(value_target)
                self.total_moves_processed += 1
                self.game_results_counter[value_target] += 1
            for move_san, freq in move_frequencies.items():
                batch_move_freq[move_san] = batch_move_freq.get(move_san, 0) + freq
            batch_game_lengths.append(game_length)
            if avg_rating:
                batch_player_ratings.append(avg_rating)

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

            if self.log_callback:
                self.log_callback(f"Processed {self.total_games_processed} games, Total samples: {self.total_samples}")

            for move_san, freq in batch_move_freq.items():
                self.move_frequencies[move_san] = self.move_frequencies.get(move_san, 0) + freq
            self.game_lengths.extend(batch_game_lengths)
            self.player_ratings.extend(batch_player_ratings)

            if self.stats_callback:
                stats = {
                    'total_games_processed': self.total_games_processed,
                    'total_moves_processed': self.total_moves_processed,
                    'game_results_counter': self.game_results_counter.copy(),
                    'move_frequencies': self.move_frequencies.copy(),
                    'game_lengths': self.game_lengths.copy(),
                    'player_ratings': self.player_ratings.copy()
                }
                self.stats_callback(stats)

import time

def split_dataset(processed_data_dir, log_callback=None):
    output_dir = processed_data_dir
    h5_file_path = os.path.join(output_dir, 'dataset.h5')
    train_indices_path = os.path.join(output_dir, 'train_indices.npy')
    val_indices_path = os.path.join(output_dir, 'val_indices.npy')
    test_indices_path = os.path.join(output_dir, 'test_indices.npy')

    try:
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

        if log_callback:
            log_callback("Dataset split into training, validation, and test sets.")
    except Exception as e:
        if log_callback:
            log_callback(f"Error during dataset splitting: {e}")