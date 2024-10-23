import os
import io
import chess.pgn
import chess
import numpy as np
import h5py
import glob
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtCore import QThread, pyqtSignal

def initialize_move_mappings():
    MOVE_MAPPING = {}
    INDEX_MAPPING = {}
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
    TOTAL_MOVES = index
    return MOVE_MAPPING, INDEX_MAPPING, TOTAL_MOVES

MOVE_MAPPING, INDEX_MAPPING, TOTAL_MOVES = initialize_move_mappings()

def flip_board(board):
    flipped_board = board.mirror()
    return flipped_board

def flip_move(move):
    flipped_move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), promotion=move.promotion)
    return flipped_move

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

    return planes

def process_game(game, min_elo):
    try:
        board = game.board()
        inputs = []
        policy_targets = []
        value_targets = []

        result = game.headers.get('Result', None)
        if result == '1-0':
            game_result = 1.0
        elif result == '0-1':
            game_result = -1.0
        elif result == '1/2-1/2':
            game_result = 0.0
        else:
            return None

        try:
            white_rating = int(game.headers.get('WhiteElo', 0))
            black_rating = int(game.headers.get('BlackElo', 0))
        except ValueError:
            return None

        if white_rating == 0 or black_rating == 0:
            return None
        avg_rating = (white_rating + black_rating) / 2
        if avg_rating < min_elo:
            return None

        move_count = 0
        for move in game.mainline_moves():
            input_tensor = convert_board_to_tensor(board).astype(np.float32)
            try:
                move_index = INDEX_MAPPING[move]
            except KeyError:
                board.push(move)
                continue
            inputs.append(input_tensor)
            policy_targets.append(move_index)
            value_targets.append(game_result)

            flipped_board = flip_board(board)
            flipped_input = convert_board_to_tensor(flipped_board).astype(np.float32)
            flipped_move = flip_move(move)
            try:
                flipped_move_index = INDEX_MAPPING[flipped_move]
            except KeyError:
                board.push(move)
                continue
            inputs.append(flipped_input)
            policy_targets.append(flipped_move_index)
            value_targets.append(game_result)

            board.push(move)
            move_count += 1

        game_length = move_count

        return {
            'inputs': inputs,
            'policy_targets': policy_targets,
            'value_targets': value_targets,
            'game_length': game_length,
            'avg_rating': avg_rating,
            'game_result': game_result
        }
    except Exception as e:
        print(f"Error processing game: {e}")
        return None

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

class DataPreparationWorker(QThread):
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    stats_update = pyqtSignal(dict)
    time_left_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, raw_data_dir, processed_data_dir, max_games, min_elo):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.max_games = max_games
        self.min_elo = min_elo
        self._is_stopped = False

    def run(self):
        try:
            processor = DataProcessor(
                self.raw_data_dir, self.processed_data_dir, self.max_games, self.min_elo,
                progress_callback=self.progress_update.emit,
                log_callback=self.log_update.emit,
                stats_callback=self.stats_update.emit,
                time_left_callback=self.time_left_update.emit,
                stop_event=lambda: self._is_stopped
            )
            processor.process_pgn_files()
            if not self._is_stopped:
                split_dataset(self.processed_data_dir, log_callback=self.log_update.emit)
        except Exception as e:
            self.log_update.emit(f"Error: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._is_stopped = True

class DataProcessor:
    def __init__(self, raw_data_dir, processed_data_dir, max_games, min_elo,
                 progress_callback=None, log_callback=None, stats_callback=None,
                 time_left_callback=None, stop_event=None):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.max_games = max_games
        self.min_elo = min_elo
        self.total_samples = 0
        self.total_games_processed = 0
        self.total_moves_processed = 0
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.stats_callback = stats_callback
        self.time_left_callback = time_left_callback
        self.stop_event = stop_event or (lambda: False)
        self.game_results_counter = {1.0: 0, -1.0: 0, 0.0: 0}
        self.game_length_bins = np.arange(0, 200, 5)
        self.game_length_histogram = np.zeros(len(self.game_length_bins)-1, dtype=int)
        self.player_rating_bins = np.arange(1000, 3000, 50)
        self.player_rating_histogram = np.zeros(len(self.player_rating_bins)-1, dtype=int)
        self.start_time = None
        self.all_inputs = []
        self.all_policy_targets = []
        self.all_value_targets = []

    def estimate_game_count(self):
        pgn_files = glob.glob(os.path.join(self.raw_data_dir, '*.pgn'))
        total_games = 0
        for filename in pgn_files:
            file_size = os.path.getsize(filename)
            avg_game_size = 5000
            estimated_games = file_size // avg_game_size
            total_games += estimated_games
        return total_games

    def game_texts_generator(self, filename):
        with open(filename, 'r', errors='ignore') as f:
            game_lines = []
            for line in f:
                if line.strip() == '' and game_lines:
                    yield ''.join(game_lines)
                    game_lines = []
                else:
                    game_lines.append(line)
            if game_lines:
                yield ''.join(game_lines)

    def process_pgn_files(self):
        pgn_files = glob.glob(os.path.join(self.raw_data_dir, '*.pgn'))
        os.makedirs(self.processed_data_dir, exist_ok=True)
        h5_file_path = os.path.join(self.processed_data_dir, 'dataset.h5')

        self.start_time = time.time()
        total_estimated_games = self.estimate_game_count()
        total_estimated_games = min(total_estimated_games, self.max_games)

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                for filename in pgn_files:
                    if self.total_games_processed >= self.max_games or self.stop_event():
                        break
                    if self.log_callback:
                        self.log_callback(f"Processing file: {filename}")
                    with open(filename, 'r', errors='ignore') as f:
                        game_iterator = iter(lambda: chess.pgn.read_game(f), None)
                        for game in game_iterator:
                            if self.total_games_processed >= self.max_games or self.stop_event():
                                break
                            future = executor.submit(process_game, game, self.min_elo)
                            try:
                                result = future.result()
                                if result:
                                    self._process_data_entry(result)
                                    self.total_games_processed += 1
                                    if self.total_games_processed % 100 == 0:
                                        self._update_progress_and_time_left(total_estimated_games)
                                        if self.stats_callback:
                                            stats = {
                                                'total_games_processed': self.total_games_processed,
                                                'total_moves_processed': self.total_moves_processed,
                                                'game_results_counter': self.game_results_counter.copy(),
                                                'game_length_bins': self.game_length_bins,
                                                'game_length_histogram': self.game_length_histogram.copy(),
                                                'player_rating_bins': self.player_rating_bins,
                                                'player_rating_histogram': self.player_rating_histogram.copy()
                                            }
                                            self.stats_callback(stats)
                            except Exception as e:
                                if self.log_callback:
                                    self.log_callback(f"Error processing game: {e}")
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"Error during data processing: {e}")
        finally:
            if self.log_callback:
                self.log_callback(f"Total samples collected: {self.total_samples}")
            self._save_data(h5_file_path)

    def _process_data_entry(self, data):
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

            idx = np.digitize(game_length, self.game_length_bins) - 1
            if 0 <= idx < len(self.game_length_histogram):
                self.game_length_histogram[idx] += 1

            if avg_rating:
                idx = np.digitize(avg_rating, self.player_rating_bins) - 1
                if 0 <= idx < len(self.player_rating_histogram):
                    self.player_rating_histogram[idx] += 1

            self.all_inputs.extend(inputs)
            self.all_policy_targets.extend(policy_targets)
            self.all_value_targets.extend(value_targets)

    def _save_data(self, h5_file_path):
        if self.log_callback:
            self.log_callback("Saving data to HDF5 file...")
        with h5py.File(h5_file_path, 'w') as h5_file:
            h5_file.create_dataset(
                'inputs', data=np.array(self.all_inputs, dtype=np.float32), compression='gzip'
            )
            h5_file.create_dataset(
                'policy_targets', data=np.array(self.all_policy_targets, dtype=np.int64), compression='gzip'
            )
            h5_file.create_dataset(
                'value_targets', data=np.array(self.all_value_targets, dtype=np.float32), compression='gzip'
            )
        if self.log_callback:
            self.log_callback("Data saved successfully.")

    def _update_progress_and_time_left(self, total_estimated_games):
        if self.progress_callback:
            progress_percentage = int((self.total_games_processed / total_estimated_games) * 100)
            self.progress_callback(progress_percentage)

        if self.time_left_callback:
            elapsed_time = time.time() - self.start_time
            if self.total_games_processed > 0:
                estimated_total_time = (elapsed_time / self.total_games_processed) * total_estimated_games
                time_left = estimated_total_time - elapsed_time
                time_left = max(0, time_left)
                time_left_str = time.strftime('%H:%M:%S', time.gmtime(time_left))
                self.time_left_callback(time_left_str)
            else:
                self.time_left_callback("Calculating...")