import os
import glob
import time
import numpy as np
import h5py
import chess
import chess.pgn
from concurrent.futures import ThreadPoolExecutor
from src.gui.visualizations.data_preparation_visualization import DataPreparationVisualization
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, 
    QHBoxLayout, QProgressBar, QLabel, QTextEdit, QFileDialog, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal

class DataPreparationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = DataPreparationVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        settings_group = QGroupBox("Data Preparation Settings")
        settings_layout = QFormLayout()

        self.max_games_input = QLineEdit("100000")
        self.min_elo_input = QLineEdit("2000")
        self.raw_data_dir_input = QLineEdit("data/raw")
        self.processed_data_dir_input = QLineEdit("data/processed")

        raw_browse_button = QPushButton("Browse")
        raw_browse_button.clicked.connect(self.browse_raw_dir)
        processed_browse_button = QPushButton("Browse")
        processed_browse_button.clicked.connect(self.browse_processed_dir)

        raw_dir_layout = QHBoxLayout()
        raw_dir_layout.addWidget(self.raw_data_dir_input)
        raw_dir_layout.addWidget(raw_browse_button)

        processed_dir_layout = QHBoxLayout()
        processed_dir_layout.addWidget(self.processed_data_dir_input)
        processed_dir_layout.addWidget(processed_browse_button)

        settings_layout.addRow("Max Games:", self.max_games_input)
        settings_layout.addRow("Minimum ELO:", self.min_elo_input)
        settings_layout.addRow("Raw Data Directory:", raw_dir_layout)
        settings_layout.addRow("Processed Data Directory:", processed_dir_layout)

        settings_group.setLayout(settings_layout)

        control_buttons_layout = self.create_control_buttons()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label = QLabel("Time Left: Calculating...")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        main_layout.addWidget(settings_group)
        main_layout.addLayout(control_buttons_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.remaining_time_label)
        main_layout.addWidget(self.log_text_edit)
        main_layout.addWidget(self.create_visualization_group())

    def create_control_buttons(self):
        layout = QHBoxLayout()
        self.start_button = QPushButton("Start Data Preparation")
        self.stop_button = QPushButton("Stop")
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addStretch()

        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_data_preparation)
        self.stop_button.clicked.connect(self.stop_data_preparation)
        return layout

    def create_visualization_group(self):
        visualization_group = QGroupBox("Data Preparation Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        visualization_group.setLayout(vis_layout)
        return visualization_group

    def browse_raw_dir(self):
        self.browse_dir(self.raw_data_dir_input, "Raw Data")

    def browse_processed_dir(self):
        self.browse_dir(self.processed_data_dir_input, "Processed Data")

    def browse_dir(self, line_edit, title):
        dir_path = QFileDialog.getExistingDirectory(self, f"Select {title} Directory", line_edit.text())
        if dir_path:
            line_edit.setText(dir_path)

    def start_data_preparation(self):
        try:
            max_games = int(self.max_games_input.text())
            if max_games <= 0:
                raise ValueError
            min_elo = int(self.min_elo_input.text())
            if min_elo <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Max Games and Minimum ELO must be positive integers.")
            return

        raw_data_dir = self.raw_data_dir_input.text()
        processed_data_dir = self.processed_data_dir_input.text()
        if not os.path.exists(raw_data_dir):
            QMessageBox.warning(self, "Error", "Raw data directory does not exist.")
            return
        os.makedirs(processed_data_dir, exist_ok=True)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_text_edit.clear()
        self.log_text_edit.append("Starting data preparation...")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")

        self.visualization.reset_visualizations()

        self.worker = DataPreparationWorker(raw_data_dir, processed_data_dir, max_games, min_elo)
        self.worker.log_update.connect(self.log_text_edit.append)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.time_left_update.connect(self.update_time_left)
        self.worker.finished.connect(self.on_data_preparation_finished)
        self.worker.stats_update.connect(self.visualization.update_data_visualization)
        self.worker.start()

    def stop_data_preparation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log_text_edit.append("Stopping data preparation...")

    def on_data_preparation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Data Preparation Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_text_edit.append("Data preparation process finished.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Progress: {value}%")

    def update_time_left(self, time_left_str):
        self.remaining_time_label.setText(f"Time Left: {time_left_str}")

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
        self.batch_size = 10000
        self.batch_inputs = []
        self.batch_policy_targets = []
        self.batch_value_targets = []
        self.current_dataset_size = 0

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
            with h5py.File(h5_file_path, 'w') as h5_file:
                h5_inputs = h5_file.create_dataset(
                    'inputs', shape=(0, 20, 8, 8), maxshape=(None, 20, 8, 8),
                    dtype=np.float32, compression='gzip'
                )
                h5_policy_targets = h5_file.create_dataset(
                    'policy_targets', shape=(0,), maxshape=(None,),
                    dtype=np.int64, compression='gzip'
                )
                h5_value_targets = h5_file.create_dataset(
                    'value_targets', shape=(0,), maxshape=(None,),
                    dtype=np.float32, compression='gzip'
                )

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
                                        self._process_data_entry(result, h5_inputs, h5_policy_targets, h5_value_targets)
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
                if len(self.batch_inputs) > 0:
                    self._write_batch(h5_inputs, h5_policy_targets, h5_value_targets)
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"Error during data processing: {e}")
        finally:
            if self.log_callback:
                self.log_callback(f"Total samples collected: {self.total_samples}")

    def _process_data_entry(self, data, h5_inputs, h5_policy_targets, h5_value_targets):
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

            self.batch_inputs.extend(inputs)
            self.batch_policy_targets.extend(policy_targets)
            self.batch_value_targets.extend(value_targets)

            if len(self.batch_inputs) >= self.batch_size:
                self._write_batch(h5_inputs, h5_policy_targets, h5_value_targets)

    def _write_batch(self, h5_inputs, h5_policy_targets, h5_value_targets):
        batch_size = len(self.batch_inputs)
        h5_inputs.resize((self.current_dataset_size + batch_size, 20, 8, 8))
        h5_policy_targets.resize((self.current_dataset_size + batch_size,))
        h5_value_targets.resize((self.current_dataset_size + batch_size,))

        h5_inputs[self.current_dataset_size:self.current_dataset_size + batch_size] = np.array(self.batch_inputs, dtype=np.float32)
        h5_policy_targets[self.current_dataset_size:self.current_dataset_size + batch_size] = np.array(self.batch_policy_targets, dtype=np.int64)
        h5_value_targets[self.current_dataset_size:self.current_dataset_size + batch_size] = np.array(self.batch_value_targets, dtype=np.float32)

        self.current_dataset_size += batch_size

        self.batch_inputs = []
        self.batch_policy_targets = []
        self.batch_value_targets = []

    def _format_time_left(self, seconds):
        days = seconds // 86400
        remainder = seconds % 86400
        hours = remainder // 3600
        minutes = (remainder % 3600) // 60
        secs = remainder % 60

        if days >= 1:
            day_str = f"{int(days)}d " if days > 1 else "1d "
            return f"{day_str}{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
        else:
            return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

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
                time_left_str = self._format_time_left(time_left)
                self.time_left_callback(time_left_str)
            else:
                self.time_left_callback("Calculating...")