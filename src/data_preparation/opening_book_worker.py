from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
import chess.pgn
from collections import defaultdict
from src.utils.common_utils import should_stop, wait_if_paused, log_message, format_time_left
import time, os, json

class OpeningBookWorker(BaseWorker):
    positions_update = pyqtSignal(dict)
    
    def __init__(self, pgn_file_path, max_games, min_elo, max_opening_moves, processed_data_dir):
        super().__init__()
        self.pgn_file_path = pgn_file_path
        self.max_games = max_games
        self.min_elo = min_elo
        self.max_opening_moves = max_opening_moves
        self.processed_data_dir = processed_data_dir
        self.positions = defaultdict(
            lambda: defaultdict(lambda: {'win': 0, 'draw': 0, 'loss': 0, 'eco': '', 'name': ''})
        )
        self.game_counter = 0
        self.start_time = None

    def run_task(self):
        self.start_time = time.time()
        try:
            total_estimated_games = self._estimate_total_games()
            with open(self.pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                while True:
                    if should_stop(self._is_stopped):
                        log_message("Stopping opening book generation due to stop event.", self.log_update)
                        break
                    wait_if_paused(self._is_paused)
                    
                    if self.game_counter >= self.max_games:
                        log_message("Reached maximum number of games.", self.log_update)
                        break
                    
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    self.process_game(game)
                    self.game_counter += 1
                    
                    if self.game_counter % 1000 == 0:
                        self._update_progress_and_time_left(total_estimated_games)
                        log_message(f"Processed {self.game_counter} games so far...", self.log_update)
        
            self._update_progress_and_time_left(total_estimated_games)
            log_message(f"Processed {self.game_counter} games in total.", self.log_update)
        
            if self.positions_update:
                self.positions_update.emit({'positions': dict(self.positions)})
        
        except Exception as e:
            log_message(f"Error during opening book generation: {str(e)}", self.log_update)
        
        self.save_opening_book()

    def process_game(self, game):
        white_elo = game.headers.get('WhiteElo')
        black_elo = game.headers.get('BlackElo')
        
        if white_elo is None or black_elo is None:
            log_message("Skipped a game: Missing WhiteElo or BlackElo.", self.log_update)
            return
        
        try:
            white_elo = int(white_elo)
            black_elo = int(black_elo)
        except ValueError:
            log_message("Skipped a game: Non-integer ELO value.", self.log_update)
            return
        
        if white_elo < self.min_elo or black_elo < self.min_elo:
            return
        
        result = game.headers.get('Result', '*')
        outcome = self._determine_outcome(result)
        
        if outcome is None:
            log_message("Skipped a game: Unrecognized result format.", self.log_update)
            return
        
        eco_code = game.headers.get('ECO', '')
        opening_name = game.headers.get('Opening', '')
        board = game.board()
        move_counter = 0
        
        for move in game.mainline_moves():
            if should_stop(self._is_stopped):
                log_message("Stopping processing of current game due to stop event.", self.log_update)
                break
            wait_if_paused(self._is_paused)
            
            if move_counter >= self.max_opening_moves:
                break
            
            fen = board.fen()
            uci_move = move.uci()
            move_data = self.positions[fen][uci_move]
            
            if outcome:
                move_data[outcome] += 1
            if not move_data['eco']:
                move_data['eco'] = eco_code
            if not move_data['name']:
                move_data['name'] = opening_name
            
            board.push(move)
            move_counter += 1

    def _determine_outcome(self, result):
        if result == '1-0':
            return 'win'
        elif result == '0-1':
            return 'loss'
        elif result == '1/2-1/2':
            return 'draw'
        else:
            return None

    def _estimate_total_games(self):
        try:
            file_size = os.path.getsize(self.pgn_file_path)
            avg_game_size = 5000
            estimated_total_games = min(file_size // avg_game_size, self.max_games)
            return estimated_total_games
        except Exception as e:
            log_message(f"Error estimating total games: {str(e)}", self.log_update)
            return self.max_games

    def _update_progress_and_time_left(self, total_estimated_games):
        if self.progress_update:
            progress_percentage = int((self.game_counter / total_estimated_games) * 100)
            self.progress_update.emit(progress_percentage)
        
        if self.time_left_update:
            elapsed_time = time.time() - self.start_time
            if self.game_counter > 0:
                estimated_total_time = (elapsed_time / self.game_counter) * total_estimated_games
                time_left = estimated_total_time - elapsed_time
                time_left = max(0, time_left)
                time_left_str = format_time_left(time_left)
                self.time_left_update.emit(time_left_str)
            else:
                self.time_left_update.emit("Calculating...")

    def save_opening_book(self):
        try:
            positions = {fen: {move: stats for move, stats in moves.items()} 
                         for fen, moves in self.positions.items()}
            
            opening_book_file = os.path.join(self.processed_data_dir, 'opening_book.json')
            opening_book_file = os.path.abspath(opening_book_file)
            os.makedirs(os.path.dirname(opening_book_file), exist_ok=True)
            
            with open(opening_book_file, 'w') as f:
                json.dump(positions, f, indent=4)
            
            log_message(f"Opening book saved to {opening_book_file}", self.log_update)
        except Exception as e:
            log_message(f"Error saving opening book: {str(e)}", self.log_update)