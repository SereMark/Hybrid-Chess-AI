import chess.pgn
from collections import defaultdict
from src.utils.common_utils import should_stop, wait_if_paused, log_message, format_time_left
import time
import threading
import os

class OpeningBookProcessor:
    def __init__(
        self,
        pgn_file_path,
        max_games,
        min_elo,
        max_opening_moves,
        progress_callback=None,
        log_callback=None,
        positions_callback=None,
        time_left_callback=None,
        stop_event=None,
        pause_event=None
    ):
        self.pgn_file_path = pgn_file_path
        self.max_games = max_games
        self.min_elo = min_elo
        self.max_opening_moves = max_opening_moves
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.positions_callback = positions_callback
        self.time_left_callback = time_left_callback
        self.stop_event = stop_event or threading.Event()
        self.pause_event = pause_event or threading.Event()
        self.pause_event.set()

        self.positions = defaultdict(lambda: defaultdict(lambda: {'win': 0, 'draw': 0, 'loss': 0, 'eco': '', 'name': ''}))
        self.game_counter = 0
        self.start_time = None

    def process_pgn_file(self):
        self.start_time = time.time()
        try:
            total_estimated_games = self._estimate_total_games()
            with open(self.pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                while True:
                    if should_stop(self.stop_event):
                        log_message("Stopping opening book generation due to stop event.", self.log_callback)
                        break
                    wait_if_paused(self.pause_event)
                    if self.game_counter >= self.max_games:
                        log_message("Reached maximum number of games.", self.log_callback)
                        break
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    self.process_game(game)
                    self.game_counter += 1
                    if self.game_counter % 1000 == 0:
                        self._update_progress_and_time_left(total_estimated_games)
                        log_message(f"Processed {self.game_counter} games so far...", self.log_callback)
                self._update_progress_and_time_left(total_estimated_games)
                log_message(f"Processed {self.game_counter} games in total.", self.log_callback)
                if self.positions_callback:
                    self.positions_callback({'positions': dict(self.positions)})
        except Exception as e:
            log_message(f"Error during opening book generation: {str(e)}", self.log_callback)
        
        self.save_opening_book()

    def process_game(self, game):
        white_elo = game.headers.get('WhiteElo')
        black_elo = game.headers.get('BlackElo')
        if white_elo is None or black_elo is None:
            log_message("Skipped a game: Missing WhiteElo or BlackElo.", self.log_callback)
            return
        try:
            white_elo = int(white_elo)
            black_elo = int(black_elo)
        except ValueError:
            log_message("Skipped a game: Non-integer ELO value.", self.log_callback)
            return
        if white_elo < self.min_elo or black_elo < self.min_elo:
            return
        result = game.headers.get('Result', '*')
        outcome = None
        if result == '1-0':
            outcome = 'win'
        elif result == '0-1':
            outcome = 'loss'
        elif result == '1/2-1/2':
            outcome = 'draw'
        else:
            log_message("Skipped a game: Unrecognized result format.", self.log_callback)
            return
        eco_code = game.headers.get('ECO', '')
        opening_name = game.headers.get('Opening', '')
        board = game.board()
        move_counter = 0
        for move in game.mainline_moves():
            if should_stop(self.stop_event):
                log_message("Stopping processing of current game due to stop event.", self.log_callback)
                break
            wait_if_paused(self.pause_event)
            if move_counter >= self.max_opening_moves:
                break
            fen = ' '.join(board.fen().split(' ')[:4])
            san = board.san(move)
            move_data = self.positions[fen][san]
            if outcome:
                move_data[outcome] += 1
            if not move_data['eco']:
                move_data['eco'] = eco_code
            if not move_data['name']:
                move_data['name'] = opening_name
            board.push(move)
            move_counter += 1

    def _estimate_total_games(self):
        try:
            file_size = os.path.getsize(self.pgn_file_path)
            avg_game_size = 5000
            estimated_total_games = min(file_size // avg_game_size, self.max_games)
            return estimated_total_games
        except Exception as e:
            log_message(f"Error estimating total games: {str(e)}", self.log_callback)
            return self.max_games

    def _update_progress_and_time_left(self, total_estimated_games):
        if self.progress_callback:
            progress_percentage = int((self.game_counter / total_estimated_games) * 100)
            self.progress_callback(progress_percentage)
        if self.time_left_callback:
            elapsed_time = time.time() - self.start_time
            if self.game_counter > 0:
                estimated_total_time = (elapsed_time / self.game_counter) * total_estimated_games
                time_left = estimated_total_time - elapsed_time
                time_left = max(0, time_left)
                time_left_str = format_time_left(time_left)
                self.time_left_callback(time_left_str)
            else:
                self.time_left_callback("Calculating...")

    def save_opening_book(self):
        positions = {k: dict(v) for k, v in self.positions.items()}
        for k in positions:
            positions[k] = {m: dict(stats) for m, stats in positions[k].items()}
        import json
        opening_book_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'processed', 'opening_book.json')
        opening_book_file = os.path.abspath(opening_book_file)
        os.makedirs(os.path.dirname(opening_book_file), exist_ok=True)
        with open(opening_book_file, 'w') as f:
            json.dump(positions, f)

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
        if self.pause_event:
            self.pause_event.set()