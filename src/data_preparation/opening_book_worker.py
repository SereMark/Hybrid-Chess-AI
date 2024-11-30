from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.data_preparation.opening_book_processor import OpeningBookProcessor

class OpeningBookWorker(BaseWorker):
    positions_update = pyqtSignal(dict)

    def __init__(self, pgn_file_path, max_games, min_elo, max_opening_moves):
        super().__init__()
        self.pgn_file_path = pgn_file_path
        self.max_games = max_games
        self.min_elo = min_elo
        self.max_opening_moves = max_opening_moves
        self.processor = None

    def run_task(self):
        self.processor = OpeningBookProcessor(
            pgn_file_path=self.pgn_file_path,
            max_games=self.max_games,
            min_elo=self.min_elo,
            max_opening_moves=self.max_opening_moves,
            progress_callback=self.progress_update.emit,
            log_callback=self.log_update.emit,
            positions_callback=self.positions_update.emit,
            time_left_callback=self.time_left_update.emit,
            stop_event=self._is_stopped,
            pause_event=self._is_paused
        )
        self.processor.process_pgn_file()

    def stop(self):
        super().stop()
        if self.processor:
            self.processor.stop()