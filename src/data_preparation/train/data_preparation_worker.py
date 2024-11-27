from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.data_preparation.train.data_preparation import DataProcessor, split_dataset


class DataPreparationWorker(BaseWorker):
    stats_update = pyqtSignal(dict)

    def __init__(self, raw_data_dir: str, processed_data_dir: str, max_games: int, min_elo: int, batch_size: int):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.max_games = max_games
        self.min_elo = min_elo
        self.batch_size = batch_size
        self.processor = None

    def run_task(self):
        self.processor = DataProcessor(
            raw_data_dir=self.raw_data_dir,
            processed_data_dir=self.processed_data_dir,
            max_games=self.max_games,
            min_elo=self.min_elo,
            batch_size=self.batch_size,
            progress_callback=self.progress_update.emit,
            log_callback=self.log_update.emit,
            stats_callback=self.stats_update.emit,
            time_left_callback=self.time_left_update.emit,
            stop_event=self._is_stopped,
            pause_event=self._is_paused
        )
        self.processor.process_pgn_files()
        if not self._is_stopped.is_set():
            self.log_update.emit("Splitting dataset into training and validation sets...")
            split_dataset(self.processed_data_dir, log_callback=self.log_update.emit)
        else:
            self.log_update.emit("Data preparation stopped by user request.")

    def stop(self):
        super().stop()
        if self.processor:
            self.processor.stop()