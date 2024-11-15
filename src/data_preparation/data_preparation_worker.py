import traceback
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.data_preparation.data_preparation import DataProcessor, split_dataset

class DataPreparationWorker(BaseWorker):
    stats_update = pyqtSignal(dict)
    data_preparation_finished = pyqtSignal()

    def __init__(self, raw_data_dir: str, processed_data_dir: str, max_games: int, min_elo: int, batch_size: int):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.max_games = max_games
        self.min_elo = min_elo
        self.batch_size = batch_size
        self.processor = None

    def run(self):
        try:
            self.log_update.emit("Starting data preparation...")
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
                self.log_update.emit("Data preparation completed successfully.")
                self.data_preparation_finished.emit()
            else:
                self.log_update.emit("Data preparation stopped by user request.")
        except Exception as e:
            error_msg = f"Error during data preparation: {str(e)}\n{traceback.format_exc()}"
            self.log_update.emit(error_msg)
        finally:
            self.finished.emit()

    def stop(self):
        super().stop()
        if self.processor:
            self.processor.stop()