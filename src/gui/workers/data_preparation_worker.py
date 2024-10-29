from PyQt5.QtCore import QThread, pyqtSignal
import traceback

from src.data.data_pipeline import DataProcessor, split_dataset


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
            error_msg = f"Error during data preparation: {str(e)}\n{traceback.format_exc()}"
            self.log_update.emit(error_msg)
        finally:
            self.finished.emit()

    def stop(self):
        self._is_stopped = True