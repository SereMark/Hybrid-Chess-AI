import threading
import traceback
from src.utils.logger import Logger
from PyQt5.QtCore import QObject, pyqtSignal

class BaseWorker(QObject):
    # Signals
    log_update = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)
    time_left_update = pyqtSignal(str)
    finished = pyqtSignal()
    paused = pyqtSignal(bool)
    task_finished = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Threading Events
        self._is_stopped = threading.Event()
        self._is_paused = threading.Event()
        self._is_paused.set()  # Initially not paused

        # Logger Setup
        self.logger = Logger(log_signal=self.log_update, level="DEBUG")

    def run(self):
        try:
            self.logger.info(f"Worker {self.__class__.__name__} is starting.")
            self.run_task()
            if not self._is_stopped.is_set():
                self.logger.info(f"Worker {self.__class__.__name__} completed successfully.")
                self.task_finished.emit()
            else:
                self.logger.info(f"Worker {self.__class__.__name__} stopped by user request.")
        except Exception as e:
            msg = f"Error in worker {self.__class__.__name__}: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(msg)
        finally:
            self.finished.emit()

    def pause(self):
        if not self._is_paused.is_set():
            return
        self._is_paused.clear()
        self.logger.info(f"Worker {self.__class__.__name__} paused.")
        self.paused.emit(True)

    def resume(self):
        if self._is_paused.is_set():
            return
        self._is_paused.set()
        self.logger.info(f"Worker {self.__class__.__name__} resumed.")
        self.paused.emit(False)

    def stop(self):
        self._is_stopped.set()
        self._is_paused.set()
        self.logger.info(f"Worker {self.__class__.__name__} stopped.")
        self.task_finished.emit()
        self.paused.emit(False)

    def run_task(self):
        raise NotImplementedError("Subclasses must implement the 'run_task' method.")