import threading
from PyQt5.QtCore import QObject, pyqtSignal

class BaseWorker(QObject):
    log_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    time_left_update = pyqtSignal(str)
    finished = pyqtSignal()
    paused = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._is_stopped = threading.Event()
        self._is_paused = threading.Event()
        self._is_paused.set()

    def run(self):
        pass

    def pause(self):
        if not self._is_paused.is_set():
            return
        self._is_paused.clear()
        self.log_update.emit("Worker paused by user.")
        self.paused.emit(True)

    def resume(self):
        if self._is_paused.is_set():
            return
        self._is_paused.set()
        self.log_update.emit("Worker resumed by user.")
        self.paused.emit(False)

    def stop(self):
        self._is_stopped.set()
        self._is_paused.set()
        self.log_update.emit("Worker stopped by user.")