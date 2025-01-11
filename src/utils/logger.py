import datetime
import traceback
import threading
from PyQt5.QtCore import QObject, pyqtSignal

class Logger(QObject):
    # Signals
    log_signal = pyqtSignal(str, str)

    # Log Levels
    LOG_LEVELS = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}

    def __init__(self, log_signal=None, level='INFO'):
        super().__init__()

        # Configuration
        self.log_level = self.LOG_LEVELS.get(level.upper(), 20)
        self.lock = threading.Lock()

        # Signal Setup
        self.log_signal = log_signal if log_signal else self.log_signal

    def log(self, level, message):
        # Decide if we should log
        if self.LOG_LEVELS.get(level.upper(), 20) >= self.log_level:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            thread_name = threading.current_thread().name
            formatted = f"[{now}] [{thread_name}] [{level}] {message}"

            # Emit to signal
            self.log_signal.emit(level, formatted)

    # Convenience Methods for Common Log Levels
    def debug(self, message):
        self.log('DEBUG', message)

    def info(self, message):
        self.log('INFO', message)

    def warning(self, message):
        self.log('WARNING', message)

    def error(self, message):
        self.log('ERROR', message)

    def critical(self, message):
        self.log('CRITICAL', message)

    def exception(self, message):
        exc_info = traceback.format_exc()
        full_msg = f"{message}\nException Traceback:\n{exc_info}"
        self.error(full_msg)

    def set_log_level(self, level):
        self.log_level = self.LOG_LEVELS.get(level.upper(), 20)