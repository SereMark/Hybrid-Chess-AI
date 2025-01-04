import datetime
import threading
import traceback
from PyQt5.QtCore import QObject, pyqtSignal

class TimestampedLogger(QObject):
    log_signal = pyqtSignal(str, str)
    LOG_LEVELS = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}

    def __init__(self, log_signal=None, level='INFO', log_to_file=False, file_path='app.log', timestamp_format="%Y-%m-%d %H:%M:%S"):
        super().__init__()
        self.log_level = self.LOG_LEVELS.get(level.upper(), 20)
        self.timestamp_format = timestamp_format
        self.log_to_file = log_to_file
        self.file_path = file_path
        self.lock = threading.Lock()
        self.log_signal = log_signal if log_signal else self.log_signal
        if self.log_to_file:
            try:
                self.file = open(self.file_path, 'a', encoding='utf-8')
            except Exception as e:
                self.log_to_file = False
                self.error(f"Failed to open log file {self.file_path}: {e}")

    def _log_internal(self, level, message):
        now = datetime.datetime.now().strftime(self.timestamp_format)
        thread_name = threading.current_thread().name
        formatted_message = f"[{now}] [{thread_name}] [{level}] {message}"
        self.log_signal.emit(level, formatted_message)
        if self.log_to_file:
            with self.lock:
                try:
                    self.file.write(formatted_message + '\n')
                    self.file.flush()
                except Exception as e:
                    self.log_signal.emit('ERROR', f"Failed to write to log file: {e}")

    def _should_log(self, level):
        return self.LOG_LEVELS.get(level.upper(), 20) >= self.log_level

    def log(self, level, message):
        if self._should_log(level):
            self._log_internal(level, message)

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
        full_message = f"{message}\nException Traceback:\n{exc_info}"
        self.error(full_message)

    def set_log_level(self, level):
        self.log_level = self.LOG_LEVELS.get(level.upper(), 20)

    def close(self):
        if self.log_to_file and hasattr(self, 'file') and not self.file.closed:
            try:
                self.file.close()
            except Exception as e:
                self.log('ERROR', f"Failed to close log file: {e}")

    def __del__(self):
        self.close()