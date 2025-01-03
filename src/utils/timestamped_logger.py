import datetime
from PyQt5.QtCore import QObject

class TimestampedLogger(QObject):
    def __init__(self, log_signal):
        super().__init__()
        self.log_signal = log_signal

    def log(self, message):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_signal.emit(f"[{now}] {message}")