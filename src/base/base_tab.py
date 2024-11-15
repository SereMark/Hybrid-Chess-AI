from PyQt5.QtWidgets import QWidget, QTextEdit, QProgressBar, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QThread
import traceback

class BaseTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.worker = None
        self.log_text_edit = None
        self.progress_bar = None
        self.remaining_time_label = None

    def create_log_text_edit(self):
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        return self.log_text_edit

    def create_progress_layout(self):
        layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label = QLabel("Time Left: N/A")
        self.remaining_time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.remaining_time_label)
        return layout

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Progress: {value}%")

    def update_time_left(self, time_left_str):
        self.remaining_time_label.setText(f"Time Left: {time_left_str}")

    def log_message(self, message):
        if self.log_text_edit:
            self.log_text_edit.append(message)
        else:
            print(message)

    def start_worker(self, worker_class, *args, **kwargs):
        if self.thread is not None and self.thread.isRunning():
            self.log_message("A worker is already running.")
            return False

        try:
            self.thread = QThread()
            self.worker = worker_class(*args, **kwargs)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.finished.connect(self.on_worker_finished)
            self.worker.log_update.connect(self.log_message)
            self.worker.progress_update.connect(self.update_progress)
            self.worker.time_left_update.connect(self.update_time_left)
            self.thread.start()
            return True
        except Exception as e:
            self.log_message(f"Error starting worker: {str(e)}\n{traceback.format_exc()}")
            return False

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.log_message("Worker stop requested.")
        else:
            self.log_message("No worker to stop.")

    def on_worker_finished(self):
        self.worker = None
        self.thread = None