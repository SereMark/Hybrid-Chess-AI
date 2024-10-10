from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar, QTextEdit, QFileDialog
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class BaseTab(QWidget):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.config_layout = QVBoxLayout()
        self.buttons = self.init_buttons()
        buttons_layout = QHBoxLayout()
        for button in self.buttons.values():
            buttons_layout.addWidget(button)
        self.config_layout.addLayout(buttons_layout)
        self.progress_bar = QProgressBar()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.figure = plt.figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.init_visualization()
        self.layout.addLayout(self.config_layout)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.log_text_edit)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        self.log_signal.connect(self.update_log)
        self.progress_signal.connect(self.update_progress)

    def init_buttons(self):
        buttons = {name: QPushButton(name.capitalize()) for name in ["start", "pause", "resume", "stop"]}
        for button in buttons.values():
            button.setEnabled(False)
        buttons["start"].setEnabled(True)
        return buttons

    def browse_dir(self, input_field, label="Directory"):
        directory = QFileDialog.getExistingDirectory(self, f"Select {label}")
        if directory:
            input_field.setText(directory)

    def update_log(self, message):
        self.log_text_edit.append(message)
        self.progress_bar.setFormat(message)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def init_visualization(self):
        pass

    def update_visualization(self, *args, **kwargs):
        pass