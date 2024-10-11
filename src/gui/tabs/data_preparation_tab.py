from PyQt5.QtWidgets import QMessageBox, QVBoxLayout, QFileDialog, QGroupBox, QHBoxLayout, QPushButton, QProgressBar, QTextEdit, QWidget, QLabel, QLineEdit
from PyQt5.QtCore import pyqtSignal, QThread
import os, numpy as np, h5py
from scripts.data_pipeline import DataProcessor, split_dataset
from src.gui.visualizations.data_preparation_visualization import DataPreparationVisualization

class DataPreparationWorker(QThread):
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, raw_data_dir, processed_data_dir, max_games):
        super().__init__()
        self.raw_data_dir, self.processed_data_dir, self.max_games = raw_data_dir, processed_data_dir, max_games
        self._is_paused, self._is_stopped = False, False

    def run(self):
        try:
            processor = DataProcessor(self.raw_data_dir, self.processed_data_dir, self.max_games)
            processor.process_pgn_files(progress_callback=self.emit_progress, log_callback=self.log_update.emit)
            if not self._is_stopped: split_dataset(self.processed_data_dir, log_callback=self.log_update.emit)
        except Exception as e: self.log_update.emit(f"Error: {e}")
        finally: self.finished.emit()

    def emit_progress(self, value):
        if not self._is_paused and not self._is_stopped: self.progress_update.emit(value)
    def pause(self): self._is_paused = True
    def resume(self): self._is_paused = False
    def stop(self): self._is_stopped = True

class DataPreparationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = DataPreparationVisualization()
        self.init_specific_ui()

    def init_specific_ui(self):
        layout = QVBoxLayout(self)
        
        self.max_games_input = QLineEdit("100000")
        self.raw_data_dir_input = QLineEdit("data/raw")
        self.processed_data_dir_input = QLineEdit("data/processed")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        
        max_games_layout = self.create_input_layout("Max Games:", self.max_games_input)
        raw_dir_layout = self.create_input_layout("Raw Data Directory:", self.raw_data_dir_input, "Browse", self.browse_raw_dir)
        processed_dir_layout = self.create_input_layout("Processed Data Directory:", self.processed_data_dir_input, "Browse", self.browse_processed_dir)
        control_buttons_layout = self.create_control_buttons()
        
        layout.addLayout(max_games_layout)
        layout.addLayout(raw_dir_layout)
        layout.addLayout(processed_dir_layout)
        layout.addLayout(control_buttons_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_text_edit)
        layout.addWidget(self.create_visualization_group())

    def create_input_layout(self, label_text, input_widget, button_text=None, button_callback=None):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        layout.addWidget(input_widget)
        if button_text and button_callback:
            button = QPushButton(button_text)
            button.clicked.connect(button_callback)
            layout.addWidget(button)
        return layout

    def create_control_buttons(self):
        layout = QHBoxLayout()
        self.start_button, self.pause_button, self.resume_button, self.stop_button = QPushButton("Start Data Preparation"), QPushButton("Pause"), QPushButton("Resume"), QPushButton("Stop")
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.resume_button)
        layout.addWidget(self.stop_button)

        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_data_preparation)
        self.pause_button.clicked.connect(self.pause_data_preparation)
        self.resume_button.clicked.connect(self.resume_data_preparation)
        self.stop_button.clicked.connect(self.stop_data_preparation)
        return layout

    def create_visualization_group(self):
        visualization_group = QGroupBox("Data Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        visualization_group.setLayout(vis_layout)
        return visualization_group

    def browse_raw_dir(self):
        self.browse_dir(self.raw_data_dir_input, "Raw Data")
    def browse_processed_dir(self):
        self.browse_dir(self.processed_data_dir_input, "Processed Data")

    def browse_dir(self, line_edit, title):
        dir_path = QFileDialog.getExistingDirectory(self, f"Select {title} Directory", line_edit.text())
        if dir_path: line_edit.setText(dir_path)

    def start_data_preparation(self):
        try:
            max_games = int(self.max_games_input.text())
            if max_games <= 0: raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Max Games must be a positive integer.")
            return

        raw_data_dir, processed_data_dir = self.raw_data_dir_input.text(), self.processed_data_dir_input.text()
        if not os.path.exists(raw_data_dir):
            QMessageBox.warning(self, "Error", "Raw data directory does not exist.")
            return
        os.makedirs(processed_data_dir, exist_ok=True)

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.log_text_edit.append("Starting data preparation...")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")

        self.worker = DataPreparationWorker(raw_data_dir, processed_data_dir, max_games)
        self.worker.log_update.connect(self.log_text_edit.append)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.on_data_preparation_finished)
        self.worker.start()

    def pause_data_preparation(self):
        if self.worker and self.worker.isRunning():
            self.worker.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)
            self.log_text_edit.append("Pausing data preparation...")

    def resume_data_preparation(self):
        if self.worker and self.worker.isRunning():
            self.worker.resume()
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.log_text_edit.append("Resuming data preparation...")

    def stop_data_preparation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.log_text_edit.append("Stopping data preparation...")

    def on_data_preparation_finished(self):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Data Preparation Finished")
        self.log_text_edit.append("Data preparation process finished.")
        self.update_data_visualization()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Progress: {value}%")

    def update_data_visualization(self):
        dataset_path = os.path.join(self.processed_data_dir_input.text(), 'dataset.h5')
        if not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Error", "Processed dataset not found.")
            return

        try:
            with h5py.File(dataset_path, 'r') as h5_file:
                value_targets = h5_file['value_targets'][:]
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load dataset: {e}")
            return

        game_results = value_targets
        results = [np.sum(game_results == val) for val in [1.0, -1.0, 0.0]]
        total_games = sum(results)
        if total_games == 0:
            QMessageBox.warning(self, "Error", "No game results found in dataset.")
            return

        self.visualization.update_data_visualization(dataset_path)