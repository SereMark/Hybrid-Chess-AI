from PyQt5.QtWidgets import QMessageBox, QVBoxLayout, QFileDialog, QGroupBox
from PyQt5.QtCore import pyqtSignal, QThread
import numpy as np
import h5py
import os
from scripts.data_pipeline import DataProcessor, split_dataset
from src.common.base_tab import BaseTab
from src.common.common_widgets import create_labeled_input
from src.gui.visualizations.data_preparation_visualization import DataPreparationVisualization

class DataPreparationWorker(QThread):
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, raw_data_dir, processed_data_dir, max_games):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.max_games = max_games
        self._is_paused = False
        self._is_stopped = False

    def run(self):
        try:
            processor = DataProcessor(self.raw_data_dir, self.processed_data_dir, self.max_games)
            processor.process_pgn_files(progress_callback=self.emit_progress, log_callback=self.log_update.emit)
            if not self._is_stopped:
                split_dataset(self.processed_data_dir, log_callback=self.log_update.emit)
        except Exception as e:
            self.log_update.emit(f"Error: {e}")
        finally:
            self.finished.emit()

    def emit_progress(self, value):
        if not self._is_paused and not self._is_stopped:
            self.progress_update.emit(value)

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._is_stopped = True

class DataPreparationTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = DataPreparationVisualization()
        self.init_specific_ui()

    def init_specific_ui(self):
        self.max_games_label, self.max_games_input, _, max_games_layout = create_labeled_input("Max Games:", "100000")
        self.raw_data_label, self.raw_data_dir_input, self.raw_browse_button, raw_dir_layout = create_labeled_input("Raw Data Directory:", "data/raw")
        self.processed_data_label, self.processed_data_dir_input, self.processed_browse_button, processed_dir_layout = create_labeled_input("Processed Data Directory:", "data/processed")
    
        self.config_layout.addWidget(self.max_games_label)
        self.config_layout.addLayout(max_games_layout)
        self.config_layout.addWidget(self.raw_data_label)
        self.config_layout.addLayout(raw_dir_layout)
        self.config_layout.addWidget(self.processed_data_label)
        self.config_layout.addLayout(processed_dir_layout)
    
        self.raw_browse_button.clicked.connect(lambda: self.browse_dir(self.raw_data_dir_input, "Raw Data"))
        self.processed_browse_button.clicked.connect(lambda: self.browse_dir(self.processed_data_dir_input, "Processed Data"))
    
        self.buttons["start"].setText("Start Data Preparation")
        self.buttons["pause"].setText("Pause")
        self.buttons["resume"].setText("Resume")
        self.buttons["stop"].setText("Stop")
        self.buttons["pause"].setEnabled(False)
        self.buttons["resume"].setEnabled(False)
    
        self.buttons["start"].clicked.connect(self.start_data_preparation)
        self.buttons["pause"].clicked.connect(self.pause_data_preparation)
        self.buttons["resume"].clicked.connect(self.resume_data_preparation)
        self.buttons["stop"].clicked.connect(self.stop_data_preparation)
    
        self.visualization_group = QGroupBox("Data Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        self.visualization_group.setLayout(vis_layout)
        self.layout.addWidget(self.visualization_group)
    
    def browse_dir(self, line_edit, title):
        dir_path = QFileDialog.getExistingDirectory(self, f"Select {title} Directory", line_edit.text())
        if dir_path:
            line_edit.setText(dir_path)
    
    def start_data_preparation(self):
        try:
            max_games = int(self.max_games_input.text())
            if max_games <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Max Games must be a positive integer.")
            return
    
        raw_data_dir = self.raw_data_dir_input.text()
        processed_data_dir = self.processed_data_dir_input.text()
    
        if not os.path.exists(raw_data_dir):
            QMessageBox.warning(self, "Error", "Raw data directory does not exist.")
            return
        os.makedirs(processed_data_dir, exist_ok=True)
    
        self.buttons["start"].setEnabled(False)
        self.buttons["pause"].setEnabled(True)
        self.buttons["stop"].setEnabled(True)
        self.buttons["resume"].setEnabled(False)
    
        self.log_signal.emit("Starting data preparation...")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
    
        self.worker = DataPreparationWorker(raw_data_dir, processed_data_dir, max_games)
        self.worker.log_update.connect(self.log_signal.emit)
        self.worker.progress_update.connect(self.progress_signal.emit)
        self.worker.finished.connect(self.on_data_preparation_finished)
        self.worker.start()
    
    def pause_data_preparation(self):
        if self.worker and self.worker.isRunning():
            self.worker.pause()
            self.buttons["pause"].setEnabled(False)
            self.buttons["resume"].setEnabled(True)
            self.log_signal.emit("Pausing data preparation...")
    
    def resume_data_preparation(self):
        if self.worker and self.worker.isRunning():
            self.worker.resume()
            self.buttons["pause"].setEnabled(True)
            self.buttons["resume"].setEnabled(False)
            self.log_signal.emit("Resuming data preparation...")
    
    def stop_data_preparation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.buttons["start"].setEnabled(True)
            self.buttons["pause"].setEnabled(False)
            self.buttons["resume"].setEnabled(False)
            self.buttons["stop"].setEnabled(False)
            self.log_signal.emit("Stopping data preparation...")
    
    def on_data_preparation_finished(self):
        for button in self.buttons.values():
            button.setEnabled(button == self.buttons["start"])
        self.update_data_visualization()
    
    def update_data_visualization(self):
        dataset_path = os.path.join(self.processed_data_dir_input.text(), 'dataset.h5')
        if not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Error", "Processed dataset not found.")
            return
    
        try:
            with h5py.File(dataset_path, 'r') as h5_file:
                policy_targets = h5_file['policy_targets'][:]
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
    
        move_frequencies = {}
        for move_idx in policy_targets:
            move_frequencies[move_idx] = move_frequencies.get(move_idx, 0) + 1
    
        self.visualization.update_data_visualization(dataset_path)