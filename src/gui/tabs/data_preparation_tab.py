import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar,
    QTextEdit, QLabel, QLineEdit, QFileDialog, QMessageBox, QGroupBox
)
from PyQt5.QtCore import Qt
from src.gui.visualizations.data_preparation_visualization import DataPreparationVisualization
from scripts.data_pipeline import DataPreparationWorker

class DataPreparationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = DataPreparationVisualization()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.max_games_input = QLineEdit("100000")
        self.min_elo_input = QLineEdit("2000")
        self.raw_data_dir_input = QLineEdit("data/raw")
        self.processed_data_dir_input = QLineEdit("data/processed")

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label = QLabel("Time Left: Calculating...")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        max_games_layout = self.create_input_layout("Max Games:", self.max_games_input)
        min_elo_layout = self.create_input_layout("Minimum ELO:", self.min_elo_input)
        raw_dir_layout = self.create_input_layout(
            "Raw Data Directory:", self.raw_data_dir_input, "Browse", self.browse_raw_dir
        )
        processed_dir_layout = self.create_input_layout(
            "Processed Data Directory:", self.processed_data_dir_input, "Browse", self.browse_processed_dir
        )
        control_buttons_layout = self.create_control_buttons()

        layout.addLayout(max_games_layout)
        layout.addLayout(min_elo_layout)
        layout.addLayout(raw_dir_layout)
        layout.addLayout(processed_dir_layout)
        layout.addLayout(control_buttons_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.remaining_time_label)
        layout.addWidget(self.log_text_edit)
        layout.addWidget(self.create_visualization_group())

    def create_input_layout(self, label_text, input_widget, button_text=None, button_callback=None):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(150)
        input_widget.setFixedWidth(200)
        layout.addWidget(label)
        layout.addWidget(input_widget)
        if button_text and button_callback:
            button = QPushButton(button_text)
            button.clicked.connect(button_callback)
            layout.addWidget(button)
        layout.addStretch()
        return layout

    def create_control_buttons(self):
        layout = QHBoxLayout()
        self.start_button = QPushButton("Start Data Preparation")
        self.stop_button = QPushButton("Stop")
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addStretch()

        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_data_preparation)
        self.stop_button.clicked.connect(self.stop_data_preparation)
        return layout

    def create_visualization_group(self):
        visualization_group = QGroupBox("Data Preparation Visualization")
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
        if dir_path:
            line_edit.setText(dir_path)

    def start_data_preparation(self):
        try:
            max_games = int(self.max_games_input.text())
            if max_games <= 0:
                raise ValueError
            min_elo = int(self.min_elo_input.text())
            if min_elo <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Max Games and Minimum ELO must be positive integers.")
            return

        raw_data_dir = self.raw_data_dir_input.text()
        processed_data_dir = self.processed_data_dir_input.text()
        if not os.path.exists(raw_data_dir):
            QMessageBox.warning(self, "Error", "Raw data directory does not exist.")
            return
        os.makedirs(processed_data_dir, exist_ok=True)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_text_edit.clear()
        self.log_text_edit.append("Starting data preparation...")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")

        self.visualization.reset_visualizations()

        self.worker = DataPreparationWorker(raw_data_dir, processed_data_dir, max_games, min_elo)
        self.worker.log_update.connect(self.log_text_edit.append)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.time_left_update.connect(self.update_time_left)
        self.worker.finished.connect(self.on_data_preparation_finished)
        self.worker.stats_update.connect(self.visualization.update_data_visualization)
        self.worker.start()

    def stop_data_preparation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log_text_edit.append("Stopping data preparation...")

    def on_data_preparation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Data Preparation Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_text_edit.append("Data preparation process finished.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Progress: {value}%")

    def update_time_left(self, time_left_str):
        self.remaining_time_label.setText(f"Time Left: {time_left_str}")