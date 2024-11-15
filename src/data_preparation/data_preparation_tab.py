from PyQt5.QtWidgets import (
    QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton,
    QHBoxLayout, QFileDialog, QMessageBox
)
import os
from src.data_preparation.data_preparation_visualization import DataPreparationVisualization
from src.data_preparation.data_preparation_worker import DataPreparationWorker
from src.base.base_tab import BaseTab

class DataPreparationTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = DataPreparationVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        self.parameters_group = self.create_parameters_group()
        self.directories_group = self.create_directories_group()
        control_buttons_layout = self.create_control_buttons()
        progress_layout = self.create_progress_layout()
        self.log_text_edit = self.create_log_text_edit()
        self.visualization_group = self.create_visualization_group()

        main_layout.addWidget(self.parameters_group)
        main_layout.addWidget(self.directories_group)
        main_layout.addLayout(control_buttons_layout)
        main_layout.addLayout(progress_layout)
        main_layout.addWidget(self.log_text_edit)
        main_layout.addWidget(self.visualization_group)

        self.log_text_edit.setVisible(False)
        self.visualization_group.setVisible(False)

    def create_parameters_group(self):
        parameters_group = QGroupBox("Parameters")
        parameters_layout = QFormLayout()

        self.max_games_input = QLineEdit("100000")
        self.min_elo_input = QLineEdit("2000")
        self.batch_size_input = QLineEdit("10000")

        parameters_layout.addRow("Max Games:", self.max_games_input)
        parameters_layout.addRow("Minimum ELO:", self.min_elo_input)
        parameters_layout.addRow("Batch Size:", self.batch_size_input)

        parameters_group.setLayout(parameters_layout)
        return parameters_group

    def create_directories_group(self):
        directories_group = QGroupBox("Data Directories")
        directories_layout = QFormLayout()

        self.raw_data_dir_input = QLineEdit("data/raw")
        self.processed_data_dir_input = QLineEdit("data/processed")

        raw_browse_button = QPushButton("Browse")
        raw_browse_button.clicked.connect(self.browse_raw_dir)
        processed_browse_button = QPushButton("Browse")
        processed_browse_button.clicked.connect(self.browse_processed_dir)

        raw_dir_layout = QHBoxLayout()
        raw_dir_layout.addWidget(self.raw_data_dir_input)
        raw_dir_layout.addWidget(raw_browse_button)

        processed_dir_layout = QHBoxLayout()
        processed_dir_layout.addWidget(self.processed_data_dir_input)
        processed_dir_layout.addWidget(processed_browse_button)

        directories_layout.addRow("Raw Data Directory:", raw_dir_layout)
        directories_layout.addRow("Processed Data Directory:", processed_dir_layout)

        directories_group.setLayout(directories_layout)
        return directories_group

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

    def create_visualization_group(self) -> QGroupBox:
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
            min_elo = int(self.min_elo_input.text())
            batch_size = int(self.batch_size_input.text())

            if max_games <= 0 or min_elo <= 0 or batch_size <= 0:
                raise ValueError("All numerical parameters must be positive integers.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", "Max Games, Minimum ELO, and Batch Size must be positive integers.")
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
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")

        self.visualization.reset_visualizations()

        self.parameters_group.setVisible(False)
        self.directories_group.setVisible(False)
        self.log_text_edit.setVisible(True)
        self.visualization_group.setVisible(True)

        started = self.start_worker(
            DataPreparationWorker,
            raw_data_dir,
            processed_data_dir,
            max_games,
            min_elo,
            batch_size
        )
        if started:
            self.worker.stats_update.connect(self.visualization.update_data_visualization)
            self.worker.data_preparation_finished.connect(self.on_data_preparation_finished)
        else:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.parameters_group.setVisible(True)
            self.directories_group.setVisible(True)
            self.log_text_edit.setVisible(False)
            self.visualization_group.setVisible(False)

    def stop_data_preparation(self):
        self.stop_worker()
        self.log_message("Stopping data preparation...")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.parameters_group.setVisible(True)
        self.directories_group.setVisible(True)

    def on_data_preparation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Data Preparation Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_message("Data preparation process finished.")
        self.parameters_group.setVisible(True)
        self.directories_group.setVisible(True)