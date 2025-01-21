from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QGridLayout, QLineEdit, QPushButton, QMessageBox, QLabel
from src.data_processing.data_preparation.data_preparation_worker import DataPreparationWorker
from src.data_processing.data_preparation.data_preparation_visualization import DataPreparationVisualization
from src.base.base_tab import BaseTab
import os

class DataPreparationSubTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = DataPreparationVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.setup_subtab(
            main_layout,
            "Prepare your raw data and turn it into processed data for training.",
            "Data Preparation Progress",
            "Data Preparation Logs",
            "Data Preparation Visualization",
            self.visualization,
            {
                "start_text": "Start Data Preparation",
                "stop_text": "Stop",
                "start_callback": self.start_process,
                "stop_callback": self.stop_process,
                "pause_text": "Pause",
                "resume_text": "Resume",
                "pause_callback": self.pause_worker,
                "resume_callback": self.resume_worker,
                "start_new_callback": self.reset_to_initial_state
            }
        )

        group = QGroupBox("Data Parameters")
        layout = QGridLayout()
        label1 = QLabel("Max Games:")
        self.max_games_input = QLineEdit("60000")
        label2 = QLabel("Minimum ELO:")
        self.min_elo_input = QLineEdit("1600")
        label3 = QLabel("Batch Size:")
        self.batch_size_input = QLineEdit("4000")
        layout.addWidget(label1, 0, 0)
        layout.addWidget(self.max_games_input, 0, 1)
        layout.addWidget(label2, 0, 2)
        layout.addWidget(self.min_elo_input, 0, 3)
        layout.addWidget(label3, 1, 0)
        layout.addWidget(self.batch_size_input, 1, 1)
        group.setLayout(layout)
        self.parameters_group = group

        group = QGroupBox("PGN File Selection")
        layout = QGridLayout()
        label1 = QLabel("PGN File:")
        self.pgn_file_input = QLineEdit("")
        pgn_browse_button = QPushButton("Browse")
        pgn_browse_button.clicked.connect(lambda: self.browse_file(self.pgn_file_input, "Select PGN File", "PGN Files (*.pgn)"))
        layout.addWidget(label1, 0, 0)
        layout.addLayout(self.create_browse_layout(self.pgn_file_input, pgn_browse_button), 0, 1, 1, 3)
        group.setLayout(layout)
        self.pgn_file_group = group

        group = QGroupBox("Chess Engine Configuration")
        layout = QGridLayout()
        label_engine_path = QLabel("Engine Path:")
        self.engine_path_input = QLineEdit("")
        engine_path_button = QPushButton("Browse")
        engine_path_button.clicked.connect(lambda: self.browse_file(self.engine_path_input, "Select Engine", "All Files (*)"))

        label_engine_depth = QLabel("Depth:")
        self.engine_depth_input = QLineEdit("10")

        label_engine_threads = QLabel("Threads:")
        self.engine_threads_input = QLineEdit("8")

        label_engine_hash = QLabel("Hash (MB):")
        self.engine_hash_input = QLineEdit("1024")

        layout.addWidget(label_engine_path, 0, 0)
        layout.addLayout(self.create_browse_layout(self.engine_path_input, engine_path_button), 0, 1, 1, 3)
        layout.addWidget(label_engine_depth, 1, 0)
        layout.addWidget(self.engine_depth_input, 1, 1)
        layout.addWidget(label_engine_threads, 1, 2)
        layout.addWidget(self.engine_threads_input, 1, 3)
        layout.addWidget(label_engine_hash, 2, 0)
        layout.addWidget(self.engine_hash_input, 2, 1)
        group.setLayout(layout)
        self.engine_config_group = group

        self.layout().insertWidget(1, self.parameters_group)
        self.layout().insertWidget(2, self.pgn_file_group)
        self.layout().insertWidget(3, self.engine_config_group)

        self.stop_button.setEnabled(False)
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)

    def start_process(self):
        try:
            max_games = int(self.max_games_input.text())
            min_elo = int(self.min_elo_input.text())
            batch_size = int(self.batch_size_input.text())
            if max_games <= 0 or min_elo <= 0 or batch_size <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Max Games, Minimum ELO, and Batch Size must be positive integers.")
            return

        pgn_file = self.pgn_file_input.text().strip()

        if not os.path.isfile(pgn_file):
            QMessageBox.warning(self, "Error", "Selected PGN file does not exist.")
            return

        engine_path = self.engine_path_input.text().strip()
        if not engine_path:
            QMessageBox.warning(self, "Error", "Engine path cannot be empty.")
            return

        try:
            engine_depth = int(self.engine_depth_input.text())
            engine_threads = int(self.engine_threads_input.text())
            engine_hash = int(self.engine_hash_input.text())
            if engine_depth <= 0 or engine_threads <= 0 or engine_hash <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Depth, Threads, and Hash must be positive integers.")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        if self.pause_button:
            self.pause_button.setEnabled(True)
        if self.resume_button:
            self.resume_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()
        self.parameters_group.setVisible(False)
        self.pgn_file_group.setVisible(False)
        self.engine_config_group.setVisible(False)
        self.progress_group.setVisible(True)
        self.control_group.setVisible(True)
        self.log_group.setVisible(True)
        if self.visualization_group:
            self.visualization_group.setVisible(False)
        if self.show_logs_button:
            self.show_logs_button.setVisible(True)
        if self.show_graphs_button:
            self.show_graphs_button.setVisible(True)
        if self.start_new_button:
            self.start_new_button.setVisible(False)
        self.init_ui_state = False

        started = self.start_worker(DataPreparationWorker, pgn_file, max_games, min_elo, batch_size, engine_path, engine_depth, engine_threads, engine_hash)
        if started:
            self.worker.stats_update.connect(self.visualization.update_data_visualization)
        else:
            self.reset_to_initial_state()

    def stop_process(self):
        self.stop_worker()
        self.reset_to_initial_state()

    def reset_to_initial_state(self):
        self.parameters_group.setVisible(True)
        self.pgn_file_group.setVisible(True)
        self.engine_config_group.setVisible(True)
        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)
        if self.visualization_group:
            self.visualization_group.setVisible(False)
        if self.start_new_button:
            self.start_new_button.setVisible(False)
        if self.show_logs_button:
            self.show_logs_button.setVisible(False)
        if self.show_graphs_button:
            self.show_graphs_button.setVisible(False)
        if self.show_logs_button:
            self.show_logs_button.setChecked(True)
        if self.show_graphs_button:
            self.show_graphs_button.setChecked(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()
        if self.start_button:
            self.start_button.setEnabled(True)
        if self.stop_button:
            self.stop_button.setEnabled(False)
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)
        self.init_ui_state = True