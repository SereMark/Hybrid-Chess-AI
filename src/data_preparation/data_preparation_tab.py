from PyQt5.QtWidgets import (
    QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, QMessageBox, QWidget, 
    QTabWidget, QLabel, QFrame, QHBoxLayout
)
from PyQt5.QtCore import Qt
from src.data_preparation.data_preparation_visualization import DataPreparationVisualization
from src.data_preparation.data_preparation_worker import DataPreparationWorker
from src.data_preparation.opening_book_worker import OpeningBookWorker
from src.data_preparation.opening_book_visualization import OpeningBookVisualization
from src.base.base_tab import BaseTab
import os

class DataProcessingTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        intro_label = QLabel("Prepare your raw data and turn it into processed data for training.")
        intro_label.setWordWrap(True)
        intro_label.setAlignment(Qt.AlignLeft)

        self.visualization = DataPreparationVisualization()

        self.parameters_group = self.create_parameters_group()
        self.directories_group = self.create_directories_group()

        self.control_group = QGroupBox("Actions")
        cg_layout = QVBoxLayout(self.control_group)
        cg_layout.setSpacing(10)
        control_buttons_layout = self.create_control_buttons(
            "Start Data Preparation",
            "Stop",
            self.start_data_preparation,
            self.stop_data_preparation,
            pause_text="Pause",
            resume_text="Resume",
            pause_callback=self.pause_worker,
            resume_callback=self.resume_worker
        )
        cg_layout.addLayout(control_buttons_layout)

        self.toggle_buttons_layout = QHBoxLayout()
        self.show_logs_button = QPushButton("Show Logs")
        self.show_logs_button.setCheckable(True)
        self.show_logs_button.setChecked(True)
        self.show_logs_button.clicked.connect(self.show_logs_view)
        self.show_graphs_button = QPushButton("Show Graphs")
        self.show_graphs_button.setCheckable(True)
        self.show_graphs_button.setChecked(False)
        self.show_graphs_button.clicked.connect(self.show_graphs_view)
        self.show_logs_button.clicked.connect(lambda: self.show_graphs_button.setChecked(not self.show_logs_button.isChecked()))
        self.show_graphs_button.clicked.connect(lambda: self.show_logs_button.setChecked(not self.show_graphs_button.isChecked()))
        self.toggle_buttons_layout.addWidget(self.show_logs_button)
        self.toggle_buttons_layout.addWidget(self.show_graphs_button)
        cg_layout.addLayout(self.toggle_buttons_layout)

        self.start_new_button = QPushButton("Start New")
        self.start_new_button.setToolTip("Start a new configuration and preparation process.")
        self.start_new_button.clicked.connect(self.reset_to_initial_state)
        cg_layout.addWidget(self.start_new_button)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        self.progress_group = QGroupBox("Data Preparation Progress")
        pg_layout = QVBoxLayout(self.progress_group)
        pg_layout.setSpacing(10)
        pg_layout.addLayout(self.create_progress_layout())

        self.log_group = QGroupBox("Data Preparation Logs")
        lg_layout = QVBoxLayout(self.log_group)
        lg_layout.setSpacing(10)
        self.log_text_edit = self.create_log_text_edit()
        lg_layout.addWidget(self.log_text_edit)
        self.log_group.setLayout(lg_layout)

        self.visualization_group = self.create_visualization_group(self.visualization, "Data Preparation Visualization")

        main_layout.addWidget(intro_label)
        main_layout.addWidget(self.parameters_group)
        main_layout.addWidget(self.directories_group)
        main_layout.addWidget(self.control_group)
        main_layout.addWidget(separator)
        main_layout.addWidget(self.progress_group)
        main_layout.addWidget(self.log_group)
        main_layout.addWidget(self.visualization_group)
        self.setLayout(main_layout)

        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)
        self.visualization_group.setVisible(False)
        self.show_logs_button.setVisible(False)
        self.show_graphs_button.setVisible(False)
        self.start_new_button.setVisible(False)
        self.stop_button.setEnabled(False)
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(False)
        if hasattr(self, 'resume_button'):
            self.resume_button.setEnabled(False)

    def reset_to_initial_state(self):
        self.parameters_group.setVisible(True)
        self.directories_group.setVisible(True)
        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)
        self.visualization_group.setVisible(False)
        self.start_new_button.setVisible(False)
        self.show_logs_button.setVisible(False)
        self.show_graphs_button.setVisible(False)
        self.show_logs_button.setChecked(True)
        self.show_graphs_button.setChecked(False)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(False)
        if hasattr(self, 'resume_button'):
            self.resume_button.setEnabled(False)

        self.init_ui_state = True

    def show_logs_view(self):
        if self.show_logs_button.isChecked():
            self.show_graphs_button.setChecked(False)
            self.log_group.setVisible(True)
            self.visualization_group.setVisible(False)
            self.showing_logs = True
            self.visualization.update_visualization()

    def show_graphs_view(self):
        if self.show_graphs_button.isChecked():
            self.show_logs_button.setChecked(False)
            self.log_group.setVisible(False)
            self.visualization_group.setVisible(True)
            self.showing_logs = False
            self.visualization.update_visualization()

    def create_parameters_group(self):
        parameters_group = QGroupBox("Data Parameters")
        parameters_layout = QFormLayout()
        parameters_layout.setSpacing(10)

        self.max_games_input = QLineEdit("100000")
        self.max_games_input.setPlaceholderText("e.g. 100000")
        self.max_games_input.setToolTip("Maximum number of games to process.")

        self.min_elo_input = QLineEdit("2000")
        self.min_elo_input.setPlaceholderText("e.g. 2000")
        self.min_elo_input.setToolTip("Minimum player ELO rating to consider.")

        self.batch_size_input = QLineEdit("10000")
        self.batch_size_input.setPlaceholderText("e.g. 10000")
        self.batch_size_input.setToolTip("Number of games to process in one batch.")

        parameters_layout.addRow("Max Games:", self.max_games_input)
        parameters_layout.addRow("Minimum ELO:", self.min_elo_input)
        parameters_layout.addRow("Batch Size:", self.batch_size_input)

        parameters_group.setLayout(parameters_layout)
        return parameters_group

    def create_directories_group(self):
        directories_group = QGroupBox("Data Directories")
        directories_layout = QFormLayout()
        directories_layout.setSpacing(10)

        self.raw_data_dir_input = QLineEdit("data/raw")
        self.raw_data_dir_input.setPlaceholderText("Path to raw data directory")
        self.raw_data_dir_input.setToolTip("Directory containing raw input data (PGN files).")

        self.processed_data_dir_input = QLineEdit("data/processed")
        self.processed_data_dir_input.setPlaceholderText("Path to processed data directory")
        self.processed_data_dir_input.setToolTip("Directory where processed data will be saved.")

        raw_browse_button = QPushButton("Browse")
        raw_browse_button.setToolTip("Select the raw data directory.")
        raw_browse_button.clicked.connect(lambda: self.browse_dir(self.raw_data_dir_input, "Select Raw Data Directory"))

        processed_browse_button = QPushButton("Browse")
        processed_browse_button.setToolTip("Select the processed data directory.")
        processed_browse_button.clicked.connect(lambda: self.browse_dir(self.processed_data_dir_input, "Select Processed Data Directory"))

        directories_layout.addRow("Raw Data Directory:", self.create_browse_layout(self.raw_data_dir_input, raw_browse_button))
        directories_layout.addRow("Processed Data Directory:", self.create_browse_layout(self.processed_data_dir_input, processed_browse_button))

        directories_group.setLayout(directories_layout)
        return directories_group

    def start_data_preparation(self):
        try:
            max_games = int(self.max_games_input.text())
            min_elo = int(self.min_elo_input.text())
            batch_size = int(self.batch_size_input.text())
            if max_games <= 0 or min_elo <= 0 or batch_size <= 0:
                raise ValueError("All numerical parameters must be positive integers.")
        except ValueError:
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
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(True)
        if hasattr(self, 'resume_button'):
            self.resume_button.setEnabled(False)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()

        self.parameters_group.setVisible(False)
        self.directories_group.setVisible(False)
        self.progress_group.setVisible(True)
        self.control_group.setVisible(True)
        self.log_group.setVisible(True)
        self.visualization_group.setVisible(False)
        self.show_logs_button.setVisible(True)
        self.show_graphs_button.setVisible(True)
        self.start_new_button.setVisible(False)

        self.init_ui_state = False

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
        else:
            self.reset_to_initial_state()

    def stop_data_preparation(self):
        self.stop_worker()
        self.reset_to_initial_state()

    def on_data_preparation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(False)
        if hasattr(self, 'resume_button'):
            self.resume_button.setEnabled(False)
        self.progress_bar.setFormat("Data Preparation Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.start_new_button.setVisible(True)


class OpeningBookTab(BaseTab):
    def __init__(self, parent=None, processed_data_dir_default="data/processed"):
        super().__init__(parent)
        self.processed_data_dir_default = processed_data_dir_default
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        intro_label = QLabel("Generate an Opening Book from processed PGN files.")
        intro_label.setWordWrap(True)
        intro_label.setAlignment(Qt.AlignLeft)
        intro_label.setToolTip("Generate an opening book from a large PGN dataset.")

        self.visualization = OpeningBookVisualization()
        self.opening_book_group = self.create_opening_book_group()

        self.control_group = QGroupBox("Actions")
        cg_layout = QVBoxLayout(self.control_group)
        cg_layout.setSpacing(10)
        control_buttons_layout = self.create_control_buttons(
            "Start Opening Book Generation",
            "Stop",
            self.start_opening_book_generation,
            self.stop_opening_book_generation,
            pause_text="Pause",
            resume_text="Resume",
            pause_callback=self.pause_worker,
            resume_callback=self.resume_worker
        )
        cg_layout.addLayout(control_buttons_layout)

        self.toggle_buttons_layout = QHBoxLayout()
        self.show_logs_button = QPushButton("Show Logs")
        self.show_logs_button.setCheckable(True)
        self.show_logs_button.setChecked(True)
        self.show_logs_button.clicked.connect(self.show_logs_view)
        self.show_graphs_button = QPushButton("Show Graphs")
        self.show_graphs_button.setCheckable(True)
        self.show_graphs_button.setChecked(False)
        self.show_graphs_button.clicked.connect(self.show_graphs_view)
        self.show_logs_button.clicked.connect(lambda: self.show_graphs_button.setChecked(not self.show_logs_button.isChecked()))
        self.show_graphs_button.clicked.connect(lambda: self.show_logs_button.setChecked(not self.show_graphs_button.isChecked()))
        self.toggle_buttons_layout.addWidget(self.show_logs_button)
        self.toggle_buttons_layout.addWidget(self.show_graphs_button)
        cg_layout.addLayout(self.toggle_buttons_layout)

        self.start_new_button = QPushButton("Start New")
        self.start_new_button.setToolTip("Start a new configuration and generation process.")
        self.start_new_button.clicked.connect(self.reset_to_initial_state)
        cg_layout.addWidget(self.start_new_button)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        self.progress_group = QGroupBox("Opening Book Generation Progress")
        pg_layout = QVBoxLayout(self.progress_group)
        pg_layout.setSpacing(10)
        pg_layout.addLayout(self.create_progress_layout())

        self.log_group = QGroupBox("Opening Book Generation Logs")
        lg_layout = QVBoxLayout(self.log_group)
        lg_layout.setSpacing(10)
        self.log_text_edit = self.create_log_text_edit()
        lg_layout.addWidget(self.log_text_edit)
        self.log_group.setLayout(lg_layout)

        self.visualization_group = self.create_visualization_group(self.visualization, "Opening Book Visualization")

        main_layout.addWidget(intro_label)
        main_layout.addWidget(self.opening_book_group)
        main_layout.addWidget(self.control_group)
        main_layout.addWidget(separator)
        main_layout.addWidget(self.progress_group)
        main_layout.addWidget(self.log_group)
        main_layout.addWidget(self.visualization_group)

        self.setLayout(main_layout)

        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)
        self.visualization_group.setVisible(False)
        self.show_logs_button.setVisible(False)
        self.show_graphs_button.setVisible(False)
        self.start_new_button.setVisible(False)
        self.stop_button.setEnabled(False)
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(False)
        if hasattr(self, 'resume_button'):
            self.resume_button.setEnabled(False)

    def reset_to_initial_state(self):
        self.opening_book_group.setVisible(True)
        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)
        self.visualization_group.setVisible(False)
        self.start_new_button.setVisible(False)
        self.show_logs_button.setVisible(False)
        self.show_graphs_button.setVisible(False)
        self.show_logs_button.setChecked(True)
        self.show_graphs_button.setChecked(False)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(False)
        if hasattr(self, 'resume_button'):
            self.resume_button.setEnabled(False)
        self.init_ui_state = True

    def show_logs_view(self):
        if self.show_logs_button.isChecked():
            self.show_graphs_button.setChecked(False)
            self.log_group.setVisible(True)
            self.visualization_group.setVisible(False)
            self.showing_logs = True

    def show_graphs_view(self):
        if self.show_graphs_button.isChecked():
            self.show_logs_button.setChecked(False)
            self.log_group.setVisible(False)
            self.visualization_group.setVisible(True)
            self.showing_logs = False

    def create_opening_book_group(self):
        opening_book_group = QGroupBox("Opening Book Configuration")
        opening_book_layout = QFormLayout()
        opening_book_layout.setSpacing(10)

        self.pgn_file_input = QLineEdit("data/raw/lichess_db_standard_rated_2024-11.pgn")
        self.pgn_file_input.setPlaceholderText("Path to PGN file")
        self.pgn_file_input.setToolTip("PGN file containing raw chess games.")

        pgn_browse_button = QPushButton("Browse")
        pgn_browse_button.setToolTip("Select the PGN file.")
        pgn_browse_button.clicked.connect(lambda: self.browse_file(self.pgn_file_input, "Select PGN File", "PGN Files (*.pgn)"))

        self.ob_max_games_input = QLineEdit("100000")
        self.ob_max_games_input.setPlaceholderText("e.g. 100000")
        self.ob_max_games_input.setToolTip("Maximum number of games to consider for the opening book.")

        self.ob_min_elo_input = QLineEdit("1500")
        self.ob_min_elo_input.setPlaceholderText("e.g. 1500")
        self.ob_min_elo_input.setToolTip("Minimum ELO rating for included games.")

        self.ob_max_opening_moves_input = QLineEdit("20")
        self.ob_max_opening_moves_input.setPlaceholderText("e.g. 20")
        self.ob_max_opening_moves_input.setToolTip("Maximum depth of the opening moves.")

        self.ob_processed_data_dir_input = QLineEdit(self.processed_data_dir_default)
        self.ob_processed_data_dir_input.setPlaceholderText("Path to processed data directory")
        self.ob_processed_data_dir_input.setToolTip("Directory to store the generated opening book data.")

        ob_processed_browse_button = QPushButton("Browse")
        ob_processed_browse_button.setToolTip("Select the processed data directory.")
        ob_processed_browse_button.clicked.connect(lambda: self.browse_dir(self.ob_processed_data_dir_input, "Select Processed Data Directory"))

        opening_book_layout.addRow("PGN File:", self.create_browse_layout(self.pgn_file_input, pgn_browse_button))
        opening_book_layout.addRow("Max Games:", self.ob_max_games_input)
        opening_book_layout.addRow("Minimum ELO:", self.ob_min_elo_input)
        opening_book_layout.addRow("Max Opening Moves:", self.ob_max_opening_moves_input)
        opening_book_layout.addRow("Processed Data Directory:", self.create_browse_layout(self.ob_processed_data_dir_input, ob_processed_browse_button))

        opening_book_group.setLayout(opening_book_layout)
        return opening_book_group

    def start_opening_book_generation(self):
        try:
            pgn_file_path = self.pgn_file_input.text()
            max_games = int(self.ob_max_games_input.text())
            min_elo = int(self.ob_min_elo_input.text())
            max_opening_moves = int(self.ob_max_opening_moves_input.text())

            if not os.path.exists(pgn_file_path):
                QMessageBox.warning(self, "Input Error", "The specified PGN file does not exist.")
                return

            if max_games <= 0 or min_elo <= 0 or max_opening_moves <= 0:
                raise ValueError("All numerical parameters must be positive integers.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", "Please ensure all inputs are valid positive integers.")
            return

        processed_data_dir = self.ob_processed_data_dir_input.text()
        if not os.path.exists(processed_data_dir):
            try:
                os.makedirs(processed_data_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Directory Error", f"Failed to create processed data directory: {str(e)}")
                return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(True)
        if hasattr(self, 'resume_button'):
            self.resume_button.setEnabled(False)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()

        self.opening_book_group.setVisible(False)
        self.progress_group.setVisible(True)
        self.control_group.setVisible(True)
        self.log_group.setVisible(True)
        self.visualization_group.setVisible(False)
        self.show_logs_button.setVisible(True)
        self.show_graphs_button.setVisible(True)
        self.start_new_button.setVisible(False)

        self.init_ui_state = False

        started = self.start_worker(
            OpeningBookWorker,
            pgn_file_path,
            max_games,
            min_elo,
            max_opening_moves,
            processed_data_dir
        )
        if not started:
            self.reset_to_initial_state()

    def stop_opening_book_generation(self):
        self.stop_worker()
        self.reset_to_initial_state()

    def on_opening_book_generation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(False)
        if hasattr(self, 'resume_button'):
            self.resume_button.setEnabled(False)
        self.progress_bar.setFormat("Opening Book Generation Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.start_new_button.setVisible(True)


class DataPreparationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setTabToolTip(0, "Convert raw chess data into a processed format.")
        self.tab_widget.setTabToolTip(1, "Create an opening book from processed PGN data.")
        self.tab_widget.setMovable(False)
        self.tab_widget.setElideMode(Qt.ElideRight)

        self.data_processing_tab = DataProcessingTab()
        self.opening_book_tab = OpeningBookTab()

        self.tab_widget.addTab(self.data_processing_tab, "Data Processing")
        self.tab_widget.addTab(self.opening_book_tab, "Opening Book Generation")

        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)