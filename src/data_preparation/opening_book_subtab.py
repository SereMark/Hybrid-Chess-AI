from PyQt5.QtWidgets import QVBoxLayout, QLabel, QGridLayout, QLineEdit, QPushButton, QGroupBox, QMessageBox
from PyQt5.QtCore import Qt
from src.data_preparation.opening_book_visualization import OpeningBookVisualization
from src.data_preparation.opening_book_worker import OpeningBookWorker
from src.base.base_tab import BaseTab
import os

class OpeningBookSubTab(BaseTab):
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
            self.start_process,
            self.stop_process,
            pause_text="Pause",
            resume_text="Resume",
            pause_callback=self.pause_worker,
            resume_callback=self.resume_worker
        )
        cg_layout.addLayout(control_buttons_layout)
        self.toggle_buttons_layout = self.create_log_graph_buttons(self.show_logs_view, self.show_graphs_view, "Show Logs", "Show Graphs", True)
        cg_layout.addLayout(self.toggle_buttons_layout)
        self.start_new_button = self.create_start_new_button("Start New", self.reset_to_initial_state)
        cg_layout.addWidget(self.start_new_button)
        separator = self.create_separator()
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
        group = QGroupBox("Opening Book Configuration")
        layout = QGridLayout()
        label1 = QLabel("PGN File:")
        self.pgn_file_input = QLineEdit("data/raw/lichess_db_standard_rated_2024-11.pgn")
        pgn_browse_button = QPushButton("Browse")
        pgn_browse_button.clicked.connect(lambda: self.browse_file(self.pgn_file_input, "Select PGN File", "PGN Files (*.pgn)"))
        label2 = QLabel("Max Games:")
        self.ob_max_games_input = QLineEdit("100000")
        label3 = QLabel("Minimum ELO:")
        self.ob_min_elo_input = QLineEdit("1500")
        label4 = QLabel("Max Opening Moves:")
        self.ob_max_opening_moves_input = QLineEdit("20")
        label5 = QLabel("Processed Data Directory:")
        self.ob_processed_data_dir_input = QLineEdit(self.processed_data_dir_default)
        ob_processed_browse_button = QPushButton("Browse")
        ob_processed_browse_button.clicked.connect(lambda: self.browse_dir(self.ob_processed_data_dir_input, "Select Processed Data Directory"))
        layout.addWidget(label1, 0, 0)
        layout.addLayout(self.create_browse_layout(self.pgn_file_input, pgn_browse_button), 0, 1, 1, 3)
        layout.addWidget(label2, 1, 0)
        layout.addWidget(self.ob_max_games_input, 1, 1)
        layout.addWidget(label3, 1, 2)
        layout.addWidget(self.ob_min_elo_input, 1, 3)
        layout.addWidget(label4, 2, 0)
        layout.addWidget(self.ob_max_opening_moves_input, 2, 1)
        layout.addWidget(label5, 3, 0)
        layout.addLayout(self.create_browse_layout(self.ob_processed_data_dir_input, ob_processed_browse_button), 3, 1, 1, 3)
        group.setLayout(layout)
        return group

    def start_process(self):
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
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please ensure all inputs are valid positive integers.")
            return
        processed_data_dir = self.ob_processed_data_dir_input.text()
        if not os.path.exists(processed_data_dir):
            try:
                os.makedirs(processed_data_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Directory Error", str(e))
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
        started = self.start_worker(OpeningBookWorker, pgn_file_path, max_games, min_elo, max_opening_moves, processed_data_dir)
        if started:
            self.worker.positions_update.connect(self.visualization.update_opening_book)
        else:
            self.reset_to_initial_state()

    def stop_process(self):
        self.stop_worker()
        self.reset_to_initial_state()

    def on_process_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(False)
        if hasattr(self, 'resume_button'):
            self.resume_button.setEnabled(False)
        self.progress_bar.setFormat("Opening Book Generation Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.start_new_button.setVisible(True)