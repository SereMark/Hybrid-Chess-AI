from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QGridLayout, QLineEdit, QPushButton, QMessageBox, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox
from src.analysis.benchmark.benchmark_worker import BenchmarkWorker
from src.analysis.benchmark.benchmark_visualization import BenchmarkVisualization
from src.base.base_tab import BaseTab
import os

class BenchmarkSubTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = BenchmarkVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.setup_subtab(
            main_layout,
            "Benchmark games between two bots (Bot1 vs Bot2).",
            "Benchmark Progress",
            "Benchmark Logs",
            "Benchmark Visualization",
            self.visualization,
            {
                "start_text": "Start Benchmark",
                "stop_text": "Stop Benchmark",
                "start_callback": self.start_benchmark,
                "stop_callback": self.stop_benchmark,
                "pause_text": "Pause Benchmark",
                "resume_text": "Resume Benchmark",
                "pause_callback": self.pause_benchmark,
                "resume_callback": self.resume_benchmark,
                "start_new_callback": self.reset_to_initial_state
            }
        )
        self.benchmark_group = self.create_benchmark_group()
        self.layout().insertWidget(1, self.benchmark_group)
        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)
        if self.visualization_group:
            self.visualization_group.setVisible(False)
        if self.show_logs_button:
            self.show_logs_button.setVisible(False)
        if self.show_graphs_button:
            self.show_graphs_button.setVisible(False)
        if self.start_new_button:
            self.start_new_button.setVisible(False)
        if self.stop_button:
            self.stop_button.setEnabled(False)
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)

    def create_benchmark_group(self):
        group = QGroupBox("Benchmark Configuration")
        layout = QGridLayout()
        label_bot1 = QLabel("Bot1 Path:")
        self.bot1_path_input = QLineEdit("")
        self.bot1_use_mcts_checkbox = QCheckBox("Use MCTS")
        self.bot1_use_opening_book_checkbox = QCheckBox("Use Opening Book")
        browse_bot1_btn = QPushButton("Browse")
        browse_bot1_btn.clicked.connect(
            lambda: self.browse_file(
                self.bot1_path_input,
                "Select Bot1 File",
                "PTH Files (*.pth)"
            )
        )
        label_bot2 = QLabel("Bot2 Path:")
        self.bot2_path_input = QLineEdit("")
        self.bot2_use_mcts_checkbox = QCheckBox("Use MCTS")
        self.bot2_use_opening_book_checkbox = QCheckBox("Use Opening Book")
        browse_bot2_btn = QPushButton("Browse")
        browse_bot2_btn.clicked.connect(
            lambda: self.browse_file(
                self.bot2_path_input,
                "Select Bot2 File",
                "PTH Files (*.pth)"
            )
        )
        label_num_games = QLabel("Number of Games:")
        self.num_games_spin = QSpinBox()
        self.num_games_spin.setRange(1, 10000)
        self.num_games_spin.setValue(10)
        layout.addWidget(label_bot1, 0, 0)
        layout.addLayout(self.create_browse_layout(self.bot1_path_input, browse_bot1_btn), 0, 1, 1, 2)
        layout.addWidget(self.bot1_use_mcts_checkbox, 0, 3)
        layout.addWidget(self.bot1_use_opening_book_checkbox, 0, 4)
        layout.addWidget(label_bot2, 1, 0)
        layout.addLayout(self.create_browse_layout(self.bot2_path_input, browse_bot2_btn), 1, 1, 1, 2)
        layout.addWidget(self.bot2_use_mcts_checkbox, 1, 3)
        layout.addWidget(self.bot2_use_opening_book_checkbox, 1, 4)
        layout.addWidget(label_num_games, 2, 0)
        layout.addWidget(self.num_games_spin, 2, 1)
        group.setLayout(layout)
        return group

    def start_benchmark(self):
        bot1_path = self.bot1_path_input.text()
        bot2_path = self.bot2_path_input.text()
        bot1_use_mcts = self.bot1_use_mcts_checkbox.isChecked()
        bot1_use_opening_book = self.bot1_use_opening_book_checkbox.isChecked()
        bot2_use_mcts = self.bot2_use_mcts_checkbox.isChecked()
        bot2_use_opening_book = self.bot2_use_opening_book_checkbox.isChecked()
        num_games = self.num_games_spin.value()
        if not bot1_path or not os.path.exists(bot1_path):
            QMessageBox.warning(self, "Error", "Bot1 file does not exist.")
            return
        if not bot2_path or not os.path.exists(bot2_path):
            QMessageBox.warning(self, "Error", "Bot2 file does not exist.")
            return
        self.start_button.setEnabled(False)
        if self.stop_button:
            self.stop_button.setEnabled(True)
        if self.pause_button:
            self.pause_button.setEnabled(True)
        if self.resume_button:
            self.resume_button.setEnabled(False)
        self.log_text_edit.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting Benchmark...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.visualization.reset_visualization()
        self.benchmark_group.setVisible(False)
        self.progress_group.setVisible(True)
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
        started = self.start_worker(
            BenchmarkWorker,
            bot1_path,
            bot2_path,
            num_games,
            bot1_use_mcts,
            bot1_use_opening_book,
            bot2_use_mcts,
            bot2_use_opening_book
        )
        if started:
            self.worker.benchmark_update.connect(self.on_benchmark_update)
            self.worker.task_finished.connect(self.on_benchmark_finished)
            self.worker.progress_update.connect(self.update_progress)
        else:
            self.reset_to_initial_state()

    def stop_benchmark(self):
        self.stop_worker()
        self.reset_to_initial_state()

    def pause_benchmark(self):
        self.pause_worker()
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(True)
        self.progress_bar.setFormat("Benchmark Paused")

    def resume_benchmark(self):
        self.resume_worker()
        if self.pause_button:
            self.pause_button.setEnabled(True)
        if self.resume_button:
            self.resume_button.setEnabled(False)
        self.progress_bar.setFormat("Resuming Benchmark...")

    def on_benchmark_update(self, stats_dict):
        bot1_wins = stats_dict.get("bot1_wins", 0)
        bot2_wins = stats_dict.get("bot2_wins", 0)
        draws = stats_dict.get("draws", 0)
        total = stats_dict.get("total_games", 0)
        self.log_text_edit.append(
            "=== BENCHMARK RESULTS ===\n"
            f"Bot1 wins: {bot1_wins}\n"
            f"Bot2 wins: {bot2_wins}\n"
            f"Draws: {draws}\n"
            f"Out of {total} games.\n"
        )
        self.visualization.update_benchmark_visualization(bot1_wins, bot2_wins, draws, total)

    def on_benchmark_finished(self):
        if self.start_button:
            self.start_button.setEnabled(True)
        if self.stop_button:
            self.stop_button.setEnabled(False)
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)
        self.progress_bar.setFormat("Benchmark Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        if self.start_new_button:
            self.start_new_button.setVisible(True)

    def reset_to_initial_state(self):
        self.benchmark_group.setVisible(True)
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

    def show_logs_view(self):
        super().show_logs_view()

    def show_graphs_view(self):
        super().show_graphs_view()