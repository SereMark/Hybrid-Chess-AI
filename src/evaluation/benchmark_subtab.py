from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QGridLayout, QLineEdit, QPushButton, QMessageBox, QLabel, QSpinBox, QDoubleSpinBox
from src.evaluation.benchmark_worker import BenchmarkWorker
from src.evaluation.benchmark_visualization import BenchmarkVisualization
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
            "Benchmark games against another engine or model.",
            "Benchmark Progress",
            "Benchmark Logs",
            "Benchmark Visualization",
            self.visualization,
            {
                "start_text": "Start Benchmark",
                "stop_text": "Stop Benchmark",
                "start_callback": self.start_benchmark,
                "stop_callback": self.stop_benchmark,
                "pause_text": "Pause",
                "resume_text": "Resume",
                "pause_callback": self.pause_worker,
                "resume_callback": self.resume_worker,
                "start_new_callback": self.reset_to_initial_state
            },
            "Start New Benchmark",
            spacing=10
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

    def create_benchmark_group(self):
        group = QGroupBox("Benchmark Configuration")
        layout = QGridLayout()
        label_engine = QLabel("Engine/Model Path:")
        self.engine_path_input = QLineEdit("path/to/engine_or_model")
        browse_engine_btn = QPushButton("Browse")
        browse_engine_btn.clicked.connect(lambda: self.browse_file(self.engine_path_input, "Select Engine/Model File", "All Files (*.*)"))
        label_num_games = QLabel("Number of Games:")
        self.num_games_spin = QSpinBox()
        self.num_games_spin.setRange(1, 10000)
        self.num_games_spin.setValue(10)
        label_time_per_move = QLabel("Time/Move (sec):")
        self.time_per_move_spin = QDoubleSpinBox()
        self.time_per_move_spin.setRange(0.1, 600.0)
        self.time_per_move_spin.setValue(1.0)
        self.time_per_move_spin.setSingleStep(0.1)
        layout.addWidget(label_engine, 0, 0)
        layout.addLayout(self.create_browse_layout(self.engine_path_input, browse_engine_btn), 0, 1, 1, 3)
        layout.addWidget(label_num_games, 1, 0)
        layout.addWidget(self.num_games_spin, 1, 1)
        layout.addWidget(label_time_per_move, 2, 0)
        layout.addWidget(self.time_per_move_spin, 2, 1)
        group.setLayout(layout)
        return group

    def start_benchmark(self):
        engine_path = self.engine_path_input.text()
        if not engine_path or not os.path.exists(engine_path):
            QMessageBox.warning(self, "Error", "Engine/Model file does not exist.")
            return
        num_games = self.num_games_spin.value()
        time_per_move = self.time_per_move_spin.value()
        self.log_text_edit.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting Benchmark...")
        self.remaining_time_label.setText("Time Left: Calculating...")
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
        if self.start_button:
            self.start_button.setEnabled(False)
        if self.stop_button:
            self.stop_button.setEnabled(True)
        started = self.start_worker(BenchmarkWorker, engine_path, num_games, time_per_move)
        if started:
            self.worker.benchmark_update.connect(self.on_benchmark_update)
        else:
            self.reset_to_initial_state()

    def stop_benchmark(self):
        self.stop_worker()
        self.reset_to_initial_state()

    def on_benchmark_update(self, stats_dict):
        engine_wins = stats_dict.get("engine_wins", 0)
        our_model_wins = stats_dict.get("our_model_wins", 0)
        draws = stats_dict.get("draws", 0)
        total = stats_dict.get("total_games", 0)
        self.log_text_edit.append(
            "=== BENCHMARK RESULTS ===\n"
            "Engine wins: {}\n"
            "OurModel wins: {}\n"
            "Draws: {}\n"
            "Out of {} games.\n".format(engine_wins, our_model_wins, draws, total)
        )
        self.visualization.update_benchmark_visualization(engine_wins, our_model_wins, draws, total)
        if self.start_new_button:
            self.start_new_button.setVisible(True)

    def reset_to_initial_state(self):
        self.benchmark_group.setVisible(True)
        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)
        if self.visualization_group:
            self.visualization_group.setVisible(False)
        self.controls_group = getattr(self, 'control_group', None)
        if self.controls_group:
            self.controls_group.setVisible(True)
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

    def show_logs_view(self):
        super().show_logs_view()

    def show_graphs_view(self):
        super().show_graphs_view()