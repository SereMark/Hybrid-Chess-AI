from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QGridLayout, QLineEdit, QPushButton, QMessageBox, QLabel, QSpinBox, QDoubleSpinBox
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
            "Benchmark games between two custom models (Model1 vs Model2).",
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
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)

    def create_benchmark_group(self):
        group = QGroupBox("Benchmark Configuration")
        layout = QGridLayout()
        label_model1 = QLabel("Model1 Path:")
        self.model1_path_input = QLineEdit("path/to/model1")
        browse_model1_btn = QPushButton("Browse")
        browse_model1_btn.clicked.connect(lambda: self.browse_file(self.model1_path_input, "Select Model1 File", "All Files (*.*)"))
        label_model2 = QLabel("Model2 Path:")
        self.model2_path_input = QLineEdit("path/to/model2")
        browse_model2_btn = QPushButton("Browse")
        browse_model2_btn.clicked.connect(lambda: self.browse_file(self.model2_path_input, "Select Model2 File", "All Files (*.*)"))
        label_num_games = QLabel("Number of Games:")
        self.num_games_spin = QSpinBox()
        self.num_games_spin.setRange(1, 10000)
        self.num_games_spin.setValue(10)
        label_time_per_move = QLabel("Time/Move (sec):")
        self.time_per_move_spin = QDoubleSpinBox()
        self.time_per_move_spin.setRange(0.1, 600.0)
        self.time_per_move_spin.setValue(1.0)
        self.time_per_move_spin.setSingleStep(0.1)
        layout.addWidget(label_model1, 0, 0)
        layout.addLayout(self.create_browse_layout(self.model1_path_input, browse_model1_btn), 0, 1, 1, 3)
        layout.addWidget(label_model2, 1, 0)
        layout.addLayout(self.create_browse_layout(self.model2_path_input, browse_model2_btn), 1, 1, 1, 3)
        layout.addWidget(label_num_games, 2, 0)
        layout.addWidget(self.num_games_spin, 2, 1)
        layout.addWidget(label_time_per_move, 3, 0)
        layout.addWidget(self.time_per_move_spin, 3, 1)
        group.setLayout(layout)
        return group

    def start_benchmark(self):
        model1_path = self.model1_path_input.text()
        model2_path = self.model2_path_input.text()
        num_games = self.num_games_spin.value()
        time_per_move = self.time_per_move_spin.value()
        if not model1_path or not os.path.exists(model1_path):
            QMessageBox.warning(self, "Error", "Model1 file does not exist.")
            return
        if not model2_path or not os.path.exists(model2_path):
            QMessageBox.warning(self, "Error", "Model2 file does not exist.")
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
        started = self.start_worker(BenchmarkWorker, model1_path, model2_path, num_games, time_per_move)
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
        model1_wins = stats_dict.get("model1_wins", 0)
        model2_wins = stats_dict.get("model2_wins", 0)
        draws = stats_dict.get("draws", 0)
        total = stats_dict.get("total_games", 0)
        self.log_text_edit.append(
            "=== BENCHMARK RESULTS ===\n"
            f"Model1 wins: {model1_wins}\n"
            f"Model2 wins: {model2_wins}\n"
            f"Draws: {draws}\n"
            f"Out of {total} games.\n"
        )
        self.visualization.update_benchmark_visualization(model1_wins, model2_wins, draws, total)

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