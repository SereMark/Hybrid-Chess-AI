from PyQt5.QtWidgets import (
    QWidget, QTabWidget, QVBoxLayout, 
    QGroupBox, QGridLayout, QLineEdit, QPushButton,
    QMessageBox, QLabel, QFrame, QHBoxLayout, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
import os

from src.base.base_tab import BaseTab
from src.evaluation.evaluation_worker import EvaluationWorker
from src.evaluation.benchmark_worker import BenchmarkWorker
from src.evaluation.evaluation_visualization import EvaluationVisualization
from src.evaluation.benchmark_visualization import BenchmarkVisualization

class EvaluationSubTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = EvaluationVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        intro_label = QLabel("Evaluate a trained model's performance on a test dataset.")
        intro_label.setWordWrap(True)
        intro_label.setAlignment(Qt.AlignLeft)

        self.paths_group = self.create_paths_group()
        self.controls_group = QGroupBox("Evaluation Actions")
        controls_layout = QVBoxLayout(self.controls_group)
        controls_layout.setSpacing(8)
        control_buttons_layout = self.create_control_buttons(
            "Start Evaluation",
            "Stop Evaluation",
            self.start_evaluation,
            self.stop_evaluation
        )
        controls_layout.addLayout(control_buttons_layout)
        self.toggle_buttons_layout = QHBoxLayout()
        self.show_logs_button = QPushButton("Show Logs")
        self.show_logs_button.setCheckable(True)
        self.show_logs_button.setChecked(True)
        self.show_logs_button.clicked.connect(self.show_logs_view)
        self.show_graphs_button = QPushButton("Show Graphs")
        self.show_graphs_button.setCheckable(True)
        self.show_graphs_button.setChecked(False)
        self.show_graphs_button.clicked.connect(self.show_graphs_view)
        self.show_logs_button.clicked.connect(
            lambda: self.show_graphs_button.setChecked(not self.show_logs_button.isChecked())
        )
        self.show_graphs_button.clicked.connect(
            lambda: self.show_logs_button.setChecked(not self.show_graphs_button.isChecked())
        )
        self.toggle_buttons_layout.addWidget(self.show_logs_button)
        self.toggle_buttons_layout.addWidget(self.show_graphs_button)
        controls_layout.addLayout(self.toggle_buttons_layout)
        self.start_new_button = QPushButton("Start New Evaluation")
        self.start_new_button.setToolTip("Reset to initial state for a new evaluation.")
        self.start_new_button.clicked.connect(self.reset_to_initial_state)
        controls_layout.addWidget(self.start_new_button)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        self.progress_group = QGroupBox("Evaluation Progress")
        progress_layout = QVBoxLayout(self.progress_group)
        progress_layout.setSpacing(8)
        progress_layout.addLayout(self.create_progress_layout())
        self.log_group = QGroupBox("Evaluation Logs")
        log_layout = QVBoxLayout(self.log_group)
        log_layout.setSpacing(8)
        self.log_text_edit = self.create_log_text_edit()
        log_layout.addWidget(self.log_text_edit)
        self.log_group.setLayout(log_layout)
        self.visualization_group = self.create_visualization_group(self.visualization, "Evaluation Visualization")
        main_layout.addWidget(intro_label)
        main_layout.addWidget(self.paths_group)
        main_layout.addWidget(self.controls_group)
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

    def reset_to_initial_state(self):
        self.paths_group.setVisible(True)
        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)
        self.visualization_group.setVisible(False)
        self.controls_group.setVisible(True)
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
        self.init_ui_state = True

    def show_logs_view(self):
        if self.show_logs_button.isChecked():
            self.show_graphs_button.setChecked(False)
            self.log_group.setVisible(True)
            self.visualization_group.setVisible(False)

    def show_graphs_view(self):
        if self.show_graphs_button.isChecked():
            self.show_logs_button.setChecked(False)
            self.log_group.setVisible(False)
            self.visualization_group.setVisible(True)

    def create_paths_group(self):
        group = QGroupBox("Evaluation Files")
        layout = QGridLayout()
        label_model = QLabel("Model Path:")
        self.model_path_input = QLineEdit("models/saved_models/final_model.pth")
        browse_model_btn = QPushButton("Browse")
        browse_model_btn.clicked.connect(
            lambda: self.browse_file(self.model_path_input, "Select Model File", "PyTorch Model (*.pth *.pt)")
        )
        label_indices = QLabel("Dataset Indices:")
        self.dataset_indices_input = QLineEdit("data/processed/test_indices.npy")
        browse_indices_btn = QPushButton("Browse")
        browse_indices_btn.clicked.connect(
            lambda: self.browse_file(self.dataset_indices_input, "Select Dataset Indices File", "NumPy Files (*.npy)")
        )
        label_h5 = QLabel("H5 Dataset File:")
        self.h5_file_input = QLineEdit("data/processed/dataset.h5")
        browse_h5_btn = QPushButton("Browse")
        browse_h5_btn.clicked.connect(
            lambda: self.browse_file(self.h5_file_input, "Select H5 File", "HDF5 Files (*.h5 *.hdf5)")
        )
        layout.addWidget(label_model, 0, 0)
        layout.addLayout(self.create_browse_layout(self.model_path_input, browse_model_btn), 0, 1, 1, 3)
        layout.addWidget(label_indices, 1, 0)
        layout.addLayout(self.create_browse_layout(self.dataset_indices_input, browse_indices_btn), 1, 1, 1, 3)
        layout.addWidget(label_h5, 2, 0)
        layout.addLayout(self.create_browse_layout(self.h5_file_input, browse_h5_btn), 2, 1, 1, 3)
        group.setLayout(layout)
        return group

    def start_evaluation(self):
        model_path = self.model_path_input.text()
        indices_path = self.dataset_indices_input.text()
        h5_path = self.h5_file_input.text()
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Model file does not exist.")
            return
        if not os.path.exists(indices_path):
            QMessageBox.warning(self, "Error", "Dataset indices file does not exist.")
            return
        if not os.path.exists(h5_path):
            QMessageBox.warning(self, "Error", "H5 dataset file does not exist.")
            return
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting Evaluation...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()
        self.paths_group.setVisible(False)
        self.progress_group.setVisible(True)
        self.controls_group.setVisible(True)
        self.log_group.setVisible(True)
        self.visualization_group.setVisible(False)
        self.show_logs_button.setVisible(True)
        self.show_graphs_button.setVisible(True)
        self.start_new_button.setVisible(False)
        self.init_ui_state = False
        started = self.start_worker(EvaluationWorker, model_path, indices_path, h5_path)
        if started:
            self.worker.metrics_update.connect(self.visualization.update_metrics_visualization)
        else:
            self.reset_to_initial_state()

    def stop_evaluation(self):
        self.stop_worker()
        self.reset_to_initial_state()


class BenchmarkSubTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = BenchmarkVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        intro_label = QLabel("Benchmark games against another engine or model.")
        intro_label.setWordWrap(True)
        intro_label.setAlignment(Qt.AlignLeft)
        self.benchmark_group = self.create_benchmark_group()
        self.controls_group = QGroupBox("Benchmark Actions")
        controls_layout = QVBoxLayout(self.controls_group)
        controls_layout.setSpacing(8)
        control_buttons_layout = self.create_control_buttons(
            "Start Benchmark",
            "Stop Benchmark",
            self.start_benchmark,
            self.stop_benchmark,
            pause_text="Pause",
            resume_text="Resume",
            pause_callback=self.pause_worker,
            resume_callback=self.resume_worker
        )
        controls_layout.addLayout(control_buttons_layout)
        self.toggle_buttons_layout = QHBoxLayout()
        self.show_logs_button = QPushButton("Show Logs")
        self.show_logs_button.setCheckable(True)
        self.show_logs_button.setChecked(True)
        self.show_logs_button.clicked.connect(self.show_logs_view)
        self.show_graphs_button = QPushButton("Show Graphs")
        self.show_graphs_button.setCheckable(True)
        self.show_graphs_button.setChecked(False)
        self.show_graphs_button.clicked.connect(self.show_graphs_view)
        self.show_logs_button.clicked.connect(
            lambda: self.show_graphs_button.setChecked(not self.show_logs_button.isChecked())
        )
        self.show_graphs_button.clicked.connect(
            lambda: self.show_logs_button.setChecked(not self.show_graphs_button.isChecked())
        )
        self.toggle_buttons_layout.addWidget(self.show_logs_button)
        self.toggle_buttons_layout.addWidget(self.show_graphs_button)
        controls_layout.addLayout(self.toggle_buttons_layout)
        self.start_new_button = QPushButton("Start New Benchmark")
        self.start_new_button.setToolTip("Reset to initial state for a new benchmark.")
        self.start_new_button.clicked.connect(self.reset_to_initial_state)
        controls_layout.addWidget(self.start_new_button)
        self.progress_group = QGroupBox("Benchmark Progress")
        progress_layout = QVBoxLayout(self.progress_group)
        progress_layout.setSpacing(8)
        progress_layout.addLayout(self.create_progress_layout())
        self.log_group = QGroupBox("Benchmark Logs")
        log_layout = QVBoxLayout(self.log_group)
        log_layout.setSpacing(8)
        self.log_text_edit = self.create_log_text_edit()
        log_layout.addWidget(self.log_text_edit)
        self.log_group.setLayout(log_layout)
        self.visualization_group = self.create_visualization_group(self.visualization, "Benchmark Visualization")
        main_layout.addWidget(intro_label)
        main_layout.addWidget(self.benchmark_group)
        main_layout.addWidget(self.controls_group)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
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

    def reset_to_initial_state(self):
        self.benchmark_group.setVisible(True)
        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)
        self.visualization_group.setVisible(False)
        self.controls_group.setVisible(True)
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

    def show_logs_view(self):
        if self.show_logs_button.isChecked():
            self.show_graphs_button.setChecked(False)
            self.log_group.setVisible(True)
            self.visualization_group.setVisible(False)

    def show_graphs_view(self):
        if self.show_graphs_button.isChecked():
            self.show_logs_button.setChecked(False)
            self.log_group.setVisible(False)
            self.visualization_group.setVisible(True)

    def create_benchmark_group(self):
        group = QGroupBox("Benchmark Configuration")
        layout = QGridLayout()
        label_engine = QLabel("Engine/Model Path:")
        self.engine_path_input = QLineEdit("path/to/engine_or_model")
        browse_engine_btn = QPushButton("Browse")
        browse_engine_btn.clicked.connect(
            lambda: self.browse_file(self.engine_path_input, "Select Engine/Model File", "All Files (*.*)")
        )
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
        self.visualization_group.setVisible(False)
        self.show_logs_button.setVisible(True)
        self.show_graphs_button.setVisible(True)
        self.start_new_button.setVisible(False)
        self.start_button.setEnabled(False)
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
        engine_wins = stats_dict.get('engine_wins', 0)
        our_model_wins = stats_dict.get('our_model_wins', 0)
        draws = stats_dict.get('draws', 0)
        total = stats_dict.get('total_games', 0)
        self.log_text_edit.append(
            f"=== BENCHMARK RESULTS ===\n"
            f"Engine wins: {engine_wins}\n"
            f"OurModel wins: {our_model_wins}\n"
            f"Draws: {draws}\n"
            f"Out of {total} games.\n"
        )
        self.visualization.update_benchmark_visualization(
            engine_wins,
            our_model_wins,
            draws,
            total
        )
        self.start_new_button.setVisible(True)


class EvaluationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        self.evaluation_subtab = EvaluationSubTab()
        self.benchmark_subtab = BenchmarkSubTab()
        self.tab_widget.addTab(self.evaluation_subtab, "Model Evaluation")
        self.tab_widget.addTab(self.benchmark_subtab, "Benchmarking")
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)