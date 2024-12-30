from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QGridLayout, QLineEdit, QPushButton, QMessageBox, QLabel
from PyQt5.QtCore import Qt
import os
from src.base.base_tab import BaseTab
from src.evaluation.evaluation_worker import EvaluationWorker
from src.evaluation.evaluation_visualization import EvaluationVisualization

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
        self.toggle_buttons_layout = self.create_log_graph_buttons(self.show_logs_view, self.show_graphs_view, "Show Logs", "Show Graphs", True)
        controls_layout.addLayout(self.toggle_buttons_layout)
        self.start_new_button = self.create_start_new_button("Start New Evaluation", self.reset_to_initial_state)
        controls_layout.addWidget(self.start_new_button)
        separator = self.create_separator()
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
        browse_model_btn.clicked.connect(lambda: self.browse_file(self.model_path_input, "Select Model File", "PyTorch Model (*.pth *.pt)"))
        label_indices = QLabel("Dataset Indices:")
        self.dataset_indices_input = QLineEdit("data/processed/test_indices.npy")
        browse_indices_btn = QPushButton("Browse")
        browse_indices_btn.clicked.connect(lambda: self.browse_file(self.dataset_indices_input, "Select Dataset Indices File", "NumPy Files (*.npy)"))
        label_h5 = QLabel("H5 Dataset File:")
        self.h5_file_input = QLineEdit("data/processed/dataset.h5")
        browse_h5_btn = QPushButton("Browse")
        browse_h5_btn.clicked.connect(lambda: self.browse_file(self.h5_file_input, "Select H5 File", "HDF5 Files (*.h5 *.hdf5)"))
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