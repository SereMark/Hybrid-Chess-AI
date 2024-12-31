from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QGridLayout, QLineEdit, QPushButton, QMessageBox, QLabel
from src.analysis.evaluation.evaluation_worker import EvaluationWorker
from src.analysis.evaluation.evaluation_visualization import EvaluationVisualization
from src.base.base_tab import BaseTab
import os

class EvaluationSubTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = EvaluationVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.setup_subtab(
            main_layout,
            "Evaluate a trained model's performance on a test dataset.",
            "Evaluation Progress",
            "Evaluation Logs",
            "Evaluation Visualization",
            self.visualization,
            {
                "start_text": "Start Evaluation",
                "stop_text": "Stop Evaluation",
                "start_callback": self.start_evaluation,
                "stop_callback": self.stop_evaluation,
                "start_new_callback": self.reset_to_initial_state
            },
            "Start New Evaluation",
            spacing=10
        )
        self.paths_group = self.create_paths_group()
        self.layout().insertWidget(1, self.paths_group)
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
        if self.stop_button:
            self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting Evaluation...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()
        self.paths_group.setVisible(False)
        self.progress_group.setVisible(True)
        if self.control_group:
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
        started = self.start_worker(EvaluationWorker, model_path, indices_path, h5_path)
        if started:
            self.worker.metrics_update.connect(self.visualization.update_metrics_visualization)
        else:
            self.reset_to_initial_state()

    def stop_evaluation(self):
        self.stop_worker()
        self.reset_to_initial_state()

    def reset_to_initial_state(self):
        self.paths_group.setVisible(True)
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
        self.init_ui_state = True

    def show_logs_view(self):
        super().show_logs_view()

    def show_graphs_view(self):
        super().show_graphs_view()