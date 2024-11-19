from PyQt5.QtWidgets import (
    QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, QMessageBox
)
from src.evaluation.evaluation_visualization import EvaluationVisualization
from src.evaluation.evaluation_worker import EvaluationWorker
from src.base.base_tab import BaseTab
import os


class EvaluationTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization = EvaluationVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        self.paths_group = self.create_paths_group()
        control_buttons_layout = self.create_control_buttons(
            "Start Evaluation",
            "Stop Evaluation",
            self.start_evaluation,
            self.stop_evaluation
        )
        progress_layout = self.create_progress_layout()
        self.log_text_edit = self.create_log_text_edit()
        self.visualization_group = self.create_visualization_group("Evaluation Visualization")

        main_layout.addWidget(self.paths_group)
        main_layout.addLayout(control_buttons_layout)
        main_layout.addLayout(progress_layout)
        main_layout.addWidget(self.log_text_edit)
        main_layout.addWidget(self.visualization_group)

        self.log_text_edit.setVisible(False)
        self.visualization_group.setVisible(False)

    def create_paths_group(self) -> QGroupBox:
        paths_group = QGroupBox("Paths")
        paths_layout = QFormLayout()

        self.model_path_input = QLineEdit("models/saved_models/final_model.pth")
        self.dataset_indices_input = QLineEdit("data/processed/test_indices.npy")
        self.h5_file_input = QLineEdit("data/processed/dataset.h5")

        model_browse_button = QPushButton("Browse")
        model_browse_button.clicked.connect(lambda: self.browse_file(
            self.model_path_input, "Select Model File", "PyTorch Model (*.pth *.pt)"
        ))
        dataset_browse_button = QPushButton("Browse")
        dataset_browse_button.clicked.connect(lambda: self.browse_file(
            self.dataset_indices_input, "Select Evaluation Dataset Indices File", "NumPy Array (*.npy)"
        ))
        h5_file_browse_button = QPushButton("Browse")
        h5_file_browse_button.clicked.connect(lambda: self.browse_file(
            self.h5_file_input, "Select H5 Dataset File", "HDF5 Files (*.h5 *.hdf5)"
        ))

        paths_layout.addRow("Model Path:", self.create_browse_layout(self.model_path_input, model_browse_button))
        paths_layout.addRow("Dataset Indices:", self.create_browse_layout(self.dataset_indices_input, dataset_browse_button))
        paths_layout.addRow("H5 Dataset File:", self.create_browse_layout(self.h5_file_input, h5_file_browse_button))

        paths_group.setLayout(paths_layout)
        return paths_group

    def start_evaluation(self):
        model_path = self.model_path_input.text()
        dataset_indices_path = self.dataset_indices_input.text()
        h5_file_path = self.h5_file_input.text()

        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Model file does not exist.")
            return
        if not os.path.exists(dataset_indices_path):
            QMessageBox.warning(self, "Error", "Dataset indices file does not exist.")
            return
        if not os.path.exists(h5_file_path):
            QMessageBox.warning(self, "Error", "H5 Dataset file does not exist.")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting Evaluation...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()
        self.log_message("Starting evaluation...")
        self.visualization.reset_visualization()

        self.paths_group.setVisible(False)
        self.log_text_edit.setVisible(True)
        self.visualization_group.setVisible(True)

        started = self.start_worker(
            EvaluationWorker, model_path, dataset_indices_path, h5_file_path
        )
        if started:
            self.worker.metrics_update.connect(self.visualization.update_metrics_visualization)
            self.worker.task_finished.connect(self.on_evaluation_finished)
        else:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.paths_group.setVisible(True)
            self.log_text_edit.setVisible(False)
            self.visualization_group.setVisible(False)

    def stop_evaluation(self):
        self.stop_worker()
        self.log_message("Stopping evaluation...")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.paths_group.setVisible(True)
        self.log_text_edit.setVisible(False)
        self.visualization_group.setVisible(False)

    def on_evaluation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Evaluation Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_message("Evaluation process finished.")
        self.paths_group.setVisible(True)