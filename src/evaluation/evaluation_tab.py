from PyQt5.QtWidgets import (
    QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, QMessageBox, QLabel, 
    QFrame, QHBoxLayout
)
from PyQt5.QtCore import Qt
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
        main_layout.setSpacing(15)

        intro_label = QLabel("Evaluate a trained model's performance on a test dataset.")
        intro_label.setWordWrap(True)
        intro_label.setAlignment(Qt.AlignLeft)
        intro_label.setToolTip("Evaluate the model on test data.")

        self.paths_group = self.create_paths_group()

        self.controls_group = QGroupBox("Actions")
        cg_layout = QVBoxLayout(self.controls_group)
        cg_layout.setSpacing(10)
        control_buttons_layout = self.create_control_buttons(
            "Start Evaluation",
            "Stop Evaluation",
            self.start_evaluation,
            self.stop_evaluation
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
        self.start_new_button.setToolTip("Start a new evaluation configuration.")
        self.start_new_button.clicked.connect(self.reset_to_initial_state)
        cg_layout.addWidget(self.start_new_button)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        self.progress_group = QGroupBox("Evaluation Progress")
        pg_layout = QVBoxLayout(self.progress_group)
        pg_layout.setSpacing(10)
        pg_layout.addLayout(self.create_progress_layout())

        self.log_group = QGroupBox("Evaluation Logs")
        lg_layout = QVBoxLayout(self.log_group)
        lg_layout.setSpacing(10)
        self.log_text_edit = self.create_log_text_edit()
        lg_layout.addWidget(self.log_text_edit)
        self.log_group.setLayout(lg_layout)

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
            self.showing_logs = True

    def show_graphs_view(self):
        if self.show_graphs_button.isChecked():
            self.show_logs_button.setChecked(False)
            self.log_group.setVisible(False)
            self.visualization_group.setVisible(True)
            self.showing_logs = False

    def create_paths_group(self) -> QGroupBox:
        paths_group = QGroupBox("Evaluation Files")
        paths_layout = QFormLayout()
        paths_layout.setSpacing(10)

        self.model_path_input = QLineEdit("models/saved_models/final_model.pth")
        self.model_path_input.setPlaceholderText("Path to the trained model file")
        self.model_path_input.setToolTip("The trained model to evaluate.")

        self.dataset_indices_input = QLineEdit("data/processed/test_indices.npy")
        self.dataset_indices_input.setPlaceholderText("Path to test indices file")
        self.dataset_indices_input.setToolTip("Indices of test samples for evaluation.")

        self.h5_file_input = QLineEdit("data/processed/dataset.h5")
        self.h5_file_input.setPlaceholderText("Path to dataset H5 file")
        self.h5_file_input.setToolTip("HDF5 dataset file containing evaluation data.")

        model_browse_button = QPushButton("Browse")
        model_browse_button.setToolTip("Browse for the model file.")
        model_browse_button.clicked.connect(lambda: self.browse_file(self.model_path_input, "Select Model File", "PyTorch Model (*.pth *.pt)"))

        dataset_browse_button = QPushButton("Browse")
        dataset_browse_button.setToolTip("Browse for the test indices file.")
        dataset_browse_button.clicked.connect(lambda: self.browse_file(self.dataset_indices_input, "Select Evaluation Dataset Indices File", "NumPy Files (*.npy)"))

        h5_file_browse_button = QPushButton("Browse")
        h5_file_browse_button.setToolTip("Browse for the H5 dataset file.")
        h5_file_browse_button.clicked.connect(lambda: self.browse_file(self.h5_file_input, "Select H5 Dataset File", "HDF5 Files (*.h5 *.hdf5)"))

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

        started = self.start_worker(
            EvaluationWorker,
            model_path,
            dataset_indices_path,
            h5_file_path
        )
        if not started:
            self.reset_to_initial_state()

    def stop_evaluation(self):
        self.stop_worker()
        self.reset_to_initial_state()

    def on_evaluation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Evaluation Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.start_new_button.setVisible(True)