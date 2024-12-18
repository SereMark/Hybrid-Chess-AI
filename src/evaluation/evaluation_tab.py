from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, QMessageBox, QLabel, QFrame
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

        intro_label = QLabel("Evaluate a trained model's performance on a test dataset.")
        intro_label.setWordWrap(True)
        intro_label.setAlignment(Qt.AlignLeft)
        intro_label.setToolTip("This tab allows you to evaluate the model on test data to measure its performance.")

        self.paths_group = self.create_paths_group()

        progress_group = QGroupBox("Evaluation Progress")
        pg_layout = QVBoxLayout(progress_group)
        pg_layout.addLayout(self.create_progress_layout())

        logs_group = QGroupBox("Evaluation Logs")
        lg_layout = QVBoxLayout(logs_group)
        self.log_text_edit = self.create_log_text_edit()
        lg_layout.addWidget(self.log_text_edit)

        self.visualization_group = self.create_visualization_group(self.visualization, "Evaluation Visualization")

        controls_group = QGroupBox("Actions")
        cg_layout = QVBoxLayout(controls_group)
        control_buttons_layout = self.create_control_buttons(
            "Start Evaluation",
            "Stop Evaluation",
            self.start_evaluation,
            self.stop_evaluation
        )
        cg_layout.addLayout(control_buttons_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        main_layout.addWidget(intro_label)
        main_layout.addWidget(self.paths_group)
        main_layout.addWidget(separator)
        main_layout.addWidget(progress_group)
        main_layout.addWidget(logs_group)
        main_layout.addWidget(self.visualization_group)
        main_layout.addWidget(controls_group)

        self.toggle_widget_state([self.log_text_edit, self.visualization_group], state=False, attribute="visible")

    def create_paths_group(self) -> QGroupBox:
        paths_group = QGroupBox("Evaluation Files")
        paths_layout = QFormLayout()

        self.model_path_input = QLineEdit("models/saved_models/final_model.pth")
        self.model_path_input.setPlaceholderText("Path to the trained model file (e.g. models/saved_models/final_model.pth)")
        self.model_path_input.setToolTip("The trained model to evaluate.")

        self.dataset_indices_input = QLineEdit("data/processed/test_indices.npy")
        self.dataset_indices_input.setPlaceholderText("Path to test indices file (e.g. data/processed/test_indices.npy)")
        self.dataset_indices_input.setToolTip("Indices of test examples for evaluation.")

        self.h5_file_input = QLineEdit("data/processed/dataset.h5")
        self.h5_file_input.setPlaceholderText("Path to dataset H5 file (e.g. data/processed/dataset.h5)")
        self.h5_file_input.setToolTip("HDF5 dataset file containing evaluation data.")

        model_browse_button = QPushButton("Browse")
        model_browse_button.setToolTip("Browse for the model file.")
        model_browse_button.clicked.connect(lambda: self.browse_file(self.model_path_input, "Select Model File", "PyTorch Model (*.pth *.pt)"))

        dataset_browse_button = QPushButton("Browse")
        dataset_browse_button.setToolTip("Browse for the test indices file.")
        dataset_browse_button.clicked.connect(lambda: self.browse_file(self.dataset_indices_input, "Select Evaluation Dataset Indices File", "NumPy Array (*.npy)"))

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

        self.toggle_widget_state([self.start_button], state=False, attribute="enabled")
        self.toggle_widget_state([self.stop_button], state=True, attribute="enabled")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting Evaluation...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()

        self.toggle_widget_state([self.paths_group], state=False, attribute="visible")
        self.toggle_widget_state([self.log_text_edit, self.visualization_group], state=True, attribute="visible")

        started = self.start_worker(
            EvaluationWorker,
            model_path,
            dataset_indices_path,
            h5_file_path
        )
        if started:
            self.worker.metrics_update.connect(self.visualization.update_metrics_visualization)
            self.worker.task_finished.connect(self.on_evaluation_finished)
        else:
            self.toggle_widget_state([self.start_button], state=True, attribute="enabled")
            self.toggle_widget_state([self.stop_button], state=False, attribute="enabled")
            self.toggle_widget_state([self.paths_group], state=True, attribute="visible")
            self.toggle_widget_state([self.log_text_edit, self.visualization_group], state=False, attribute="visible")

    def stop_evaluation(self):
        self.stop_worker()
        self.toggle_widget_state([self.start_button], state=True, attribute="enabled")
        self.toggle_widget_state([self.stop_button], state=False, attribute="enabled")
        self.toggle_widget_state([self.paths_group], state=True, attribute="visible")
        self.toggle_widget_state([self.log_text_edit, self.visualization_group], state=False, attribute="visible")

    def on_evaluation_finished(self):
        self.toggle_widget_state([self.start_button], state=True, attribute="enabled")
        self.toggle_widget_state([self.stop_button], state=False, attribute="enabled")
        self.progress_bar.setFormat("Evaluation Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.toggle_widget_state([self.paths_group], state=True, attribute="visible")