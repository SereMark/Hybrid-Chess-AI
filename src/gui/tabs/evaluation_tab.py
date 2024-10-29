import os
import time
import numpy as np
import torch
import h5py
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, QFileDialog, QHBoxLayout, QProgressBar, QLabel, QTextEdit, QMessageBox
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from src.gui.visualizations.evaluation_visualization import EvaluationVisualization
from src.gui.tabs.supervised_training_tab import TOTAL_MOVES, MOVE_MAPPING, ChessModel, H5Dataset

class EvaluationWorker(QThread):
    log_update = pyqtSignal(str)
    metrics_update = pyqtSignal(float, float, dict, dict, np.ndarray, list)
    progress_update = pyqtSignal(int)
    time_left_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_path, dataset_indices_path):
        super().__init__()
        self.model_path = model_path
        self.dataset_indices_path = dataset_indices_path
        self._is_stopped = False

    def run(self):
        try:
            evaluator = ModelEvaluator(
                model_path=self.model_path,
                dataset_indices_path=self.dataset_indices_path,
                log_fn=self.log_update.emit,
                metrics_fn=self.metrics_update.emit,
                progress_fn=self.progress_update.emit,
                time_left_fn=self.time_left_update.emit,
                stop_fn=lambda: self._is_stopped
            )
            evaluator.evaluate_model()
            if not self._is_stopped:
                self.log_update.emit("Evaluation completed successfully.")
        except Exception as e:
            self.log_update.emit(f"Error during evaluation: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._is_stopped = True
        self.log_update.emit("Evaluation stopped by user.")


class EvaluationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = EvaluationVisualization()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        settings_group = QGroupBox("Evaluation Settings")
        settings_layout = QFormLayout()

        self.model_path_input = QLineEdit("models/saved_models/final_model.pth")
        self.evaluation_dataset_indices_input = QLineEdit("data/processed/test_indices.npy")

        model_browse_button = QPushButton("Browse")
        model_browse_button.clicked.connect(lambda: self.browse_model(self.model_path_input))
        dataset_browse_button = QPushButton("Browse")
        dataset_browse_button.clicked.connect(lambda: self.browse_dataset(self.evaluation_dataset_indices_input))

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(model_browse_button)

        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.evaluation_dataset_indices_input)
        dataset_layout.addWidget(dataset_browse_button)

        settings_layout.addRow("Model Path:", model_layout)
        settings_layout.addRow("Evaluation Dataset Indices:", dataset_layout)

        settings_group.setLayout(settings_layout)

        control_buttons_layout = self.create_buttons_layout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label = QLabel("Time Left: Calculating...")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        main_layout.addWidget(settings_group)
        main_layout.addLayout(control_buttons_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.remaining_time_label)
        main_layout.addWidget(self.log_text_edit)
        main_layout.addWidget(self.create_visualization_group())

    def create_buttons_layout(self):
        layout = QHBoxLayout()
        self.start_button = QPushButton("Start Evaluation")
        self.stop_button = QPushButton("Stop Evaluation")
        self.stop_button.setEnabled(False)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addStretch()
        self.start_button.clicked.connect(self.start_evaluation)
        self.stop_button.clicked.connect(self.stop_evaluation)
        return layout

    def create_visualization_group(self):
        visualization_group = QGroupBox("Evaluation Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        visualization_group.setLayout(vis_layout)
        return visualization_group

    def browse_model(self, input_field):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", input_field.text(), "PyTorch Model (*.pth *.pt)")
        if file_path:
            input_field.setText(file_path)

    def browse_dataset(self, input_field):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Evaluation Dataset Indices File", input_field.text(), "NumPy Array (*.npy)")
        if file_path:
            input_field.setText(file_path)

    def start_evaluation(self):
        model_path = self.model_path_input.text()
        dataset_indices_path = self.evaluation_dataset_indices_input.text()

        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Model file does not exist.")
            return
        if not os.path.exists(dataset_indices_path):
            QMessageBox.warning(self, "Error", "Dataset indices file does not exist.")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting Evaluation...")
        self.remaining_time_label.setText("Time Left: Calculating...")
        self.log_text_edit.clear()
        self.visualization.reset_visualization()

        self.worker = EvaluationWorker(model_path, dataset_indices_path)
        self.worker.log_update.connect(self.update_log)
        self.worker.metrics_update.connect(self.visualization.update_metrics_visualization)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.time_left_update.connect(self.update_time_left)
        self.worker.finished.connect(self.on_evaluation_finished)
        self.worker.start()

    def stop_evaluation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.log_text_edit.append("Stopping evaluation...")

    def update_log(self, message):
        self.log_text_edit.append(message)

    def on_evaluation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Evaluation Finished")
        self.remaining_time_label.setText("Time Left: N/A")
        self.log_text_edit.append("Evaluation process finished.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Progress: {value}%")

    def update_time_left(self, time_left_str):
        self.remaining_time_label.setText(f"Time Left: {time_left_str}")

class ModelEvaluator:
    def __init__(self, model_path, dataset_indices_path, log_fn=None, metrics_fn=None, progress_fn=None,
                 time_left_fn=None, stop_fn=None):
        self.model_path = model_path
        self.dataset_indices_path = dataset_indices_path
        self.log_fn = log_fn
        self.metrics_fn = metrics_fn
        self.progress_fn = progress_fn
        self.time_left_fn = time_left_fn
        self.stop_fn = stop_fn or (lambda: False)

    def format_time_left(self, seconds):
            days = seconds // 86400
            remainder = seconds % 86400
            hours = remainder // 3600
            minutes = (remainder % 3600) // 60
            secs = remainder % 60

            if days >= 1:
                day_str = f"{int(days)}d " if days > 1 else "1d "
                return f"{day_str}{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
            else:
                return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

    def evaluate_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.log_fn:
            self.log_fn(f"Using device: {device}")

        model = ChessModel(num_moves=TOTAL_MOVES)
        try:
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            if self.log_fn:
                self.log_fn(f"Model loaded from {self.model_path}")
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Failed to load model: {e}")
            return

        model.to(device)
        model.eval()

        data_dir = os.path.dirname(self.dataset_indices_path)
        h5_path = os.path.join(data_dir, 'dataset.h5')
        if not os.path.exists(h5_path):
            if self.log_fn:
                self.log_fn(f"Dataset file not found at {h5_path}.")
            return
        if not os.path.exists(self.dataset_indices_path):
            if self.log_fn:
                self.log_fn(f"Dataset indices file not found at {self.dataset_indices_path}.")
            return

        try:
            h5_file = h5py.File(h5_path, 'r')
            dataset_indices = np.load(self.dataset_indices_path)
            dataset = H5Dataset(h5_file, dataset_indices)
            if self.log_fn:
                self.log_fn(f"Loaded dataset indices from {self.dataset_indices_path}")
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Failed to load dataset: {e}")
            return

        loader = DataLoader(
            dataset, batch_size=1024, shuffle=False, num_workers=0
        )

        all_predictions = []
        all_actuals = []
        total_batches = len(loader)
        k = 5
        topk_predictions = []

        start_time = time.time()
        steps_done = 0
        total_steps = total_batches

        if self.log_fn:
            self.log_fn("Starting evaluation...")

        for batch_idx, (inputs, policy_targets, _) in enumerate(loader, 1):
            if self.stop_fn():
                if self.log_fn:
                    self.log_fn("Evaluation stopped by user.")
                return

            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            with torch.no_grad():
                policy_outputs, _ = model(inputs)
                _, preds = torch.max(policy_outputs, 1)
                all_predictions.extend(preds.cpu().numpy())
                all_actuals.extend(policy_targets.cpu().numpy())

                _, topk_preds = torch.topk(policy_outputs, k, dim=1)
                topk_predictions.extend(topk_preds.cpu().numpy())

            steps_done += 1
            progress = int((steps_done / total_steps) * 100)
            elapsed_time = time.time() - start_time

            if self.progress_fn:
                self.progress_fn(progress)

            if self.time_left_fn:
                estimated_total_time = (elapsed_time / steps_done) * total_steps
                time_left = estimated_total_time - elapsed_time
                time_left_str = self.format_time_left(time_left)
                self.time_left_fn(time_left_str)

        h5_file.close()

        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        topk_predictions = np.array(topk_predictions)

        accuracy = np.mean(all_predictions == all_actuals)
        if self.log_fn:
            self.log_fn(f"Accuracy: {accuracy * 100:.2f}%")

        topk_correct = sum([1 for actual, preds in zip(all_actuals, topk_predictions) if actual in preds])
        topk_accuracy = topk_correct / len(all_actuals)
        if self.log_fn:
            self.log_fn(f"Top-{k} Accuracy: {topk_accuracy * 100:.2f}%")

        N = 10
        from collections import Counter
        class_counts = Counter(all_actuals)
        most_common_classes = [item[0] for item in class_counts.most_common(N)]
        indices = np.isin(all_actuals, most_common_classes)
        filtered_actuals = all_actuals[indices]
        filtered_predictions = all_predictions[indices]

        confusion = confusion_matrix(filtered_actuals, filtered_predictions, labels=most_common_classes)
        if self.log_fn:
            self.log_fn("Confusion Matrix computed.")

        report = classification_report(filtered_actuals, filtered_predictions, labels=most_common_classes,
                                       output_dict=True, zero_division=0)
        macro_avg = report.get('macro avg', {})
        weighted_avg = report.get('weighted avg', {})

        class_labels = [MOVE_MAPPING[cls].uci() for cls in most_common_classes]

        if self.log_fn:
            self.log_fn(f"Macro Avg - Precision: {macro_avg.get('precision', 0.0):.4f}, "
                        f"Recall: {macro_avg.get('recall', 0.0):.4f}, "
                        f"F1-Score: {macro_avg.get('f1-score', 0.0):.4f}")
            self.log_fn(f"Weighted Avg - Precision: {weighted_avg.get('precision', 0.0):.4f}, "
                        f"Recall: {weighted_avg.get('recall', 0.0):.4f}, "
                        f"F1-Score: {weighted_avg.get('f1-score', 0.0):.4f}")

        if self.metrics_fn:
            self.metrics_fn(accuracy, topk_accuracy, macro_avg, weighted_avg, confusion, class_labels)

        if self.log_fn:
            self.log_fn("Evaluation process finished.")