from PyQt5.QtWidgets import QMessageBox, QVBoxLayout, QProgressBar, QTextEdit, QWidget, QLabel, QLineEdit, QHBoxLayout, QPushButton, QGroupBox
from PyQt5.QtCore import QThread, pyqtSignal
from src.neural_network.train import ModelTrainer
from src.gui.visualizations.training_visualization import TrainingVisualization

class TrainingWorker(QThread):
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    loss_update = pyqtSignal(float, float)
    finished = pyqtSignal()

    def __init__(self, epochs, batch_size, learning_rate, momentum, weight_decay):
        super().__init__()
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        self.momentum, self.weight_decay = momentum, weight_decay
        self.stop_training, self.pause_training = False, False

    def run(self):
        try:
            trainer = ModelTrainer(
                epochs=self.epochs, batch_size=self.batch_size, lr=self.learning_rate,
                momentum=self.momentum, weight_decay=self.weight_decay,
                log_fn=self.log_update.emit, progress_fn=self.emit_progress, 
                loss_fn=self.emit_loss, stop_fn=lambda: self.stop_training, 
                pause_fn=lambda: self.pause_training
            )
            self.log_update.emit("Starting training...")
            trainer.train_model()
            if not self.stop_training:
                self.log_update.emit("Training completed successfully.")
        except Exception as e:
            self.log_update.emit(f"Error during training: {e}")
        finally:
            self.finished.emit()

    def emit_progress(self, value):
        if not self.pause_training and not self.stop_training:
            self.progress_update.emit(value)

    def emit_loss(self, policy_loss, value_loss):
        if not self.pause_training and not self.stop_training:
            self.loss_update.emit(policy_loss, value_loss)

    def pause(self): self.pause_training = True
    def resume(self): self.pause_training = False
    def stop(self): self.stop_training = True

class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker, self.visualization = None, TrainingVisualization()
        self.init_specific_ui()

    def init_specific_ui(self):
        layout = QVBoxLayout(self)
        layout.addLayout(self.create_input_layout("Epochs:", "10", "epochs_input"))
        layout.addLayout(self.create_input_layout("Batch Size:", "512", "batch_size_input"))
        layout.addLayout(self.create_input_layout("Learning Rate:", "0.01", "learning_rate_input"))
        layout.addLayout(self.create_input_layout("Momentum:", "0.9", "momentum_input"))
        layout.addLayout(self.create_input_layout("Weight Decay:", "1e-4", "weight_decay_input"))

        buttons_layout = self.create_buttons_layout()
        layout.addLayout(buttons_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        layout.addWidget(self.progress_bar)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        layout.addWidget(self.log_text_edit)

        vis_group = QGroupBox("Training Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

    def create_input_layout(self, label_text, default_value, input_attr_name):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        input_field = QLineEdit(default_value)
        setattr(self, input_attr_name, input_field)
        layout.addWidget(label)
        layout.addWidget(input_field)
        return layout

    def create_buttons_layout(self):
        layout = QHBoxLayout()
        self.start_button, self.pause_button, self.resume_button, self.stop_button = (
            QPushButton("Start Training"), QPushButton("Pause Training"), 
            QPushButton("Resume Training"), QPushButton("Stop Training")
        )
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.resume_button)
        layout.addWidget(self.stop_button)

        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_training)
        self.pause_button.clicked.connect(self.pause_training)
        self.resume_button.clicked.connect(self.resume_training)
        self.stop_button.clicked.connect(self.stop_training)

        return layout

    def start_training(self):
        try:
            epochs, batch_size = int(self.epochs_input.text()), int(self.batch_size_input.text())
            learning_rate, momentum = float(self.learning_rate_input.text()), float(self.momentum_input.text())
            weight_decay = float(self.weight_decay_input.text())
            if any(v <= 0 for v in [epochs, batch_size, learning_rate, momentum, weight_decay]):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid and positive hyperparameters.")
            return
    
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.resume_button.setEnabled(False)
    
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
    
        self.worker = TrainingWorker(epochs, batch_size, learning_rate, momentum, weight_decay)
        self.worker.log_update.connect(self.log_text_edit.append)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.loss_update.connect(self.visualization.update_training_visualization)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.start()

    def pause_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)
            self.log_text_edit.append("Pausing training...")

    def resume_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.resume()
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.log_text_edit.append("Resuming training...")

    def stop_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.log_text_edit.append("Stopping training...")

    def on_training_finished(self):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Training Finished")
        self.log_text_edit.append("Training process finished.")
        self.visualization.update_training_visualization(None, None)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Progress: {value}%")