from PyQt5.QtWidgets import QMessageBox, QVBoxLayout, QGroupBox
from PyQt5.QtCore import QThread, pyqtSignal
from src.neural_network.train import ModelTrainer
from src.common.base_tab import BaseTab
from src.common.common_widgets import create_labeled_input
from src.gui.visualizations.training_visualization import TrainingVisualization

class TrainingWorker(QThread):
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    loss_update = pyqtSignal(float, float)
    finished = pyqtSignal()

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.stop_training = False
        self.pause_training = False

    def run(self):
        try:
            trainer = ModelTrainer( epochs=self.epochs, batch_size=self.batch_size, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay, log_fn=self.log_update.emit, progress_fn=self.emit_progress, loss_fn=self.emit_loss, stop_fn=lambda: self.stop_training, pause_fn=lambda: self.pause_training)
            self.log_update.emit("Starting training...")
            trainer.train_model()
            self.log_update.emit("Training completed successfully." if not self.stop_training else "Training was stopped by user.")
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

    def pause(self):
        self.pause_training = True

    def resume(self):
        self.pause_training = False

class TrainingTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.visualization = TrainingVisualization()
        self.init_specific_ui()

    def init_specific_ui(self):
        self.epochs_label, self.epochs_input, _, epochs_layout = create_labeled_input("Epochs:", "10")
        self.batch_size_label, self.batch_size_input, _, batch_size_layout = create_labeled_input("Batch Size:", "512")
        self.learning_rate_label, self.learning_rate_input, _, lr_layout = create_labeled_input("Learning Rate:", "0.01")
        self.momentum_label, self.momentum_input, _, momentum_layout = create_labeled_input("Momentum:", "0.9")
        self.weight_decay_label, self.weight_decay_input, _, wd_layout = create_labeled_input("Weight Decay:", "1e-4")
    
        self.config_layout.addWidget(self.epochs_label)
        self.config_layout.addLayout(epochs_layout)
        self.config_layout.addWidget(self.batch_size_label)
        self.config_layout.addLayout(batch_size_layout)
        self.config_layout.addWidget(self.learning_rate_label)
        self.config_layout.addLayout(lr_layout)
        self.config_layout.addWidget(self.momentum_label)
        self.config_layout.addLayout(momentum_layout)
        self.config_layout.addWidget(self.weight_decay_label)
        self.config_layout.addLayout(wd_layout)
    
        self.buttons["start"].setText("Start Training")
        self.buttons["pause"].setText("Pause Training")
        self.buttons["resume"].setText("Resume Training")
        self.buttons["stop"].setText("Stop Training")
    
        self.buttons["start"].clicked.connect(self.start_training)
        self.buttons["pause"].clicked.connect(self.pause_training)
        self.buttons["resume"].clicked.connect(self.resume_training)
        self.buttons["stop"].clicked.connect(self.stop_training)
    
        self.visualization_group = QGroupBox("Training Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization)
        self.visualization_group.setLayout(vis_layout)
        self.layout.addWidget(self.visualization_group)
    
    def start_training(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Training", "Training is already running.")
            return
    
        try:
            params = {
                'epochs': int(self.epochs_input.text()),
                'batch_size': int(self.batch_size_input.text()),
                'learning_rate': float(self.learning_rate_input.text()),
                'momentum': float(self.momentum_input.text()),
                'weight_decay': float(self.weight_decay_input.text())
            }
            if any(v <= 0 for v in params.values()):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid and positive hyperparameters.")
            return
    
        self.buttons['start'].setEnabled(False)
        self.buttons['pause'].setEnabled(True)
        self.buttons['stop'].setEnabled(True)
    
        self.log_signal.emit("Starting training...")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Initializing...")
    
        self.worker = TrainingWorker(**params)
        self.worker.log_update.connect(self.log_signal.emit)
        self.worker.progress_update.connect(self.progress_signal.emit)
        self.worker.loss_update.connect(self.visualization.update_training_visualization)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.start()
    
    def pause_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.pause()
            self.buttons['pause'].setEnabled(False)
            self.buttons['resume'].setEnabled(True)
            self.log_signal.emit("Pausing training...")
    
    def resume_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.resume()
            self.buttons['pause'].setEnabled(True)
            self.buttons['resume'].setEnabled(False)
            self.log_signal.emit("Resuming training...")
    
    def stop_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop_training = True
            self.log_signal.emit("Stopping training...")
            self.buttons['start'].setEnabled(True)
            self.buttons['pause'].setEnabled(False)
            self.buttons['resume'].setEnabled(False)
            self.buttons['stop'].setEnabled(False)
    
    def on_training_finished(self):
        for button in self.buttons.values():
            button.setEnabled(button == self.buttons['start'])
        self.visualization.update_training_visualization(policy_loss=None, value_loss=None)
        self.log_signal.emit("Training process finished.")
        self.progress_bar.setFormat("Training Finished")