from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TrainingVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.train_policy_losses = []
        self.train_value_losses = []
        self.val_policy_losses = []
        self.val_value_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.init_ui()

    def init_ui(self):
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.init_visualization()

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2)
        self.ax_train_policy_loss = self.figure.add_subplot(gs[0, 0])
        self.ax_train_value_loss = self.figure.add_subplot(gs[0, 1])
        self.ax_accuracy = self.figure.add_subplot(gs[1, :])

        self.ax_train_policy_loss.set_title('Training Policy Loss')
        self.ax_train_policy_loss.set_xlabel('Batch')
        self.ax_train_policy_loss.set_ylabel('Loss')
        self.ax_train_policy_loss.grid(True)

        self.ax_train_value_loss.set_title('Training Value Loss')
        self.ax_train_value_loss.set_xlabel('Batch')
        self.ax_train_value_loss.set_ylabel('Loss')
        self.ax_train_value_loss.grid(True)

        self.ax_accuracy.set_title('Training and Validation Accuracy')
        self.ax_accuracy.set_xlabel('Epoch')
        self.ax_accuracy.set_ylabel('Accuracy')
        self.ax_accuracy.grid(True)

        self.canvas.draw()

    def update_training_visualization(self, policy_loss, value_loss):
        self.train_policy_losses.append(policy_loss)
        self.train_value_losses.append(value_loss)
        self.update_train_loss_plots()

    def update_validation_visualization(self, val_policy_loss, val_value_loss):
        self.val_policy_losses.append(val_policy_loss)
        self.val_value_losses.append(val_value_loss)

    def update_train_loss_plots(self):
        batches = range(1, len(self.train_policy_losses) + 1)

        self.ax_train_policy_loss.clear()
        self.ax_train_policy_loss.plot(batches, self.train_policy_losses, label='Policy Loss')
        self.ax_train_policy_loss.set_title('Training Policy Loss')
        self.ax_train_policy_loss.set_xlabel('Batch')
        self.ax_train_policy_loss.set_ylabel('Loss')
        self.ax_train_policy_loss.legend()
        self.ax_train_policy_loss.grid(True)

        self.ax_train_value_loss.clear()
        self.ax_train_value_loss.plot(batches, self.train_value_losses, label='Value Loss', color='orange')
        self.ax_train_value_loss.set_title('Training Value Loss')
        self.ax_train_value_loss.set_xlabel('Batch')
        self.ax_train_value_loss.set_ylabel('Loss')
        self.ax_train_value_loss.legend()
        self.ax_train_value_loss.grid(True)

        self.canvas.draw()

    def update_accuracy_visualization(self, training_accuracy, validation_accuracy):
        self.training_accuracies.append(training_accuracy)
        self.validation_accuracies.append(validation_accuracy)
        self.update_accuracy_plot()

    def update_accuracy_plot(self):
        epochs = range(1, len(self.training_accuracies) + 1)
        self.ax_accuracy.clear()
        self.ax_accuracy.plot(epochs, self.training_accuracies, label='Training Accuracy')
        self.ax_accuracy.plot(epochs, self.validation_accuracies, label='Validation Accuracy')
        self.ax_accuracy.set_title('Training and Validation Accuracy')
        self.ax_accuracy.set_xlabel('Epoch')
        self.ax_accuracy.set_ylabel('Accuracy')
        self.ax_accuracy.legend()
        self.ax_accuracy.grid(True)
        self.canvas.draw()

    def reset_visualization(self):
        self.train_policy_losses = []
        self.train_value_losses = []
        self.val_policy_losses = []
        self.val_value_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.init_visualization()