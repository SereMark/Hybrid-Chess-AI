from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import math

class BasePlot:
    def __init__(self, ax, title, xlabel='', ylabel=''):
        self.ax = ax
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel(ylabel, fontsize=10)
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

class TrainingVisualization(QWidget):
    def __init__(self, parent=None, max_points=1000):
        super().__init__(parent)
        self.max_points = max_points
        self.loss_batches = []
        self.accuracy_batches = []
        self.lr_batches = []
        self.train_policy_losses = []
        self.train_value_losses = []
        self.training_accuracies = []
        self.learning_rates = []
        self.init_ui()

    def init_ui(self):
        self.figure = Figure(figsize=(12, 8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.init_visualization()

    def init_visualization(self):
        self.figure.clf()
        gs = self.figure.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        self.ax_policy_loss = self.figure.add_subplot(gs[0, 0])
        BasePlot(self.ax_policy_loss, title='Policy Loss over Batches', xlabel='Batch', ylabel='Loss')
        self.ax_value_loss = self.figure.add_subplot(gs[0, 1])
        BasePlot(self.ax_value_loss, title='Value Loss over Batches', xlabel='Batch', ylabel='Loss')
        self.ax_accuracy = self.figure.add_subplot(gs[1, 0])
        BasePlot(self.ax_accuracy, title='Training Accuracy over Batches', xlabel='Batch', ylabel='Accuracy (%)')
        self.ax_lr = self.figure.add_subplot(gs[1, 1])
        BasePlot(self.ax_lr, title='Learning Rate over Batches', xlabel='Batch', ylabel='Learning Rate')
        self.canvas.draw()

    def set_total_batches(self, total_batches):
        pass

    def update_loss_plots(self, batch_idx, losses):
        if not all(math.isfinite(v) for v in losses.values()):
            print(f"Non-finite loss values detected at batch {batch_idx}. Skipping plot update.")
            return
        self.loss_batches.append(batch_idx)
        self.train_policy_losses.append(losses['policy'])
        self.train_value_losses.append(losses['value'])
        if len(self.loss_batches) > self.max_points:
            self.loss_batches = self.loss_batches[-self.max_points:]
            self.train_policy_losses = self.train_policy_losses[-self.max_points:]
            self.train_value_losses = self.train_value_losses[-self.max_points:]
        self.plot_losses()

    def update_accuracy_plot(self, batch_idx, accuracy):
        if not math.isfinite(accuracy):
            print(f"Non-finite accuracy value detected at batch {batch_idx}. Skipping plot update.")
            return
        self.accuracy_batches.append(batch_idx)
        self.training_accuracies.append(accuracy * 100)
        if len(self.accuracy_batches) > self.max_points:
            self.accuracy_batches = self.accuracy_batches[-self.max_points:]
            self.training_accuracies = self.training_accuracies[-self.max_points:]
        self.plot_accuracies()

    def update_learning_rate(self, batch_idx, lr):
        if not math.isfinite(lr):
            print(f"Non-finite learning rate detected at batch {batch_idx}. Skipping plot update.")
            return
        self.lr_batches.append(batch_idx)
        self.learning_rates.append(lr)
        if len(self.lr_batches) > self.max_points:
            self.lr_batches = self.lr_batches[-self.max_points:]
            self.learning_rates = self.learning_rates[-self.max_points:]
        self.plot_learning_rate()

    def plot_losses(self):
        try:
            self.ax_policy_loss.clear()
            BasePlot(self.ax_policy_loss, title='Policy Loss over Batches', xlabel='Batch', ylabel='Loss')
            self.ax_policy_loss.plot(
                self.loss_batches, 
                self.train_policy_losses, 
                color='#1f77b4', marker='o', markersize=3, linestyle='-'
            )
            self.ax_value_loss.clear()
            BasePlot(self.ax_value_loss, title='Value Loss over Batches', xlabel='Batch', ylabel='Loss')
            self.ax_value_loss.plot(
                self.loss_batches, 
                self.train_value_losses, 
                color='#ff7f0e', marker='o', markersize=3, linestyle='-'
            )
            self.figure.canvas.draw_idle()
        except Exception as e:
            print(f"Error in plot_losses: {e}")

    def plot_accuracies(self):
        try:
            self.ax_accuracy.clear()
            BasePlot(self.ax_accuracy, title='Training Accuracy over Batches', xlabel='Batch', ylabel='Accuracy (%)')
            self.ax_accuracy.plot(
                self.accuracy_batches, 
                self.training_accuracies, 
                color='#9467bd', marker='o', markersize=3, linestyle='-'
            )
            self.figure.canvas.draw_idle()
        except Exception as e:
            print(f"Error in plot_accuracies: {e}")

    def plot_learning_rate(self):
        try:
            self.ax_lr.clear()
            BasePlot(self.ax_lr, title='Learning Rate over Batches', xlabel='Batch', ylabel='Learning Rate')
            self.ax_lr.plot(
                self.lr_batches, 
                self.learning_rates, 
                color='#2ca02c', marker='o', markersize=3, linestyle='-'
            )
            self.figure.canvas.draw_idle()
        except Exception as e:
            print(f"Error in plot_learning_rate: {e}")

    def reset_visualization(self):
        self.loss_batches = []
        self.accuracy_batches = []
        self.lr_batches = []
        self.train_policy_losses = []
        self.train_value_losses = []
        self.training_accuracies = []
        self.learning_rates = []
        self.init_visualization()