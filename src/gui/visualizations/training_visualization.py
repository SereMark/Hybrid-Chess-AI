from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class BasePlot:
    def __init__(self, ax, title, xlabel='', ylabel=''):
        self.ax = ax
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel(ylabel, fontsize=10)
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

class TrainingVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.batches = []
        self.train_policy_losses = []
        self.train_value_losses = []
        self.training_accuracies = []
        self.learning_rates = []
        self.init_ui()

    def init_ui(self):
        self.figure = Figure(figsize=(12, 8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.init_visualization()

    def init_visualization(self):
        self.figure.clear()
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

    def update_loss_plots(self, batch_idx, losses):
        self.batches.append(batch_idx)
        self.train_policy_losses.append(losses['policy'])
        self.train_value_losses.append(losses['value'])
        self.plot_losses()

    def plot_losses(self):
        self.ax_policy_loss.clear()
        BasePlot(self.ax_policy_loss, title='Policy Loss over Batches', xlabel='Batch', ylabel='Loss')
        self.ax_policy_loss.plot(self.batches, self.train_policy_losses, color='#1f77b4', marker='o')
        self.ax_value_loss.clear()
        BasePlot(self.ax_value_loss, title='Value Loss over Batches', xlabel='Batch', ylabel='Loss')
        self.ax_value_loss.plot(self.batches, self.train_value_losses, color='#ff7f0e', marker='o')
        self.canvas.draw()

    def update_accuracy_plot(self, batch_idx, accuracy):
        self.training_accuracies.append(accuracy * 100)
        self.plot_accuracies()

    def plot_accuracies(self):
        self.ax_accuracy.clear()
        BasePlot(self.ax_accuracy, title='Training Accuracy over Batches', xlabel='Batch', ylabel='Accuracy (%)')
        self.ax_accuracy.plot(self.batches, self.training_accuracies, color='#9467bd', marker='o')
        self.canvas.draw()

    def update_learning_rate(self, batch_idx, lr):
        self.learning_rates.append(lr)
        self.plot_learning_rate()

    def plot_learning_rate(self):
        self.ax_lr.clear()
        BasePlot(self.ax_lr, title='Learning Rate over Batches', xlabel='Batch', ylabel='Learning Rate')
        self.ax_lr.plot(self.batches, self.learning_rates, color='#2ca02c', marker='o')
        self.canvas.draw()

    def reset_visualization(self):
        self.batches = []
        self.train_policy_losses = []
        self.train_value_losses = []
        self.training_accuracies = []
        self.learning_rates = []
        self.init_visualization()