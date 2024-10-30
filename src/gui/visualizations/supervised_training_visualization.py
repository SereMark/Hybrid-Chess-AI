from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
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


class SupervisedTrainingVisualization(QWidget):
    def __init__(self, parent=None, max_points=1000):
        super().__init__(parent)
        self.max_points = max_points
        self.loss_batches = []
        self.accuracy_batches = []
        self.policy_losses = []
        self.value_losses = []
        self.accuracies = []
        self.total_losses = []
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
        gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        self.ax_policy_loss = self.figure.add_subplot(gs[0, 0])
        BasePlot(self.ax_policy_loss, title='Policy Loss', xlabel='Batch', ylabel='Loss')
        
        self.ax_value_loss = self.figure.add_subplot(gs[0, 1])
        BasePlot(self.ax_value_loss, title='Value Loss', xlabel='Batch', ylabel='Loss')
        
        self.ax_total_loss = self.figure.add_subplot(gs[1, 0])
        BasePlot(self.ax_total_loss, title='Total Loss', xlabel='Batch', ylabel='Loss')
        
        self.ax_accuracy = self.figure.add_subplot(gs[1, 1])
        BasePlot(self.ax_accuracy, title='Accuracy', xlabel='Batch', ylabel='Accuracy (%)')
        
        self.figure.canvas.draw()

    def set_total_batches(self, total_batches):
        pass

    def update_loss_plots(self, batch_idx, losses):
        if not all(math.isfinite(v) for v in losses.values()):
            print(f"Non-finite loss values detected at batch {batch_idx}. Skipping plot update.")
            return
            
        self.loss_batches.append(batch_idx)
        self.policy_losses.append(losses['policy'])
        self.value_losses.append(losses['value'])
        self.total_losses.append(losses['policy'] + losses['value'])

        if len(self.loss_batches) > self.max_points:
            self.loss_batches = self.loss_batches[-self.max_points:]
            self.policy_losses = self.policy_losses[-self.max_points:]
            self.value_losses = self.value_losses[-self.max_points:]
            self.total_losses = self.total_losses[-self.max_points:]

        self.plot_all_losses()

    def update_accuracy_plot(self, batch_idx, accuracy):
        if not math.isfinite(accuracy):
            print(f"Non-finite accuracy value detected at batch {batch_idx}. Skipping plot update.")
            return

        self.accuracy_batches.append(batch_idx)
        self.accuracies.append(accuracy * 100)

        if len(self.accuracy_batches) > self.max_points:
            self.accuracy_batches = self.accuracy_batches[-self.max_points:]
            self.accuracies = self.accuracies[-self.max_points:]

        self.plot_accuracy()

    def plot_all_losses(self):
        try:
            self.ax_policy_loss.clear()
            BasePlot(self.ax_policy_loss, title='Policy Loss', xlabel='Batch', ylabel='Loss')
            self.ax_policy_loss.plot(
                self.loss_batches,
                self.policy_losses,
                color='#1f77b4',
                marker='o',
                markersize=2,
                linestyle='-',
                linewidth=1
            )

            self.ax_value_loss.clear()
            BasePlot(self.ax_value_loss, title='Value Loss', xlabel='Batch', ylabel='Loss')
            self.ax_value_loss.plot(
                self.loss_batches,
                self.value_losses,
                color='#ff7f0e',
                marker='o',
                markersize=2,
                linestyle='-',
                linewidth=1
            )

            self.ax_total_loss.clear()
            BasePlot(self.ax_total_loss, title='Total Loss', xlabel='Batch', ylabel='Loss')
            self.ax_total_loss.plot(
                self.loss_batches,
                self.total_losses,
                color='#2ca02c',
                marker='o',
                markersize=2,
                linestyle='-',
                linewidth=1
            )

            self.figure.canvas.draw_idle()
        except Exception as e:
            print(f"Error in plot_all_losses: {e}")

    def plot_accuracy(self):
        try:
            self.ax_accuracy.clear()
            BasePlot(self.ax_accuracy, title='Accuracy', xlabel='Batch', ylabel='Accuracy (%)')
            self.ax_accuracy.plot(
                self.accuracy_batches,
                self.accuracies,
                color='#9467bd',
                marker='o',
                markersize=2,
                linestyle='-',
                linewidth=1
            )
            self.figure.canvas.draw_idle()
        except Exception as e:
            print(f"Error in plot_accuracy: {e}")

    def reset_visualization(self):
        self.loss_batches = []
        self.accuracy_batches = []
        self.policy_losses = []
        self.value_losses = []
        self.accuracies = []
        self.total_losses = []
        self.init_visualization()