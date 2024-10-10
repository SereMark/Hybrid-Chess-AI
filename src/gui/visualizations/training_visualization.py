from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TrainingVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.policy_losses = []
        self.value_losses = []
        self.init_ui()
    
    def init_ui(self):
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.init_visualization()
    
    def init_visualization(self):
        titles = ['Policy Loss', 'Value Loss']

        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)

        self.axes = [ax1, ax2]

        for ax, title in zip(self.axes, titles):
            ax.set_title(title)
            ax.set_xlabel('Batch')
            ax.set_ylabel('Loss')
            ax.grid(True)

        self.canvas.draw()
    
    def update_training_visualization(self, policy_loss, value_loss):
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)

        ax1 = self.axes[0]
        ax1.clear()
        ax1.plot(self.policy_losses)
        ax1.set_title('Policy Loss')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2 = self.axes[1]
        ax2.clear()
        ax2.plot(self.value_losses)
        ax2.set_title('Value Loss')
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)

        self.canvas.draw()