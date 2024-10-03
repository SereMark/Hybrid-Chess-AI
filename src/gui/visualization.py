import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ChessAIVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.figure.patch.set_facecolor('#2b2b2b')
        self.canvas.setStyleSheet("background-color: #2b2b2b;")
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        self.move_evaluations = []

    def visualize_evaluation(self, evaluations):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.plot(evaluations, color='cyan', marker='o', label='Evaluation Score')
        ax.set_xlabel('Move Number', color='white')
        ax.set_ylabel('Evaluation Score', color='white')
        ax.set_title('Evaluation Score Progression', color='white')
        ax.legend(facecolor='#2b2b2b', edgecolor='#2b2b2b', labelcolor='white')
        self.canvas.draw()

    def clear_visualization(self):
        self.move_evaluations.clear()
        self.figure.clear()
        self.canvas.draw()