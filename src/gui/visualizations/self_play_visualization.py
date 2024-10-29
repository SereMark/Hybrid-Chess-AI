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

class SelfPlayVisualization(QWidget):
    def __init__(self, parent=None):
        super(SelfPlayVisualization, self).__init__(parent)
        
        self.layout = QVBoxLayout(self)
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        self.layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        self.plot = BasePlot(self.ax, 'Self Play Visualization', 'X-axis', 'Y-axis')
        
        self.setLayout(self.layout)
    
    def update_plot(self, x_data, y_data):
        self.ax.clear()
        self.plot = BasePlot(self.ax, 'Self Play Visualization', 'X-axis', 'Y-axis')
        self.ax.plot(x_data, y_data, 'r-')
        self.canvas.draw()