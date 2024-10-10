from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class EvaluationVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.accuracy = 0.0
        self.macro_avg = {}
        self.weighted_avg = {}
        self.init_ui()
    
    def init_ui(self):
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.init_visualization()
    
    def init_visualization(self):
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax1.bar(['Accuracy'], [self.accuracy])
        ax1.set_ylim(0, 1)
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.text(0, self.accuracy + 0.01, f"{self.accuracy*100:.2f}%", ha='center')
    
        ax2 = self.figure.add_subplot(1, 2, 2)
        ax2.text(0.5, 0.5, 'Aggregate Metrics\nNot Available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14, fontweight='bold')
    
        self.canvas.draw()
    
    def update_metrics_visualization(self, accuracy, macro_avg, weighted_avg):
        self.accuracy = accuracy
        self.macro_avg = macro_avg
        self.weighted_avg = weighted_avg
    
        self.figure.clear()
    
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax1.bar(['Accuracy'], [self.accuracy])
        ax1.set_ylim(0, 1)
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.text(0, self.accuracy + 0.01, f"{self.accuracy*100:.2f}%", ha='center')
    
        ax2 = self.figure.add_subplot(1, 2, 2)
        labels = ['Precision', 'Recall', 'F1-Score']
        macro_values = [self.macro_avg.get(metric, 0.0) for metric in ['precision', 'recall', 'f1-score']]
        weighted_values = [self.weighted_avg.get(metric, 0.0) for metric in ['precision', 'recall', 'f1-score']]
    
        x = np.arange(len(labels))
        width = 0.35
        rects1 = ax2.bar(x - width/2, macro_values, width, label='Macro Avg')
        rects2 = ax2.bar(x + width/2, weighted_values, width, label='Weighted Avg')
    
        ax2.set_ylabel('Scores')
        ax2.set_title('Aggregate Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
    
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax2.annotate(f'{height*100:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
        autolabel(rects1)
        autolabel(rects2)
    
        self.canvas.draw()