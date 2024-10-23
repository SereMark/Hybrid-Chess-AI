import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class BasePlot:
    def __init__(self, ax, title, xlabel='', ylabel='', invert_y=False):
        self.ax = ax
        self.ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        self.ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
        self.ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
        self.ax.tick_params(axis='both', which='major', labelsize=9)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        if invert_y:
            self.ax.invert_yaxis()

class EvaluationVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.accuracy = 0.0
        self.topk_accuracy = 0.0
        self.macro_avg = {}
        self.weighted_avg = {}
        self.confusion_matrix = None
        self.class_labels = []
        self.init_ui()

    def init_ui(self):
        self.figure = Figure(figsize=(15, 8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.init_visualization()

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(1, 2, wspace=0.4)
        self.ax_accuracy = self.figure.add_subplot(gs[0, 0])
        self.ax_confusion = self.figure.add_subplot(gs[0, 1])

        BasePlot(self.ax_accuracy, title='Model Accuracy and Metrics', xlabel='', ylabel='Scores')
        BasePlot(self.ax_confusion, title='Confusion Matrix', xlabel='Predicted Label', ylabel='True Label')

        self.ax_accuracy.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                              transform=self.ax_accuracy.transAxes, fontsize=12, fontweight='bold', color='gray')
        self.ax_confusion.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                               transform=self.ax_confusion.transAxes, fontsize=12, fontweight='bold', color='gray')

        self.canvas.draw()

    def update_metrics_visualization(self, accuracy, topk_accuracy, macro_avg, weighted_avg, confusion_matrix, class_labels):
        self.accuracy = accuracy
        self.topk_accuracy = topk_accuracy
        self.macro_avg = macro_avg
        self.weighted_avg = weighted_avg
        self.confusion_matrix = confusion_matrix
        self.class_labels = class_labels

        self.update_accuracy_metrics_plot()
        self.update_confusion_matrix_plot()
        self.canvas.draw()

    def update_accuracy_metrics_plot(self):
        self.ax_accuracy.clear()
        BasePlot(self.ax_accuracy, title='Model Accuracy and Metrics', xlabel='', ylabel='Scores')

        labels = ['Top-1 Acc', 'Top-5 Acc', 'Precision', 'Recall', 'F1-Score']
        values = [
            self.accuracy * 100,
            self.topk_accuracy * 100,
            self.weighted_avg.get('precision', 0.0) * 100,
            self.weighted_avg.get('recall', 0.0) * 100,
            self.weighted_avg.get('f1-score', 0.0) * 100
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.ax_accuracy.bar(labels, values, color=colors)
        self.ax_accuracy.set_ylim(0, 100)
        for i, v in enumerate(values):
            self.ax_accuracy.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')

    def update_confusion_matrix_plot(self):
        self.ax_confusion.clear()
        BasePlot(self.ax_confusion, title='Confusion Matrix', xlabel='Predicted Label', ylabel='True Label')
        if self.confusion_matrix is not None and len(self.class_labels) > 0:
            im = self.ax_confusion.imshow(self.confusion_matrix, interpolation='nearest', cmap='Blues')
            self.ax_confusion.figure.colorbar(im, ax=self.ax_confusion)
            num_classes = len(self.class_labels)
            thresh = self.confusion_matrix.max() / 2.
            for i in range(num_classes):
                for j in range(num_classes):
                    self.ax_confusion.text(j, i, format(self.confusion_matrix[i, j], 'd'),
                                           ha="center", va="center",
                                           color="white" if self.confusion_matrix[i, j] > thresh else "black")
            self.ax_confusion.set_xticks(np.arange(num_classes))
            self.ax_confusion.set_yticks(np.arange(num_classes))
            self.ax_confusion.set_xticklabels(self.class_labels, rotation=45, ha="right")
            self.ax_confusion.set_yticklabels(self.class_labels)
            self.ax_confusion.set_xlabel('Predicted Label')
            self.ax_confusion.set_ylabel('True Label')
        else:
            self.ax_confusion.text(0.5, 0.5, 'Confusion Matrix Not Available',
                                   ha='center', va='center',
                                   transform=self.ax_confusion.transAxes,
                                   fontsize=12, fontweight='bold', color='gray')

    def reset_visualization(self):
        self.accuracy = 0.0
        self.topk_accuracy = 0.0
        self.macro_avg = {}
        self.weighted_avg = {}
        self.confusion_matrix = None
        self.class_labels = []
        self.init_visualization()