from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

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
        self.figure = Figure(figsize=(15, 5))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.init_visualization()

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(1, 3)
        self.ax_accuracy = self.figure.add_subplot(gs[0, 0])
        self.ax_metrics = self.figure.add_subplot(gs[0, 1])
        self.ax_confusion = self.figure.add_subplot(gs[0, 2])

        self.ax_accuracy.bar(['Accuracy'], [0])
        self.ax_accuracy.set_ylim(0, 1)
        self.ax_accuracy.set_title('Model Accuracy')
        self.ax_accuracy.set_ylabel('Accuracy')

        self.ax_metrics.text(0.5, 0.5, 'Metrics\nNot Available', ha='center', va='center',
                             transform=self.ax_metrics.transAxes, fontsize=14, fontweight='bold')

        self.ax_confusion.text(0.5, 0.5, 'Confusion Matrix\nNot Available', ha='center', va='center',
                               transform=self.ax_confusion.transAxes, fontsize=14, fontweight='bold')

        self.canvas.draw()

    def update_metrics_visualization(self, accuracy, macro_avg, weighted_avg, confusion_matrix, topk_accuracy=None, class_labels=None):
        self.accuracy = accuracy
        self.macro_avg = macro_avg
        self.weighted_avg = weighted_avg
        self.confusion_matrix = confusion_matrix
        self.topk_accuracy = topk_accuracy if topk_accuracy else 0.0
        self.class_labels = class_labels if class_labels else []

        self.figure.clear()
        gs = self.figure.add_gridspec(1, 3)
        self.ax_accuracy = self.figure.add_subplot(gs[0, 0])
        self.ax_metrics = self.figure.add_subplot(gs[0, 1])
        self.ax_confusion = self.figure.add_subplot(gs[0, 2])

        accuracies = [self.accuracy]
        labels = ['Top-1 Accuracy']
        if self.topk_accuracy > 0:
            accuracies.append(self.topk_accuracy)
            labels.append('Top-5 Accuracy')
        self.ax_accuracy.bar(labels, accuracies, color=['blue', 'green'])
        self.ax_accuracy.set_ylim(0, 1)
        self.ax_accuracy.set_title('Model Accuracy')
        self.ax_accuracy.set_ylabel('Accuracy')
        for i, v in enumerate(accuracies):
            self.ax_accuracy.text(i, v + 0.01, f"{v*100:.2f}%", ha='center')

        metrics_labels = ['Precision', 'Recall', 'F1-Score']
        macro_values = [self.macro_avg.get(metric, 0.0) for metric in ['precision', 'recall', 'f1-score']]
        weighted_values = [self.weighted_avg.get(metric, 0.0) for metric in ['precision', 'recall', 'f1-score']]

        x = np.arange(len(metrics_labels))
        width = 0.35
        rects1 = self.ax_metrics.bar(x - width/2, macro_values, width, label='Macro Avg', color='skyblue')
        rects2 = self.ax_metrics.bar(x + width/2, weighted_values, width, label='Weighted Avg', color='orange')

        self.ax_metrics.set_ylabel('Scores')
        self.ax_metrics.set_title('Aggregate Metrics')
        self.ax_metrics.set_xticks(x)
        self.ax_metrics.set_xticklabels(metrics_labels)
        self.ax_metrics.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                self.ax_metrics.annotate(f'{height*100:.2f}%',
                                         xy=(rect.get_x() + rect.get_width() / 2, height),
                                         xytext=(0, 3),
                                         textcoords="offset points",
                                         ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        if self.confusion_matrix is not None and len(self.class_labels) > 0:
            im = self.ax_confusion.imshow(self.confusion_matrix, interpolation='nearest', cmap='Blues')
            self.ax_confusion.set_title('Confusion Matrix')
            self.ax_confusion.set_xlabel('Predicted Label')
            self.ax_confusion.set_ylabel('True Label')
            cbar = self.figure.colorbar(im, ax=self.ax_confusion)
            cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
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

        self.figure.tight_layout()
        self.canvas.draw()