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
        gs = self.figure.add_gridspec(1, 3, wspace=0.4)
        self.ax_accuracy = self.figure.add_subplot(gs[0, 0])
        self.ax_metrics = self.figure.add_subplot(gs[0, 1])
        self.ax_confusion = self.figure.add_subplot(gs[0, 2])

        BasePlot(self.ax_accuracy, title='Model Accuracy', xlabel='', ylabel='Accuracy')
        BasePlot(self.ax_metrics, title='Aggregate Metrics', xlabel='', ylabel='Scores')
        BasePlot(self.ax_confusion, title='Confusion Matrix', xlabel='Predicted Label', ylabel='True Label')

        for ax in [self.ax_accuracy, self.ax_metrics, self.ax_confusion]:
            ax.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, fontweight='bold', color='gray')

        self.canvas.draw()

    def update_metrics_visualization(self, accuracy, macro_avg, weighted_avg, confusion_matrix, topk_accuracy=None, class_labels=None):
        self.accuracy = accuracy
        self.macro_avg = macro_avg
        self.weighted_avg = weighted_avg
        self.confusion_matrix = confusion_matrix
        self.topk_accuracy = topk_accuracy if topk_accuracy else 0.0
        self.class_labels = class_labels if class_labels else []
        self.figure.clear()
        gs = self.figure.add_gridspec(1, 3, wspace=0.4)
        self.ax_accuracy = self.figure.add_subplot(gs[0, 0])
        self.ax_metrics = self.figure.add_subplot(gs[0, 1])
        self.ax_confusion = self.figure.add_subplot(gs[0, 2])

        BasePlot(self.ax_accuracy, title='Model Accuracy', xlabel='', ylabel='Accuracy')
        BasePlot(self.ax_metrics, title='Aggregate Metrics', xlabel='', ylabel='Scores')
        BasePlot(self.ax_confusion, title='Confusion Matrix', xlabel='Predicted Label', ylabel='True Label')

        labels = ['Top-1 Accuracy']
        accuracies = [self.accuracy]
        if self.topk_accuracy > 0:
            labels.append('Top-5 Accuracy')
            accuracies.append(self.topk_accuracy)
        colors = ['#1f77b4', '#ff7f0e']
        self.ax_accuracy.bar(labels, accuracies, color=colors)
        self.ax_accuracy.set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            self.ax_accuracy.text(i, v + 0.01, f"{v*100:.2f}%", ha='center', fontsize=9, fontweight='bold')

        metrics_labels = ['Precision', 'Recall', 'F1-Score']
        macro_values = [self.macro_avg.get(metric, 0.0) for metric in ['precision', 'recall', 'f1-score']]
        weighted_values = [self.weighted_avg.get(metric, 0.0) for metric in ['precision', 'recall', 'f1-score']]

        x = np.arange(len(metrics_labels))
        width = 0.35
        rects1 = self.ax_metrics.bar(x - width/2, macro_values, width, label='Macro Avg', color='#17becf')
        rects2 = self.ax_metrics.bar(x + width/2, weighted_values, width, label='Weighted Avg', color='#bcbd22')

        self.ax_metrics.set_ylabel('Scores')
        self.ax_metrics.set_title('Aggregate Metrics')
        self.ax_metrics.set_xticks(x)
        self.ax_metrics.set_xticklabels(metrics_labels)
        self.ax_metrics.legend(fontsize=9)

        for rect in rects1 + rects2:
            height = rect.get_height()
            self.ax_metrics.annotate(f'{height*100:.2f}%',
                                     xy=(rect.get_x() + rect.get_width() / 2, height),
                                     xytext=(0, 3),
                                     textcoords="offset points",
                                     ha='center', va='bottom', fontsize=8, fontweight='bold')

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
        else:
            self.ax_confusion.text(0.5, 0.5, 'Confusion Matrix Not Available',
                                   ha='center', va='center',
                                   transform=self.ax_confusion.transAxes,
                                   fontsize=12, fontweight='bold', color='gray')

        self.canvas.draw()

    def reset_visualization(self):
        self.accuracy = 0.0
        self.topk_accuracy = 0.0
        self.macro_avg = {}
        self.weighted_avg = {}
        self.confusion_matrix = None
        self.class_labels = []
        self.init_visualization()