from src.base.base_visualization import BasePlot, BaseVisualizationWidget
import numpy as np


class EvaluationVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reset_visualization()

    def init_visualization(self):
        gs = self.figure.add_gridspec(1, 2, wspace=0.4)
        self.ax_accuracy = self.figure.add_subplot(gs[0, 0])
        self.ax_confusion = self.figure.add_subplot(gs[0, 1])

        self.plots['accuracy_metrics'] = BasePlot(self.ax_accuracy, title='Model Accuracy and Metrics', ylabel='Scores')
        self.plots['confusion_matrix'] = BasePlot(self.ax_confusion, title='Confusion Matrix', xlabel='Predicted Label', ylabel='True Label')

        self.add_text_to_axis('accuracy_metrics', 'No Data Yet')
        self.add_text_to_axis('confusion_matrix', 'No Data Yet')

    def update_metrics_visualization(self, accuracy, topk_accuracy, macro_avg, weighted_avg, confusion_matrix, class_labels):
        self.accuracy = accuracy
        self.topk_accuracy = topk_accuracy
        self.macro_avg = macro_avg
        self.weighted_avg = weighted_avg
        self.confusion_matrix = confusion_matrix
        self.class_labels = class_labels
        self.update_accuracy_metrics_plot()
        self.update_confusion_matrix_plot()
        self.update_visualization()

    def update_accuracy_metrics_plot(self):
        self.clear_axis('accuracy_metrics')
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
        self.clear_axis('confusion_matrix')
        if self.confusion_matrix is not None and len(self.class_labels) > 0:
            im = self.ax_confusion.imshow(self.confusion_matrix, interpolation='nearest', cmap='Blues')
            self.figure.colorbar(im, ax=self.ax_confusion)
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
            self.add_text_to_axis('confusion_matrix', 'Confusion Matrix Not Available')

    def reset_visualization(self):
        self.accuracy = 0.0
        self.topk_accuracy = 0.0
        self.macro_avg = {}
        self.weighted_avg = {}
        self.confusion_matrix = None
        self.class_labels = []
        super().reset_visualization()