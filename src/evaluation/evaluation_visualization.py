import numpy as np
from src.base.base_visualization import BasePlot, BaseVisualizationWidget


class EvaluationVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reset_visualization()

    def init_visualization(self):
        self.figure.clf()
        gs = self.figure.add_gridspec(1, 2, wspace=0.4)
        self.ax_accuracy = self.figure.add_subplot(gs[0, 0])
        self.ax_confusion = self.figure.add_subplot(gs[0, 1])

        self.plots['accuracy_metrics'] = BasePlot(self.ax_accuracy, title='Model Accuracy and Metrics', ylabel='Scores')
        self.plots['confusion_matrix'] = BasePlot(self.ax_confusion, title='Confusion Matrix', xlabel='Predicted Label', ylabel='True Label')

        self.ax_accuracy.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', transform=self.ax_accuracy.transAxes, fontsize=12, fontweight='bold', color='gray')
        self.ax_confusion.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', transform=self.ax_confusion.transAxes, fontsize=12, fontweight='bold', color='gray')

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
        ax = self.plots['accuracy_metrics'].ax
        ax.clear()
        self.plots['accuracy_metrics'].apply_settings()
        labels = ['Top-1 Acc', 'Top-5 Acc', 'Precision', 'Recall', 'F1-Score']
        values = [
            self.accuracy * 100,
            self.topk_accuracy * 100,
            self.weighted_avg.get('precision', 0.0) * 100,
            self.weighted_avg.get('recall', 0.0) * 100,
            self.weighted_avg.get('f1-score', 0.0) * 100
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 100)
        for i, v in enumerate(values):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')

    def update_confusion_matrix_plot(self):
        ax = self.plots['confusion_matrix'].ax
        ax.clear()
        self.plots['confusion_matrix'].apply_settings()
        if self.confusion_matrix is not None and len(self.class_labels) > 0:
            im = ax.imshow(self.confusion_matrix, interpolation='nearest', cmap='Blues')
            self.figure.colorbar(im, ax=ax)
            num_classes = len(self.class_labels)
            thresh = self.confusion_matrix.max() / 2.
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(j, i, format(self.confusion_matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if self.confusion_matrix[i, j] > thresh else "black")
            ax.set_xticks(np.arange(num_classes))
            ax.set_yticks(np.arange(num_classes))
            ax.set_xticklabels(self.class_labels, rotation=45, ha="right")
            ax.set_yticklabels(self.class_labels)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
        else:
            ax.text(0.5, 0.5, 'Confusion Matrix Not Available',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color='gray')

    def reset_visualization(self):
        self.accuracy = 0.0
        self.topk_accuracy = 0.0
        self.macro_avg = {}
        self.weighted_avg = {}
        self.confusion_matrix = None
        self.class_labels = []
        super().reset_visualization()