from src.base.base_visualization import BasePlot, BaseVisualizationWidget
import numpy as np

class EvaluationVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reset_visualization()

    def init_visualization(self):
        grid_spec = self.figure.add_gridspec(1, 2, wspace=0.4)

        # Define subplots
        self.ax_accuracy = self.figure.add_subplot(grid_spec[0, 0])
        self.ax_confusion = self.figure.add_subplot(grid_spec[0, 1])

        # Configure plots
        self.plots['accuracy_metrics'] = BasePlot(self.ax_accuracy, title='Model Accuracy Metrics', ylabel='Scores')
        self.plots['confusion_matrix'] = BasePlot(self.ax_confusion, title='Confusion Matrix', xlabel='Predicted Label', ylabel='True Label')

        # Add placeholders
        self.add_text_to_axis('accuracy_metrics', 'No Data Yet')
        self.add_text_to_axis('confusion_matrix', 'No Data Yet')

    def update_metrics_visualization(self, accuracy, topk_accuracy, confusion_matrix, class_labels):
        self.accuracy = accuracy
        self.topk_accuracy = topk_accuracy
        self.confusion_matrix = confusion_matrix
        self.class_labels = class_labels

        # Update accuracy metrics plot
        self.clear_axis('accuracy_metrics')

        if self.accuracy is not None and self.topk_accuracy is not None:
            labels = ['Top-1 Accuracy', 'Top-5 Accuracy']
            values = [self.accuracy * 100, self.topk_accuracy * 100]

            # Create bar plot
            self.ax_accuracy.bar(labels, values, alpha=0.8)
            self.ax_accuracy.set_ylim(0, 100)

            # Add value annotations
            for i, value in enumerate(values):
                self.ax_accuracy.text(i, value + 1, f"{value:.1f}%", ha='center', fontsize=9, fontweight='bold', color='#333333')
        else:
            self.add_text_to_axis('accuracy_metrics', 'No Data Yet')

        # Update confusion matrix plot
        self.clear_axis('confusion_matrix')

        if self.confusion_matrix is not None and len(self.class_labels) > 0:
            # Display heatmap
            im = self.ax_confusion.imshow(self.confusion_matrix, interpolation='nearest', cmap='Blues')
            cbar = self.figure.colorbar(im, ax=self.ax_confusion)
            cbar.ax.tick_params(colors='#333333')

            # Add text annotations for matrix values
            num_classes = len(self.class_labels)
            thresh = self.confusion_matrix.max() / 2.0
            for i in range(num_classes):
                for j in range(num_classes):
                    value = self.confusion_matrix[i, j]
                    self.ax_confusion.text(j, i, format(value, 'd'), ha="center", va="center", color="white" if value > thresh else "black", fontsize=9)

            # Configure axis labels and ticks
            self.ax_confusion.set_xticks(np.arange(num_classes))
            self.ax_confusion.set_yticks(np.arange(num_classes))
            self.ax_confusion.set_xticklabels(self.class_labels, rotation=45, ha="right", fontsize=9, color='#333333')
            self.ax_confusion.set_yticklabels(self.class_labels, fontsize=9, color='#333333')
            self.ax_confusion.set_xlabel('Predicted Label', fontsize=12, color='#333333')
            self.ax_confusion.set_ylabel('True Label', fontsize=12, color='#333333')
        else:
            self.add_text_to_axis('confusion_matrix', 'Confusion Matrix Not Available')

        self.update_visualization()

    def reset_visualization(self):
        self.accuracy = 0.0
        self.topk_accuracy = 0.0
        self.confusion_matrix = None
        self.class_labels = []
        super().reset_visualization()