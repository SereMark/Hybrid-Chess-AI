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

class TrainingVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.train_policy_losses = []
        self.train_value_losses = []
        self.val_policy_losses = []
        self.val_value_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.init_ui()

    def init_ui(self):
        self.figure = Figure(figsize=(14, 10), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.init_visualization()

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(3, 2, wspace=0.3, hspace=0.4)

        self.ax_train_policy_loss = self.figure.add_subplot(gs[0, 0])
        BasePlot(self.ax_train_policy_loss, title='Training Policy Loss', xlabel='Batch', ylabel='Loss')
        self.ax_train_policy_loss.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                                       transform=self.ax_train_policy_loss.transAxes,
                                       fontsize=12, fontweight='bold', color='gray')

        self.ax_train_value_loss = self.figure.add_subplot(gs[0, 1])
        BasePlot(self.ax_train_value_loss, title='Training Value Loss', xlabel='Batch', ylabel='Loss')
        self.ax_train_value_loss.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                                     transform=self.ax_train_value_loss.transAxes,
                                     fontsize=12, fontweight='bold', color='gray')

        self.ax_val_policy_loss = self.figure.add_subplot(gs[1, 0])
        BasePlot(self.ax_val_policy_loss, title='Validation Policy Loss', xlabel='Batch', ylabel='Loss')
        self.ax_val_policy_loss.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                                     transform=self.ax_val_policy_loss.transAxes,
                                     fontsize=12, fontweight='bold', color='gray')

        self.ax_val_value_loss = self.figure.add_subplot(gs[1, 1])
        BasePlot(self.ax_val_value_loss, title='Validation Value Loss', xlabel='Batch', ylabel='Loss')
        self.ax_val_value_loss.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                                   transform=self.ax_val_value_loss.transAxes,
                                   fontsize=12, fontweight='bold', color='gray')

        self.ax_accuracy = self.figure.add_subplot(gs[2, :])
        BasePlot(self.ax_accuracy, title='Training and Validation Accuracy', xlabel='Epoch', ylabel='Accuracy')
        self.ax_accuracy.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                              transform=self.ax_accuracy.transAxes,
                              fontsize=12, fontweight='bold', color='gray')

        self.canvas.draw()

    def update_training_visualization(self, policy_loss, value_loss):
        self.train_policy_losses.append(policy_loss)
        self.train_value_losses.append(value_loss)
        self.update_train_loss_plots()

    def update_validation_visualization(self, val_policy_loss, val_value_loss):
        self.val_policy_losses.append(val_policy_loss)
        self.val_value_losses.append(val_value_loss)
        self.update_val_loss_plots()

    def update_train_loss_plots(self):
        if not self.train_policy_losses:
            self.ax_train_policy_loss.clear()
            BasePlot(self.ax_train_policy_loss, title='Training Policy Loss', xlabel='Batch', ylabel='Loss')
            self.ax_train_policy_loss.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                                           transform=self.ax_train_policy_loss.transAxes,
                                           fontsize=12, fontweight='bold', color='gray')
            self.canvas.draw()
            return

        batches = range(1, len(self.train_policy_losses) + 1)

        self.ax_train_policy_loss.clear()
        BasePlot(self.ax_train_policy_loss, title='Training Policy Loss', xlabel='Batch', ylabel='Loss')
        self.ax_train_policy_loss.plot(batches, self.train_policy_losses, label='Policy Loss', color='#1f77b4', linewidth=2)
        self.ax_train_policy_loss.legend(fontsize=9)
        self.ax_train_policy_loss.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        self.ax_train_value_loss.clear()
        BasePlot(self.ax_train_value_loss, title='Training Value Loss', xlabel='Batch', ylabel='Loss')
        self.ax_train_value_loss.plot(batches, self.train_value_losses, label='Value Loss', color='#2ca02c', linewidth=2)
        self.ax_train_value_loss.legend(fontsize=9)
        self.ax_train_value_loss.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        self.canvas.draw()

    def update_val_loss_plots(self):
        if not self.val_policy_losses:
            self.ax_val_policy_loss.clear()
            BasePlot(self.ax_val_policy_loss, title='Validation Policy Loss', xlabel='Batch', ylabel='Loss')
            self.ax_val_policy_loss.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                                         transform=self.ax_val_policy_loss.transAxes,
                                         fontsize=12, fontweight='bold', color='gray')
            self.canvas.draw()
            return

        batches = range(1, len(self.val_policy_losses) + 1)

        self.ax_val_policy_loss.clear()
        BasePlot(self.ax_val_policy_loss, title='Validation Policy Loss', xlabel='Batch', ylabel='Loss')
        self.ax_val_policy_loss.plot(batches, self.val_policy_losses, label='Validation Policy Loss', color='#ff7f0e', linestyle='--', linewidth=2)
        self.ax_val_policy_loss.legend(fontsize=9)
        self.ax_val_policy_loss.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        self.ax_val_value_loss.clear()
        BasePlot(self.ax_val_value_loss, title='Validation Value Loss', xlabel='Batch', ylabel='Loss')
        self.ax_val_value_loss.plot(batches, self.val_value_losses, label='Validation Value Loss', color='#d62728', linestyle='--', linewidth=2)
        self.ax_val_value_loss.legend(fontsize=9)
        self.ax_val_value_loss.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        self.canvas.draw()

    def update_accuracy_visualization(self, training_accuracy, validation_accuracy):
        self.training_accuracies.append(training_accuracy)
        self.validation_accuracies.append(validation_accuracy)
        self.update_accuracy_plot()

    def update_accuracy_plot(self):
        if not self.training_accuracies and not self.validation_accuracies:
            self.ax_accuracy.clear()
            BasePlot(self.ax_accuracy, title='Training and Validation Accuracy', xlabel='Epoch', ylabel='Accuracy')
            self.ax_accuracy.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                                  transform=self.ax_accuracy.transAxes,
                                  fontsize=12, fontweight='bold', color='gray')
            self.canvas.draw()
            return

        epochs = range(1, len(self.training_accuracies) + 1)

        self.ax_accuracy.clear()
        BasePlot(self.ax_accuracy, title='Training and Validation Accuracy', xlabel='Epoch', ylabel='Accuracy')
        self.ax_accuracy.plot(epochs, self.training_accuracies, label='Training Accuracy', color='#9467bd', linewidth=2)
        self.ax_accuracy.plot(epochs, self.validation_accuracies, label='Validation Accuracy', color='#8c564b', linestyle='--', linewidth=2)
        self.ax_accuracy.legend(fontsize=9)
        self.ax_accuracy.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        for i, (train_acc, val_acc) in enumerate(zip(self.training_accuracies, self.validation_accuracies), 1):
            self.ax_accuracy.text(i, train_acc + 0.01, f"{train_acc*100:.2f}%", ha='center', fontsize=8, fontweight='bold')
            self.ax_accuracy.text(i, val_acc + 0.01, f"{val_acc*100:.2f}%", ha='center', fontsize=8, fontweight='bold')

        self.canvas.draw()

    def reset_visualization(self):
        self.train_policy_losses = []
        self.train_value_losses = []
        self.val_policy_losses = []
        self.val_value_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.init_visualization()