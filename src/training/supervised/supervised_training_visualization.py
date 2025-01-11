from src.base.base_visualization import BasePlot, BaseVisualizationWidget
import numpy as np

class SupervisedVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reset_visualization()

    def init_visualization(self):
        grid_spec = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Subplots
        self.ax_policy_loss = self.figure.add_subplot(grid_spec[0, 0])
        self.ax_value_loss = self.figure.add_subplot(grid_spec[0, 1])
        self.ax_accuracy = self.figure.add_subplot(grid_spec[1, 0])
        self.ax_val_loss = self.figure.add_subplot(grid_spec[1, 1])

        # Plot configuration
        self.plots['policy_loss'] = BasePlot(self.ax_policy_loss, title='Policy Loss', xlabel='Batch', ylabel='Loss')
        self.plots['value_loss'] = BasePlot(self.ax_value_loss, title='Value Loss', xlabel='Batch', ylabel='Loss')
        self.plots['accuracy'] = BasePlot(self.ax_accuracy, title='Accuracy', xlabel='Batch', ylabel='Accuracy (%)')
        self.plots['val_loss'] = BasePlot(self.ax_val_loss, title='Validation Loss', xlabel='Epoch', ylabel='Loss')

        # Lines for plotting
        self.line_policy_loss_train, = self.ax_policy_loss.plot([], [], label='Train Policy', color='blue', alpha=0.8)
        self.line_value_loss_train, = self.ax_value_loss.plot([], [], label='Train Value', color='blue', alpha=0.8)
        self.line_accuracy_train, = self.ax_accuracy.plot([], [], label='Train Accuracy', color='blue', alpha=0.8)
        self.line_policy_loss_val, = self.ax_val_loss.plot([], [], label='Val Policy', color='orange', alpha=0.8)
        self.line_value_loss_val, = self.ax_val_loss.plot([], [], label='Val Value', color='green', alpha=0.8)

        # Add legends
        self.ax_policy_loss.legend(frameon=False, fontsize=10)
        self.ax_value_loss.legend(frameon=False, fontsize=10)
        self.ax_accuracy.legend(frameon=False, fontsize=10)
        self.ax_val_loss.legend(frameon=False, fontsize=10)

    def update_loss_plots(self, batch_idx, losses):
        if not all(isinstance(v, (float, int)) and np.isfinite(v) for v in losses.values()):
            return

        self.loss_batches.append(batch_idx)
        self.policy_losses_train.append(losses['policy'])
        self.value_losses_train.append(losses['value'])

        self.update_plot(self.line_policy_loss_train, self.loss_batches, self.policy_losses_train, 'policy_loss')
        self.update_plot(self.line_value_loss_train, self.loss_batches, self.value_losses_train, 'value_loss')

    def update_accuracy_plot(self, batch_idx, accuracy):
        if not isinstance(accuracy, (float, int)) and np.isfinite(accuracy):
            return

        self.accuracy_batches.append(batch_idx)
        self.accuracies_train.append(accuracy * 100)

        self.update_plot(self.line_accuracy_train, self.accuracy_batches, self.accuracies_train, 'accuracy')

    def update_validation_loss_plots(self, epoch_idx, losses):
        if not all(isinstance(v, (float, int)) and np.isfinite(v) for v in losses.values()):
            return

        self.epochs.append(epoch_idx)
        self.policy_losses_val.append(losses['policy'])
        self.value_losses_val.append(losses['value'])

        self.update_plot(self.line_policy_loss_val, self.epochs, self.policy_losses_val, 'val_loss')
        self.update_plot(self.line_value_loss_val, self.epochs, self.value_losses_val, 'val_loss')

    def update_validation_accuracy_plot(self, epoch_idx, accuracy):
        if not isinstance(accuracy, (float, int)) and np.isfinite(accuracy):
            return

        self.accuracy_epochs.append(epoch_idx)
        self.accuracies_val.append(accuracy * 100)

    def update_plot(self, line, x_data, y_data, plot_key):
        if line:
            line.set_data(x_data, y_data)
            ax = self.plots[plot_key].ax
            ax.relim()
            ax.autoscale_view()
            self.update_visualization()

    def reset_visualization(self):
        self.loss_batches = []
        self.accuracy_batches = []
        self.policy_losses_train = []
        self.value_losses_train = []
        self.accuracies_train = []
        self.epochs = []
        self.accuracy_epochs = []
        self.policy_losses_val = []
        self.value_losses_val = []
        self.accuracies_val = []
        super().reset_visualization()