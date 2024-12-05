from src.base.base_visualization import BasePlot, BaseVisualizationWidget
import numpy as np


class SupervisedVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None, max_points=1000):
        self.max_points = max_points
        super().__init__(parent)
        self.reset_visualization()

    def init_visualization(self):
        gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        self.ax_policy_loss = self.figure.add_subplot(gs[0, 0])
        self.ax_value_loss = self.figure.add_subplot(gs[0, 1])
        self.ax_total_loss = self.figure.add_subplot(gs[1, 0])
        self.ax_accuracy = self.figure.add_subplot(gs[1, 1])

        self.plots['policy_loss'] = BasePlot(self.ax_policy_loss, title='Policy Loss', xlabel='Batch/Epoch', ylabel='Loss')
        self.plots['value_loss'] = BasePlot(self.ax_value_loss, title='Value Loss', xlabel='Batch/Epoch', ylabel='Loss')
        self.plots['total_loss'] = BasePlot(self.ax_total_loss, title='Total Loss', xlabel='Batch/Epoch', ylabel='Loss')
        self.plots['accuracy'] = BasePlot(self.ax_accuracy, title='Accuracy', xlabel='Batch/Epoch', ylabel='Accuracy (%)')

        self.line_policy_loss_train, = self.ax_policy_loss.plot([], [], label='Training Policy Loss', color='blue')
        self.line_policy_loss_val, = self.ax_policy_loss.plot([], [], label='Validation Policy Loss', color='orange')
        self.ax_policy_loss.legend()

        self.line_value_loss_train, = self.ax_value_loss.plot([], [], label='Training Value Loss', color='blue')
        self.line_value_loss_val, = self.ax_value_loss.plot([], [], label='Validation Value Loss', color='orange')
        self.ax_value_loss.legend()

        self.line_total_loss_train, = self.ax_total_loss.plot([], [], label='Training Total Loss', color='blue')
        self.line_total_loss_val, = self.ax_total_loss.plot([], [], label='Validation Total Loss', color='orange')
        self.ax_total_loss.legend()

        self.line_accuracy_train, = self.ax_accuracy.plot([], [], label='Training Accuracy', color='blue')
        self.line_accuracy_val, = self.ax_accuracy.plot([], [], label='Validation Accuracy', color='orange')
        self.ax_accuracy.legend()

    def update_loss_plots(self, batch_idx, losses):
        if not all(isinstance(v, (float, int)) and not np.isnan(v) and np.isfinite(v) for v in losses.values()):
            return
        self.loss_batches.append(batch_idx)
        self.policy_losses_train.append(losses['policy'])
        self.value_losses_train.append(losses['value'])
        self.total_losses_train.append(losses['policy'] + losses['value'])
        if len(self.loss_batches) > self.max_points:
            self.loss_batches = self.loss_batches[-self.max_points:]
            self.policy_losses_train = self.policy_losses_train[-self.max_points:]
            self.value_losses_train = self.value_losses_train[-self.max_points:]
            self.total_losses_train = self.total_losses_train[-self.max_points:]
        self.update_plot(self.line_policy_loss_train, self.loss_batches, self.policy_losses_train, 'policy_loss')
        self.update_plot(self.line_value_loss_train, self.loss_batches, self.value_losses_train, 'value_loss')
        self.update_plot(self.line_total_loss_train, self.loss_batches, self.total_losses_train, 'total_loss')

    def update_accuracy_plot(self, batch_idx, accuracy):
        if not isinstance(accuracy, (float, int)) or np.isnan(accuracy) or not np.isfinite(accuracy):
            return
        self.accuracy_batches.append(batch_idx)
        self.accuracies_train.append(accuracy * 100)
        if len(self.accuracy_batches) > self.max_points:
            self.accuracy_batches = self.accuracy_batches[-self.max_points:]
            self.accuracies_train = self.accuracies_train[-self.max_points:]
        self.update_plot(self.line_accuracy_train, self.accuracy_batches, self.accuracies_train, 'accuracy')

    def update_validation_loss_plots(self, epoch, losses):
        if not all(isinstance(v, (float, int)) and not np.isnan(v) and np.isfinite(v) for v in losses.values()):
            return
        self.epochs.append(epoch)
        self.policy_losses_val.append(losses['policy'])
        self.value_losses_val.append(losses['value'])
        self.total_losses_val.append(losses['policy'] + losses['value'])
        if len(self.epochs) > self.max_points:
            self.epochs = self.epochs[-self.max_points:]
            self.policy_losses_val = self.policy_losses_val[-self.max_points:]
            self.value_losses_val = self.value_losses_val[-self.max_points:]
            self.total_losses_val = self.total_losses_val[-self.max_points:]
        self.update_plot(self.line_policy_loss_val, self.epochs, self.policy_losses_val, 'policy_loss')
        self.update_plot(self.line_value_loss_val, self.epochs, self.value_losses_val, 'value_loss')
        self.update_plot(self.line_total_loss_val, self.epochs, self.total_losses_val, 'total_loss')

    def update_validation_accuracy_plot(self, epoch, accuracy):
        if not isinstance(accuracy, (float, int)) or np.isnan(accuracy) or not np.isfinite(accuracy):
            return
        self.accuracy_epochs.append(epoch)
        self.accuracies_val.append(accuracy * 100)
        if len(self.accuracy_epochs) > self.max_points:
            self.accuracy_epochs = self.accuracy_epochs[-self.max_points:]
            self.accuracies_val = self.accuracies_val[-self.max_points:]
        self.update_plot(self.line_accuracy_val, self.accuracy_epochs, self.accuracies_val, 'accuracy')

    def update_plot(self, line, x_data, y_data, plot_key):
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
        self.total_losses_train = []
        self.accuracies_train = []

        self.epochs = []
        self.accuracy_epochs = []
        self.policy_losses_val = []
        self.value_losses_val = []
        self.total_losses_val = []
        self.accuracies_val = []

        super().reset_visualization()