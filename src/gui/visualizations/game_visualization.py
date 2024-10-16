import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ValueEvaluationPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.ax.set_title("Value Evaluation")
        self.ax.set_xlabel("Move Number")
        self.ax.set_ylabel("Evaluation Score")
        self.ax.grid(True)

    def plot_evaluations(self, evaluations):
        self.ax.clear()
        self.ax.set_title("Value Evaluation")
        self.ax.set_xlabel("Move Number")
        self.ax.set_ylabel("Evaluation Score")
        self.ax.grid(True)

        if evaluations:
            moves = list(range(1, len(evaluations) + 1))
            self.ax.plot(moves, evaluations, marker='o', linewidth=2, label='Evaluation')

            if len(evaluations) >= 5:
                moving_avg = np.convolve(evaluations, np.ones(5)/5, mode='valid')
                self.ax.plot(moves[4:], moving_avg, linestyle='--', color='orange', label='5-Move Moving Avg')

            self.ax.annotate(f'{evaluations[-1]:.2f}', xy=(moves[-1], evaluations[-1]),
                             xytext=(moves[-1], evaluations[-1] + 0.05 * max(evaluations)),
                             ha='center', fontsize=10)

            self.ax.legend()
        else:
            self.ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start',
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=14, fontweight='bold')

        self.canvas.draw()

class PolicyOutputPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.ax.set_title("Policy Output")
        self.ax.set_xlabel("Probability (%)")
        self.ax.set_ylabel("Move")
        self.ax.invert_yaxis()

    def plot_policy_output(self, policy_output):
        self.ax.clear()
        self.ax.set_title("Policy Output")
        self.ax.set_xlabel("Probability (%)")
        self.ax.set_ylabel("Move")
        self.ax.invert_yaxis()

        if policy_output:
            sorted_moves = sorted(policy_output.items(), key=lambda x: x[1], reverse=True)[:5]
            moves, probabilities = zip(*sorted_moves)
            probabilities_percent = [prob * 100 for prob in probabilities]
            y_pos = np.arange(len(moves))

            bars = self.ax.barh(y_pos, probabilities_percent, color=plt.cm.Blues(np.array(probabilities_percent) / 100))
            self.ax.set_yticks(y_pos)
            self.ax.set_yticklabels(moves)
            self.ax.set_xlim(0, max(probabilities_percent) + 10)

            for bar, prob in zip(bars, probabilities_percent):
                self.ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                             f'{prob:.1f}%', va='center', fontsize=10)

        else:
            self.ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start',
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=14, fontweight='bold')

        self.canvas.draw()

class MCTSStatisticsPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.ax.set_title("MCTS Statistics")
        self.ax.set_xlabel("Metric")
        self.ax.set_ylabel("Value")
        self.ax.grid(True)

    def plot_mcts_statistics(self, mcts_stats):
        self.ax.clear()
        self.ax.set_title("MCTS Statistics")
        self.ax.set_xlabel("Metric")
        self.ax.set_ylabel("Value")
        self.ax.grid(True)

        if mcts_stats:
            metrics = ['Simulations', 'Nodes Explored']
            values = [mcts_stats.get('simulations', 0), mcts_stats.get('nodes_explored', 0)]
            y_pos = np.arange(len(metrics))
            max_val = max(values) if max(values) > 0 else 1

            bars = self.ax.bar(y_pos, values, color=plt.cm.Oranges(np.array(values) / max_val))
            self.ax.set_xticks(y_pos)
            self.ax.set_xticklabels(metrics)

            for bar, val in zip(bars, values):
                self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.01,
                             f'{val}', ha='center', va='bottom', fontsize=10)

            if 'best_move' in mcts_stats and mcts_stats['best_move']:
                best_move = mcts_stats['best_move']
                self.ax.annotate(f'Best Move: {best_move}', xy=(0.5, 1.05), xycoords='axes fraction',
                                 ha='center', fontsize=12)
        else:
            self.ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start',
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=14, fontweight='bold')

        self.canvas.draw()

class GameVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.evaluations = []

    def init_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        self.value_evaluation_plot = ValueEvaluationPlot()
        self.policy_output_plot = PolicyOutputPlot()
        self.mcts_statistics_plot = MCTSStatisticsPlot()

        tabs.addTab(self.value_evaluation_plot, "Value Evaluation")
        tabs.addTab(self.policy_output_plot, "Policy Output")
        tabs.addTab(self.mcts_statistics_plot, "MCTS Statistics")

        layout.addWidget(tabs)
        self.setLayout(layout)

    def update_value_evaluation(self, evaluations):
        self.evaluations.extend(evaluations)
        self.value_evaluation_plot.plot_evaluations(self.evaluations)

    def update_policy_output(self, policy_output):
        self.policy_output_plot.plot_policy_output(policy_output)

    def update_mcts_statistics(self, mcts_stats):
        self.mcts_statistics_plot.plot_mcts_statistics(mcts_stats)

    def reset_visualizations(self):
        self.evaluations = []
        self.value_evaluation_plot.plot_evaluations(self.evaluations)