import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ValueEvaluationPlot:
    def __init__(self):
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

    def plot_evaluations(self, evaluations):
        self.ax.clear()
        if evaluations:
            moves = list(range(1, len(evaluations) + 1))
            self.ax.plot(moves, evaluations, marker='o', label='Evaluation', linewidth=2, markersize=6)
            window_size = min(5, len(evaluations))
            moving_avg = np.convolve(evaluations, np.ones(window_size)/window_size, mode='valid')
            self.ax.plot(moves[window_size-1:], moving_avg, linestyle='--', label='Moving Average')
            latest_move = moves[-1]
            latest_eval = evaluations[-1]
            self.ax.annotate(f'{latest_eval:.2f}', xy=(latest_move, latest_eval),
                             xytext=(latest_move, latest_eval + 0.1),
                             arrowprops=dict(shrink=0.05),
                             ha='center', fontsize=10)
            self.ax.legend()
        else:
            self.ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=self.ax.transAxes,
                         fontsize=14, fontweight='bold')

        self.canvas.draw()

class PolicyOutputPlot:
    def __init__(self):
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

    def plot_policy_output(self, policy_output):
        self.ax.clear()
        if policy_output:
            sorted_policy = sorted(policy_output.items(), key=lambda x: x[1], reverse=True)[:5]
            if sorted_policy:
                moves, probabilities = zip(*sorted_policy)
                probabilities_percent = [prob * 100 for prob in probabilities]
                y_pos = np.arange(len(moves))
                bars = self.ax.barh(y_pos, probabilities_percent, color=plt.cm.Blues(np.array(probabilities_percent) / 100))
                for bar, prob in zip(bars, probabilities_percent):
                    width = bar.get_width()
                    self.ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{prob:.1f}%', va='center', fontsize=10)
                self.ax.set_yticks(y_pos)
                self.ax.set_yticklabels(moves)
                self.ax.invert_yaxis()
        else:
            self.ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=self.ax.transAxes,
                         fontsize=14, fontweight='bold')

        self.canvas.draw()

class MCTSStatisticsPlot:
    def __init__(self):
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

    def plot_mcts_statistics(self, mcts_stats):
        self.ax.clear()
        if mcts_stats and any(mcts_stats.values()):
            metrics = ['Simulations', 'Nodes Explored']
            values = [
                mcts_stats.get('simulations', 0),
                mcts_stats.get('nodes_explored', 0),
            ]
            y_pos = np.arange(len(metrics))
            max_val = max(values) if max(values) > 0 else 1
            bars = self.ax.bar(y_pos, values, color=plt.cm.Oranges(np.array(values) / max_val))
            for bar, val in zip(bars, values):
                self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val}',
                             ha='center', va='bottom', fontsize=10)
            best_move = mcts_stats.get('best_move')
            if best_move:
                self.ax.annotate(f'Best Move: {best_move}',
                                 xy=(0.5, 1.05), xycoords='axes fraction',
                                 ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            self.ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=self.ax.transAxes,
                         fontsize=14, fontweight='bold')

        self.canvas.draw()

class GameVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        tabs = QTabWidget()

        self.value_evaluation_plot = ValueEvaluationPlot()
        value_eval_widget = QWidget()
        value_eval_layout = QVBoxLayout()
        value_eval_layout.addWidget(self.value_evaluation_plot.canvas)
        value_eval_widget.setLayout(value_eval_layout)
        tabs.addTab(value_eval_widget, "Value Evaluation")

        self.policy_output_plot = PolicyOutputPlot()
        policy_output_widget = QWidget()
        policy_output_layout = QVBoxLayout()
        policy_output_layout.addWidget(self.policy_output_plot.canvas)
        policy_output_widget.setLayout(policy_output_layout)
        tabs.addTab(policy_output_widget, "Policy Output")

        self.mcts_statistics_plot = MCTSStatisticsPlot()
        mcts_stats_widget = QWidget()
        mcts_stats_layout = QVBoxLayout()
        mcts_stats_layout.addWidget(self.mcts_statistics_plot.canvas)
        mcts_stats_widget.setLayout(mcts_stats_layout)
        tabs.addTab(mcts_stats_widget, "MCTS Statistics")

        main_layout.addWidget(tabs)

    def update_value_evaluation(self, evaluations):
        self.value_evaluation_plot.plot_evaluations(evaluations)

    def update_policy_output(self, policy_output):
        self.policy_output_plot.plot_policy_output(policy_output)

    def update_mcts_statistics(self, mcts_stats):
        self.mcts_statistics_plot.plot_mcts_statistics(mcts_stats)