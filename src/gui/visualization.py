import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from matplotlib.figure import Figure
import numpy as np

class ChessAIVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab {
                height: 30px;
                width: 150px;
                color: #E0E0E0;
                background-color: #1E1E1E;
                border: 1px solid #333333;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #333333;
                color: #FFD700;
            }
        """)

        self.eval_figure = Figure(figsize=(5, 4))
        self.eval_canvas = FigureCanvas(self.eval_figure)
        self.setup_figure(self.eval_figure, self.eval_canvas)
        self.eval_ax = self.eval_figure.add_subplot(111)
        self.eval_widget = self.create_tab_widget(self.eval_canvas)
        self.move_evaluations = []
        self.init_value_evaluation()

        self.policy_figure = Figure(figsize=(5, 4))
        self.policy_canvas = FigureCanvas(self.policy_figure)
        self.setup_figure(self.policy_figure, self.policy_canvas)
        self.policy_ax = self.policy_figure.add_subplot(111)
        self.policy_widget = self.create_tab_widget(self.policy_canvas)
        self.init_policy_output()

        self.mcts_figure = Figure(figsize=(5, 4))
        self.mcts_canvas = FigureCanvas(self.mcts_figure)
        self.setup_figure(self.mcts_figure, self.mcts_canvas)
        self.mcts_ax = self.mcts_figure.add_subplot(111)
        self.mcts_widget = self.create_tab_widget(self.mcts_canvas)
        self.init_mcts_statistics()

        self.tabs.addTab(self.eval_widget, "Value Evaluation")
        self.tabs.addTab(self.policy_widget, "Policy Output")
        self.tabs.addTab(self.mcts_widget, "MCTS Statistics")

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.tabs)
        self.setLayout(self.main_layout)

    def setup_figure(self, figure, canvas):
        figure.patch.set_facecolor('#121212')
        canvas.setStyleSheet("background-color: #121212;")

    def create_tab_widget(self, canvas):
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def init_value_evaluation(self):
        self.eval_ax.clear()
        self.eval_ax.set_facecolor('#121212')
        self.eval_ax.tick_params(colors='#E0E0E0', which='both')

        for spine in self.eval_ax.spines.values():
            spine.set_color('#E0E0E0')

        self.eval_ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start', 
                          horizontalalignment='center', 
                          verticalalignment='center', 
                          transform=self.eval_ax.transAxes,
                          color='#E0E0E0',
                          fontsize=14, fontweight='bold')
        self.eval_ax.set_xlabel('Move', color='#E0E0E0')
        self.eval_ax.set_ylabel('Evaluation', color='#E0E0E0')
        self.eval_ax.set_title('AI Value Evaluation Over Time', color='#E0E0E0')
        self.eval_ax.grid(True, color='#333333', linestyle='dotted')
        self.eval_canvas.draw()

    def init_policy_output(self):
        self.policy_ax.clear()
        self.policy_ax.set_facecolor('#121212')
        self.policy_ax.tick_params(colors='#E0E0E0', which='both')

        for spine in self.policy_ax.spines.values():
            spine.set_color('#E0E0E0')

        self.policy_ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start', 
                            horizontalalignment='center', 
                            verticalalignment='center', 
                            transform=self.policy_ax.transAxes,
                            color='#E0E0E0',
                            fontsize=14, fontweight='bold')
        self.policy_ax.set_xlabel('Moves', color='#E0E0E0')
        self.policy_ax.set_ylabel('Probability (%)', color='#E0E0E0')
        self.policy_ax.set_title('Top 5 Move Probabilities', color='#E0E0E0')
        self.policy_ax.tick_params(axis='x', rotation=45)
        self.policy_canvas.draw()

    def init_mcts_statistics(self):
        self.mcts_ax.clear()
        self.mcts_ax.set_facecolor('#121212')
        self.mcts_ax.tick_params(colors='#E0E0E0', which='both')

        for spine in self.mcts_ax.spines.values():
            spine.set_color('#E0E0E0')

        self.mcts_ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start', 
                          horizontalalignment='center', 
                          verticalalignment='center', 
                          transform=self.mcts_ax.transAxes,
                          color='#E0E0E0',
                          fontsize=14, fontweight='bold')
        self.mcts_ax.set_xlabel('MCTS Metrics', color='#E0E0E0')
        self.mcts_ax.set_ylabel('Count', color='#E0E0E0')
        self.mcts_ax.set_title('MCTS Statistics', color='#E0E0E0')
        self.mcts_canvas.draw()

    def visualize_evaluation(self, evaluations, policy_output, mcts_stats):
        self.update_value_evaluation(evaluations)
        self.update_policy_output(policy_output)
        self.update_mcts_statistics(mcts_stats)

    def update_value_evaluation(self, evaluations):
        self.eval_ax.clear()
        self.eval_ax.set_facecolor('#121212')
        self.eval_ax.tick_params(colors='#E0E0E0', which='both')

        for spine in self.eval_ax.spines.values():
            spine.set_color('#E0E0E0')

        if evaluations:
            moves = list(range(1, len(evaluations) + 1))
            self.eval_ax.plot(moves, evaluations, color='#FFD700', marker='o', label='Evaluation', linewidth=2, markersize=6)

            window_size = min(5, len(evaluations))
            moving_avg = np.convolve(evaluations, np.ones(window_size)/window_size, mode='valid')
            self.eval_ax.plot(moves[window_size-1:], moving_avg, color='#FF8C00', linestyle='--', label='Moving Average')

            latest_move = moves[-1]
            latest_eval = evaluations[-1]
            self.eval_ax.annotate(f'{latest_eval:.2f}', xy=(latest_move, latest_eval),
                                  xytext=(latest_move, latest_eval + 0.1),
                                  arrowprops=dict(facecolor='#FFD700', shrink=0.05),
                                  ha='center', color='#FFD700', fontsize=10)

            self.eval_ax.legend(facecolor='#121212', edgecolor='#121212', labelcolor='#E0E0E0')
        else:
            self.eval_ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start', 
                              horizontalalignment='center', 
                              verticalalignment='center', 
                              transform=self.eval_ax.transAxes,
                              color='#E0E0E0',
                              fontsize=14, fontweight='bold')

        self.eval_ax.set_xlabel('Move', color='#E0E0E0')
        self.eval_ax.set_ylabel('Evaluation', color='#E0E0E0')
        self.eval_ax.set_title('AI Value Evaluation Over Time', color='#E0E0E0')
        self.eval_ax.grid(True, color='#333333', linestyle='dotted')
        if evaluations:
            self.eval_ax.set_ylim(min(evaluations) - 1, max(evaluations) + 1)
        else:
            self.eval_ax.set_ylim(-1, 1)
        self.eval_canvas.draw()

    def update_policy_output(self, policy_output):
        self.policy_ax.clear()
        self.policy_ax.set_facecolor('#121212')
        self.policy_ax.tick_params(colors='#E0E0E0', which='both')

        for spine in self.policy_ax.spines.values():
            spine.set_color('#E0E0E0')

        if policy_output:
            sorted_policy = sorted(policy_output.items(), key=lambda x: x[1], reverse=True)[:5]
            if sorted_policy:
                moves, probabilities = zip(*sorted_policy)
                probabilities_percent = [prob * 100 for prob in probabilities]

                y_pos = np.arange(len(moves))
                bars = self.policy_ax.barh(y_pos, probabilities_percent, color=plt.cm.Blues(np.array(probabilities_percent) / 100))

                for bar, prob in zip(bars, probabilities_percent):
                    width = bar.get_width()
                    self.policy_ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{prob:.1f}%', va='center', color='#E0E0E0', fontsize=10)

                self.policy_ax.set_yticks(y_pos)
                self.policy_ax.set_yticklabels(moves)
                self.policy_ax.invert_yaxis()
        else:
            self.policy_ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start', 
                                horizontalalignment='center', 
                                verticalalignment='center', 
                                transform=self.policy_ax.transAxes,
                                color='#E0E0E0',
                                fontsize=14, fontweight='bold')

        self.policy_ax.set_xlabel('Probability (%)', color='#E0E0E0')
        self.policy_ax.set_ylabel('Moves', color='#E0E0E0')
        self.policy_ax.set_title('Top 5 Move Probabilities', color='#E0E0E0')
        self.policy_ax.tick_params(axis='x', rotation=0)
        self.policy_ax.set_xlim(0, 100)
        self.policy_canvas.draw()

    def update_mcts_statistics(self, mcts_stats):
        self.mcts_ax.clear()
        self.mcts_ax.set_facecolor('#121212')
        self.mcts_ax.tick_params(colors='#E0E0E0', which='both')

        for spine in self.mcts_ax.spines.values():
            spine.set_color('#E0E0E0')

        if mcts_stats and any(mcts_stats.values()):
            metrics = ['Simulations', 'Nodes Explored']
            values = [
                mcts_stats.get('simulations', 0),
                mcts_stats.get('nodes_explored', 0),
            ]

            y_pos = np.arange(len(metrics))
            bars = self.mcts_ax.bar(y_pos, values, color=plt.cm.Oranges(np.array(values) / max(values) if max(values) > 0 else 1))

            for bar, val in zip(bars, values):
                self.mcts_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val}',
                                  ha='center', va='bottom', color='#E0E0E0', fontsize=10)

            self.mcts_ax.set_xticks(y_pos)
            self.mcts_ax.set_xticklabels(metrics)
            self.mcts_ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)

            best_move = mcts_stats.get('best_move')
            if best_move:
                self.mcts_ax.annotate(f'Best Move: {best_move}', 
                                      xy=(0.5, 1.05), xycoords='axes fraction',
                                      ha='center', va='center', color='#FFD700', fontsize=12, fontweight='bold')
        else:
            self.mcts_ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start', 
                              horizontalalignment='center', 
                              verticalalignment='center', 
                              transform=self.mcts_ax.transAxes,
                              color='#E0E0E0',
                              fontsize=14, fontweight='bold')

        self.mcts_ax.set_xlabel('MCTS Metrics', color='#E0E0E0')
        self.mcts_ax.set_ylabel('Count', color='#E0E0E0')
        self.mcts_ax.set_title('MCTS Statistics', color='#E0E0E0')
        self.mcts_canvas.draw()

    def clear_visualization(self):
        self.move_evaluations.clear()
        self.init_value_evaluation()
        self.init_policy_output()
        self.init_mcts_statistics()