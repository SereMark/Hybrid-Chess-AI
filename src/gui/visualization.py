import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from matplotlib.figure import Figure

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
        self.eval_figure.patch.set_facecolor('#121212')
        self.eval_canvas.setStyleSheet("background-color: #121212;")
        self.eval_layout = QVBoxLayout()
        self.eval_layout.addWidget(self.eval_canvas)
        self.eval_widget = QWidget()
        self.eval_widget.setLayout(self.eval_layout)
        self.move_evaluations = []

        self.policy_figure = Figure(figsize=(5, 4))
        self.policy_canvas = FigureCanvas(self.policy_figure)
        self.policy_figure.patch.set_facecolor('#121212')
        self.policy_canvas.setStyleSheet("background-color: #121212;")
        self.policy_layout = QVBoxLayout()
        self.policy_layout.addWidget(self.policy_canvas)
        self.policy_widget = QWidget()
        self.policy_widget.setLayout(self.policy_layout)

        self.mcts_figure = Figure(figsize=(5, 4))
        self.mcts_canvas = FigureCanvas(self.mcts_figure)
        self.mcts_figure.patch.set_facecolor('#121212')
        self.mcts_canvas.setStyleSheet("background-color: #121212;")
        self.mcts_layout = QVBoxLayout()
        self.mcts_layout.addWidget(self.mcts_canvas)
        self.mcts_widget = QWidget()
        self.mcts_widget.setLayout(self.mcts_layout)

        self.tabs.addTab(self.eval_widget, "Value Evaluation")
        self.tabs.addTab(self.policy_widget, "Policy Output")
        self.tabs.addTab(self.mcts_widget, "MCTS Statistics")

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.tabs)
        self.setLayout(self.main_layout)

    def visualize_evaluation(self, evaluations, policy_output, mcts_stats):
        self.eval_figure.clear()
        eval_ax = self.eval_figure.add_subplot(111)
        eval_ax.set_facecolor('#121212')
        eval_ax.tick_params(colors='#E0E0E0', which='both')
        for spine in eval_ax.spines.values():
            spine.set_color('#E0E0E0')
        eval_ax.plot(evaluations, color='#FFD700', marker='o', label='Evaluation')
        eval_ax.set_xlabel('Move', color='#E0E0E0')
        eval_ax.set_ylabel('Evaluation', color='#E0E0E0')
        eval_ax.set_title('AI Value Evaluation Over Time', color='#E0E0E0')
        eval_ax.grid(True, color='#333333', linestyle='dotted')
        eval_ax.legend(facecolor='#121212', edgecolor='#121212', labelcolor='#E0E0E0')
        self.eval_canvas.draw()

        self.policy_figure.clear()
        policy_ax = self.policy_figure.add_subplot(111)
        policy_ax.set_facecolor('#121212')
        policy_ax.tick_params(colors='#E0E0E0', which='both')
        for spine in policy_ax.spines.values():
            spine.set_color('#E0E0E0')
        sorted_policy = sorted(policy_output.items(), key=lambda x: x[1], reverse=True)[:5]
        moves, probabilities = zip(*sorted_policy)
        policy_ax.bar(moves, probabilities, color='#00BFFF')
        policy_ax.set_xlabel('Moves', color='#E0E0E0')
        policy_ax.set_ylabel('Probability', color='#E0E0E0')
        policy_ax.set_title('Top 5 Move Probabilities', color='#E0E0E0')
        policy_ax.tick_params(axis='x', rotation=45)
        self.policy_canvas.draw()

        self.mcts_figure.clear()
        mcts_ax = self.mcts_figure.add_subplot(111)
        mcts_ax.set_facecolor('#121212')
        mcts_ax.tick_params(colors='#E0E0E0', which='both')
        for spine in mcts_ax.spines.values():
            spine.set_color('#E0E0E0')
        stats = ['Simulations', 'Nodes Explored']
        values = [mcts_stats['simulations'], mcts_stats['nodes_explored']]
        mcts_ax.bar(stats, values, color='#32CD32')
        mcts_ax.set_xlabel('MCTS Metrics', color='#E0E0E0')
        mcts_ax.set_ylabel('Count', color='#E0E0E0')
        mcts_ax.set_title('MCTS Statistics', color='#E0E0E0')
        mcts_ax.text(0.5, max(values)*0.9, f"Best Move: {mcts_stats['best_move']}",
                     ha='center', color='#E0E0E0', fontsize=12)
        self.mcts_canvas.draw()

    def clear_visualization(self):
        self.move_evaluations.clear()
        self.eval_figure.clear()
        self.eval_canvas.draw()
        self.policy_figure.clear()
        self.policy_canvas.draw()
        self.mcts_figure.clear()
        self.mcts_canvas.draw()