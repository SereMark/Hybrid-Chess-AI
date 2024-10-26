import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class BasePlot(QWidget):
    def __init__(self, title, xlabel, ylabel, invert_y=False, parent=None):
        super().__init__(parent)
        self.init_ui(title, xlabel, ylabel, invert_y)

    def init_ui(self, title, xlabel, ylabel, invert_y):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        self.ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
        self.ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        self.ax.tick_params(axis='both', which='major', labelsize=10)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        if invert_y:
            self.ax.invert_yaxis()


class ValueEvaluationPlot(BasePlot):
    def __init__(self, parent=None):
        super().__init__(title="Value Evaluation",
                         xlabel="Move Number",
                         ylabel="Evaluation Score",
                         parent=parent)

    def plot_evaluations(self, evaluations):
        self.ax.clear()
        self.ax.set_title("Value Evaluation", fontsize=14, fontweight='bold', pad=15)
        self.ax.set_xlabel("Move Number", fontsize=12, labelpad=10)
        self.ax.set_ylabel("Evaluation Score", fontsize=12, labelpad=10)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        if evaluations:
            moves = list(range(1, len(evaluations) + 1))
            self.ax.plot(moves, evaluations, marker='o', linewidth=2, color='#1f77b4', label='Evaluation')

            if len(evaluations) >= 5:
                moving_avg = np.convolve(evaluations, np.ones(5)/5, mode='valid')
                self.ax.plot(moves[4:], moving_avg, linestyle='--', color='#ff7f0e', label='5-Move Moving Avg')

            self.ax.annotate(f'{evaluations[-1]:.2f}',
                             xy=(moves[-1], evaluations[-1]),
                             xytext=(0, 10),
                             textcoords='offset points',
                             ha='center',
                             fontsize=10,
                             fontweight='bold',
                             arrowprops=dict(arrowstyle='->', color='gray'))

            self.ax.legend(fontsize=10, framealpha=0.9)
        else:
            self.ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start',
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=14, fontweight='bold', color='gray')

        self.canvas.draw()


class MaterialBalancePlot(BasePlot):
    def __init__(self, parent=None):
        super().__init__(title="Material Balance",
                         xlabel="Move Number",
                         ylabel="Material Difference (White - Black)",
                         parent=parent)

    def plot_material_balance(self, balances):
        self.ax.clear()
        self.ax.set_title("Material Balance", fontsize=14, fontweight='bold', pad=15)
        self.ax.set_xlabel("Move Number", fontsize=12, labelpad=10)
        self.ax.set_ylabel("Material Difference (White - Black)", fontsize=12, labelpad=10)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        if balances:
            moves = list(range(1, len(balances) + 1))
            self.ax.plot(moves, balances, marker='o', linewidth=2, color='#1f77b4', label='Material Balance')
            self.ax.legend(fontsize=10, framealpha=0.9)
        else:
            self.ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start',
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=14, fontweight='bold', color='gray')

        self.canvas.draw()


class GameVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.evaluations = []
        self.material_balances = []

    def init_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        self.value_evaluation_plot = ValueEvaluationPlot()
        self.material_balance_plot = MaterialBalancePlot()

        tabs.addTab(self.value_evaluation_plot, "Value Evaluation")
        tabs.addTab(self.material_balance_plot, "Material Balance")

        layout.addWidget(tabs)
        self.setLayout(layout)

    def update_value_evaluation(self, evaluations):
        self.evaluations.extend(evaluations)
        self.value_evaluation_plot.plot_evaluations(self.evaluations)

    def update_material_balance(self, balances):
        self.material_balances.extend(balances)
        self.material_balance_plot.plot_material_balance(self.material_balances)

    def reset_visualizations(self):
        self.evaluations = []
        self.material_balances = []
        self.value_evaluation_plot.plot_evaluations(self.evaluations)
        self.material_balance_plot.plot_material_balance(self.material_balances)