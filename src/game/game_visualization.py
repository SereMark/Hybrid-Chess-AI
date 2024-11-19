import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QTabWidget, QWidget
from src.base.base_visualization import BasePlotWidget


class ValueEvaluationPlot(BasePlotWidget):
    def __init__(self, parent=None):
        super().__init__(title="Value Evaluation", xlabel="Move Number", ylabel="Evaluation Score", parent=parent)
        self.ax.legend(fontsize=10, framealpha=0.9)

    def plot_evaluations(self, evaluations):
        self.reset_axis()
        if evaluations:
            moves = np.arange(1, len(evaluations) + 1)
            self.ax.plot(moves, evaluations, marker='o', linewidth=2, color='#1f77b4', label='Evaluation')

            if len(evaluations) >= 5:
                moving_avg = np.convolve(evaluations, np.ones(5)/5, mode='valid')
                self.ax.plot(moves[4:], moving_avg, linestyle='--', color='#ff7f0e', label='5-Move Moving Avg')

            self.ax.legend(fontsize=10, framealpha=0.9)
            self.ax.relim()
            self.ax.autoscale_view()
        else:
            self.ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=14, fontweight='bold', color='gray')
        self.update_plot()


class MaterialBalancePlot(BasePlotWidget):
    def __init__(self, parent=None):
        super().__init__(title="Material Balance", xlabel="Move Number",
                         ylabel="Material Difference (White - Black)", parent=parent)
        self.ax.legend(fontsize=10, framealpha=0.9)

    def plot_material_balance(self, balances):
        self.reset_axis()
        if balances:
            moves = np.arange(1, len(balances) + 1)
            self.ax.plot(moves, balances, marker='o', linewidth=2, color='#1f77b4', label='Material Balance')
            self.ax.legend(fontsize=10, framealpha=0.9)
            self.ax.relim()
            self.ax.autoscale_view()
        else:
            self.ax.text(0.5, 0.5, 'No Data Yet\nMake a move to start', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=14, fontweight='bold', color='gray')
        self.update_plot()


class GameVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.evaluations = []
        self.material_balances = []
        self.init_ui()

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