import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QTabWidget, QWidget
from src.base.base_visualization import BasePlotWidget
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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


class MCTSTreePlot(BasePlotWidget):
    def __init__(self, parent=None):
        super().__init__(title="MCTS Tree", xlabel="", ylabel="", parent=parent)
        self.init_tree_plot()

    def init_tree_plot(self):
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_tree(self, nodes, edges):
        self.ax.clear()
        if not nodes:
            self.ax.text(0.5, 0.5, 'No MCTS Data', horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=12, color='gray')
            self.canvas.draw_idle()
            return
        G = nx.DiGraph()
        for node_id, attrs in nodes:
            label = attrs.get('move', 'root') if attrs.get('move') else 'root'
            G.add_node(node_id, label=label, n_visits=attrs.get('n_visits', 0), Q=attrs.get('Q', 0))

        for parent, child in edges:
            G.add_edge(parent, child)

        pos = nx.spring_layout(G, k=1, iterations=20)
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        visit_counts = [G.nodes[node]['n_visits'] for node in G.nodes()]
        max_visits = max(visit_counts) if visit_counts else 1
        node_sizes = [300 + (count / max_visits) * 700 for count in visit_counts]
        node_colors = [G.nodes[node]['Q'] for node in G.nodes()]
        cmap = plt.get_cmap('coolwarm')

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=cmap, ax=self.ax)
        nx.draw_networkx_edges(G, pos, arrows=True, ax=self.ax, alpha=0.5)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=self.ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(node_colors, default=0), vmax=max(node_colors, default=0)))
        sm.set_array([])
        cbar = self.figure.colorbar(sm, ax=self.ax, fraction=0.03, pad=0.04)
        cbar.set_label('Q-value')

        self.canvas.draw_idle()


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
        self.mcts_tree_plot = MCTSTreePlot()

        tabs.addTab(self.value_evaluation_plot, "Value Evaluation")
        tabs.addTab(self.material_balance_plot, "Material Balance")
        tabs.addTab(self.mcts_tree_plot, "MCTS Tree")

        layout.addWidget(tabs)
        self.setLayout(layout)

    def update_value_evaluation(self, evaluations):
        self.evaluations.extend(evaluations)
        self.value_evaluation_plot.plot_evaluations(self.evaluations)

    def update_material_balance(self, balances):
        self.material_balances.extend(balances)
        self.material_balance_plot.plot_material_balance(self.material_balances)

    def update_mcts_tree(self, nodes, edges):
        self.mcts_tree_plot.plot_tree(nodes, edges)

    def reset_visualizations(self):
        self.evaluations = []
        self.material_balances = []
        self.value_evaluation_plot.plot_evaluations(self.evaluations)
        self.material_balance_plot.plot_material_balance(self.material_balances)
        self.mcts_tree_plot.plot_tree([], [])