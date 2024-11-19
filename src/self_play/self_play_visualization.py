import time, networkx as nx
from src.base.base_visualization import BasePlot, BaseVisualizationWidget


class SelfPlayVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialize_data_storage()
        self.last_update_time = time.time()
        self.update_interval = 0.5
        self.mcts_graph = nx.DiGraph()

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(4, 2)
        self.ax1 = self.figure.add_subplot(gs[0, :])
        self.ax2 = self.figure.add_subplot(gs[1, 0])
        self.ax3 = self.figure.add_subplot(gs[1, 1])
        self.ax4 = self.figure.add_subplot(gs[2, :])
        self.ax_tree = self.figure.add_subplot(gs[3, :])

        self.plots['game_outcomes'] = BasePlot(self.ax1, 'Game Outcomes', 'Games Played', 'Percentage', title_fontsize=10, label_fontsize=9, tick_labelsize=8)
        self.ax1.set_ylim(0, 100)
        self.plots['game_length'] = BasePlot(self.ax2, 'Game Length Distribution', 'Number of Moves', 'Frequency', title_fontsize=10, label_fontsize=9, tick_labelsize=8)
        self.plots['training_speed'] = BasePlot(self.ax3, 'Training Speed', 'Games Played', 'Games/Second', title_fontsize=10, label_fontsize=9, tick_labelsize=8)
        self.plots['training_metrics'] = BasePlot(self.ax4, 'Training Progress Metrics', 'Games Played', 'Value', title_fontsize=10, label_fontsize=9, tick_labelsize=8)
        self.ax4.set_ylim(0, 1.1)
        self.plots['mcts_tree'] = BasePlot(self.ax_tree, 'MCTS Tree Visualization', title_fontsize=10)

    def initialize_data_storage(self):
        self.games_played = []
        self.wins = []
        self.losses = []
        self.draws = []
        self.win_rates = []
        self.draw_rates = []
        self.loss_rates = []
        self.avg_game_lengths = []
        self.game_lengths_all = []
        self.games_per_second = []
        self.min_lengths = []
        self.max_lengths = []
        self.start_time = time.time()

    def reset_visualization(self):
        self.initialize_data_storage()
        self.last_update_time = time.time()
        self.mcts_graph.clear()
        super().reset_visualization()

    def update_stats(self, stats):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        if 'tree_nodes' in stats and 'tree_edges' in stats:
            self.update_mcts_tree(stats['tree_nodes'], stats['tree_edges'])
        self.update_visualization()
        self.last_update_time = current_time

    @staticmethod
    def sanitize_string(s):
        return s.replace(':', '-')

    def update_mcts_tree(self, nodes, edges):
        self.mcts_graph.clear()

        sanitized_ids = {}
        for node_id, node_data in nodes:
            node_id_str = str(node_id)

            if ':' in node_id_str:
                sanitized_id = node_id_str.replace(':', '_')
            else:
                sanitized_id = node_id_str

            sanitized_ids[node_id_str] = sanitized_id

            move = self.sanitize_string(node_data['move'])

            label = f"{move}\nQ - {node_data['Q']:.2f}\nN - {node_data['n_visits']}"

            self.mcts_graph.add_node(sanitized_id, label=label)

        sanitized_edges = []
        for source, target in edges:
            source_str = str(source)
            target_str = str(target)

            sanitized_source = sanitized_ids.get(source_str, source_str.replace(':', '_'))
            sanitized_target = sanitized_ids.get(target_str, target_str.replace(':', '_'))

            sanitized_edges.append((sanitized_source, sanitized_target))

        self.mcts_graph.add_edges_from(sanitized_edges)

        ax = self.plots['mcts_tree'].ax
        ax.clear()
        self.plots['mcts_tree'].apply_settings()
        try:
            pos = nx.nx_pydot.graphviz_layout(self.mcts_graph, prog='dot')
        except Exception as e:
            print(f"Graphviz layout failed with error: {e}. Falling back to spring layout.")
            pos = nx.spring_layout(self.mcts_graph)

        nx.draw(
            self.mcts_graph, pos, ax=ax,
            labels=nx.get_node_attributes(self.mcts_graph, 'label'),
            node_size=500, node_color='lightblue', font_size=8,
            arrowsize=10, arrowstyle='->',
            linewidths=0.5, edgecolors='black'
        )
        ax.set_axis_off()