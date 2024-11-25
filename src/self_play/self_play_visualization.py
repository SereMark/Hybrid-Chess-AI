import time
from src.base.base_visualization import BasePlot, BaseVisualizationWidget


class SelfPlayVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialize_data_storage()
        self.last_update_time = time.time()
        self.update_interval = 0.5

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(4, 2)
        self.ax1 = self.figure.add_subplot(gs[0, :])
        self.ax2 = self.figure.add_subplot(gs[1, 0])
        self.ax3 = self.figure.add_subplot(gs[1, 1])
        self.ax4 = self.figure.add_subplot(gs[2, :])

        self.plots['game_outcomes'] = BasePlot(self.ax1, 'Game Outcomes', 'Games Played', 'Percentage', title_fontsize=10, label_fontsize=9, tick_labelsize=8)
        self.ax1.set_ylim(0, 100)
        self.plots['game_length'] = BasePlot(self.ax2, 'Game Length Distribution', 'Number of Moves', 'Frequency', title_fontsize=10, label_fontsize=9, tick_labelsize=8)
        self.plots['training_speed'] = BasePlot(self.ax3, 'Training Speed', 'Games Played', 'Games/Second', title_fontsize=10, label_fontsize=9, tick_labelsize=8)
        self.plots['training_metrics'] = BasePlot(self.ax4, 'Training Progress Metrics', 'Games Played', 'Value', title_fontsize=10, label_fontsize=9, tick_labelsize=8)
        self.ax4.set_ylim(0, 1.1)

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
        super().reset_visualization()

    def update_stats(self):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        self.update_visualization()
        self.last_update_time = current_time