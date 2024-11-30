import time
from src.base.base_visualization import BasePlot, BaseVisualizationWidget


class ReinforcementVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialize_data_storage()
        self.last_update_time = time.time()
        self.update_interval = 0.5

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(5, 2)

        self.ax1 = self.figure.add_subplot(gs[0, :])
        self.plots['game_outcomes'] = BasePlot(
            self.ax1,
            title='Game Outcomes',
            xlabel='Games Played',
            ylabel='Percentage',
            title_fontsize=10,
            label_fontsize=9,
            tick_labelsize=8
        )
        self.ax1.set_ylim(0, 100)

        self.ax2 = self.figure.add_subplot(gs[1, 0])
        self.plots['game_length'] = BasePlot(
            self.ax2,
            title='Game Length Distribution',
            xlabel='Number of Moves',
            ylabel='Frequency',
            title_fontsize=10,
            label_fontsize=9,
            tick_labelsize=8
        )

        self.ax3 = self.figure.add_subplot(gs[1, 1])
        self.plots['training_speed'] = BasePlot(
            self.ax3,
            title='Training Speed',
            xlabel='Games Played',
            ylabel='Games/Second',
            title_fontsize=10,
            label_fontsize=9,
            tick_labelsize=8
        )

        self.ax4 = self.figure.add_subplot(gs[2, :])
        self.plots['training_metrics'] = BasePlot(
            self.ax4,
            title='Training Progress Metrics',
            xlabel='Games Played',
            ylabel='Percentage',
            title_fontsize=10,
            label_fontsize=9,
            tick_labelsize=8
        )
        self.ax4.set_ylim(0, 100)

        self.ax5 = self.figure.add_subplot(gs[3, :])
        self.plots['avg_mcts_visits'] = BasePlot(
            self.ax5,
            title='Average MCTS Visits per Game',
            xlabel='Iterations',
            ylabel='Average Visits',
            title_fontsize=10,
            label_fontsize=9,
            tick_labelsize=8
        )

        self.canvas.draw()

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
        self.avg_mcts_visits = []
        self.start_time = time.time()

    def reset_visualization(self):
        self.initialize_data_storage()
        self.last_update_time = time.time()
        super().reset_visualization()

    def update_stats(self, stats):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        total_games = stats['total_games']
        wins = stats['wins']
        losses = stats['losses']
        draws = stats['draws']
        avg_game_length = stats['avg_game_length']
        avg_mcts_visits = stats.get('avg_mcts_visits', 0)

        if self.games_played:
            total_games_so_far = self.games_played[-1] + total_games
            total_wins = self.wins[-1] + wins
            total_losses = self.losses[-1] + losses
            total_draws = self.draws[-1] + draws
            total_avg_mcts_visits = (self.avg_mcts_visits[-1] + avg_mcts_visits) / 2
        else:
            total_games_so_far = total_games
            total_wins = wins
            total_losses = losses
            total_draws = draws
            total_avg_mcts_visits = avg_mcts_visits

        self.games_played.append(total_games_so_far)
        self.wins.append(total_wins)
        self.losses.append(total_losses)
        self.draws.append(total_draws)

        self.win_rates.append(100 * total_wins / total_games_so_far)
        self.loss_rates.append(100 * total_losses / total_games_so_far)
        self.draw_rates.append(100 * total_draws / total_games_so_far)

        self.avg_game_lengths.append(avg_game_length)
        self.game_lengths_all.extend([avg_game_length] * total_games)

        elapsed_time = current_time - self.start_time
        games_per_second = total_games_so_far / elapsed_time if elapsed_time > 0 else 0
        self.games_per_second.append(games_per_second)

        self.avg_mcts_visits.append(total_avg_mcts_visits)

        self.update_visualization()
        self.last_update_time = current_time

    def update_visualization(self):
        self.ax1.clear()
        self.plots['game_outcomes'] = BasePlot(
            self.ax1,
            title='Game Outcomes',
            xlabel='Games Played',
            ylabel='Percentage',
            title_fontsize=10,
            label_fontsize=9,
            tick_labelsize=8
        )
        self.ax1.plot(self.games_played, self.win_rates, label='Win Rate', color='green')
        self.ax1.plot(self.games_played, self.draw_rates, label='Draw Rate', color='blue')
        self.ax1.plot(self.games_played, self.loss_rates, label='Loss Rate', color='red')
        self.ax1.legend()
        self.ax1.set_ylim(0, 100)

        self.ax2.clear()
        self.plots['game_length'] = BasePlot(
            self.ax2,
            title='Game Length Distribution',
            xlabel='Number of Moves',
            ylabel='Frequency',
            title_fontsize=10,
            label_fontsize=9,
            tick_labelsize=8
        )
        if self.game_lengths_all:
            self.ax2.hist(self.game_lengths_all, bins=20, color='purple', alpha=0.7)
        else:
            self.ax2.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')

        self.ax3.clear()
        self.plots['training_speed'] = BasePlot(
            self.ax3,
            title='Training Speed',
            xlabel='Games Played',
            ylabel='Games/Second',
            title_fontsize=10,
            label_fontsize=9,
            tick_labelsize=8
        )
        self.ax3.plot(self.games_played, self.games_per_second, color='orange')
        self.ax3.set_ylim(bottom=0)

        self.ax4.clear()
        self.plots['training_metrics'] = BasePlot(
            self.ax4,
            title='Training Progress Metrics',
            xlabel='Games Played',
            ylabel='Percentage',
            title_fontsize=10,
            label_fontsize=9,
            tick_labelsize=8
        )
        self.ax4.plot(self.games_played, self.win_rates, label='Win Rate', color='green')
        self.ax4.plot(self.games_played, self.draw_rates, label='Draw Rate', color='blue')
        self.ax4.plot(self.games_played, self.loss_rates, label='Loss Rate', color='red')
        self.ax4.legend()
        self.ax4.set_ylim(0, 100)

        self.ax5.clear()
        self.plots['avg_mcts_visits'] = BasePlot(
            self.ax5,
            title='Average MCTS Visits per Game',
            xlabel='Iterations',
            ylabel='Average Visits',
            title_fontsize=10,
            label_fontsize=9,
            tick_labelsize=8
        )
        self.ax5.plot(self.games_played, self.avg_mcts_visits, label='Avg MCTS Visits', color='magenta')
        self.ax5.legend()
        self.ax5.set_ylim(bottom=0)

        self.canvas.draw_idle()