from src.base.base_visualization import BasePlot, BaseVisualizationWidget
import time

class ReinforcementVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialize_data_storage()
        self.last_update_time = time.time()

    def init_visualization(self):
        grid_spec = self.figure.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

        # Define subplots
        self.ax_game_outcomes = self.figure.add_subplot(grid_spec[0, 0])
        self.ax_game_length = self.figure.add_subplot(grid_spec[0, 1])
        self.ax_training_speed = self.figure.add_subplot(grid_spec[1, 0])
        self.ax_avg_mcts_visits = self.figure.add_subplot(grid_spec[1, 1])

        # Configure plots
        self.plots['game_outcomes'] = BasePlot(self.ax_game_outcomes, title='Game Outcomes', xlabel='Games Played', ylabel='Percentage')
        self.ax_game_outcomes.set_ylim(0, 100)
        self.plots['game_length'] = BasePlot(self.ax_game_length, title='Average Game Length', xlabel='Games Played', ylabel='Moves')
        self.plots['training_speed'] = BasePlot(self.ax_training_speed, title='Training Speed', xlabel='Games Played', ylabel='Games/Second')
        self.plots['avg_mcts_visits'] = BasePlot(self.ax_avg_mcts_visits, title='Average MCTS Visits', xlabel='Games Played', ylabel='Visits')

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

        total_games = stats.get('total_games', stats.get('total_games_played', 0))
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        draws = stats.get('draws', 0)
        avg_game_length = stats.get('avg_game_length', 0)
        avg_visits = stats.get('avg_mcts_visits', 0)

        # Update cumulative stats
        if self.games_played:
            total_games += self.games_played[-1]
            wins += self.wins[-1]
            losses += self.losses[-1]
            draws += self.draws[-1]

        self.games_played.append(total_games)
        self.wins.append(wins)
        self.losses.append(losses)
        self.draws.append(draws)

        # Compute rates
        self.win_rates.append(100 * wins / total_games if total_games else 0)
        self.loss_rates.append(100 * losses / total_games if total_games else 0)
        self.draw_rates.append(100 * draws / total_games if total_games else 0)

        # Game length and visits
        self.avg_game_lengths.append(avg_game_length)
        self.game_lengths_all.extend([avg_game_length] * total_games)
        self.avg_mcts_visits.append(avg_visits)

        # Update games per second
        elapsed_time = current_time - self.start_time
        games_per_second = self.games_played[-1] / elapsed_time if elapsed_time > 0 else 0
        self.games_per_second.append(games_per_second)

        self.update_visualization()
        self.last_update_time = current_time

    def update_visualization(self):
        self._update_plot('game_outcomes', self.win_rates, self.draw_rates, self.loss_rates, labels=['Win', 'Draw', 'Loss'], colors=['green', 'blue', 'red'])
        self._update_simple_plot('game_length', self.avg_game_lengths, color='purple')
        self._update_simple_plot('training_speed', self.games_per_second, color='orange')
        self._update_simple_plot('avg_mcts_visits', self.avg_mcts_visits, color='magenta')
        self.canvas.draw_idle()

    def _update_plot(self, key, *data, labels, colors):
        self.clear_axis(key)
        if self.games_played:
            for y_data, label, color in zip(data, labels, colors):
                self.plots[key].ax.plot(self.games_played, y_data, label=f'{label} Rate', color=color, alpha=0.8)
            self.plots[key].ax.legend(frameon=False, fontsize=10)
        else:
            self.add_text_to_axis(key, 'No Data')

    def _update_simple_plot(self, key, y_data, color):
        self.clear_axis(key)
        if self.games_played and y_data:
            self.plots[key].ax.plot(self.games_played, y_data, color=color, alpha=0.8)
            self.plots[key].ax.set_ylim(bottom=0)
        else:
            self.add_text_to_axis(key, 'No Data')