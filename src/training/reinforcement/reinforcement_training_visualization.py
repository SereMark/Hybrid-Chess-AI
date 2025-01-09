from src.base.base_visualization import BasePlot, BaseVisualizationWidget
import time

class ReinforcementVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialize_data_storage()
        self.last_update_time = time.time()

    def init_visualization(self):
        gs = self.figure.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        self.ax_game_outcomes = self.figure.add_subplot(gs[0, 0])
        self.ax_game_length = self.figure.add_subplot(gs[0, 1])
        self.ax_training_speed = self.figure.add_subplot(gs[1, 0])
        self.ax_avg_mcts_visits = self.figure.add_subplot(gs[1, 1])
        self.plots['game_outcomes'] = BasePlot(
            self.ax_game_outcomes,
            title='Game Outcomes',
            xlabel='Games Played',
            ylabel='Percentage'
        )
        self.ax_game_outcomes.set_ylim(0, 100)
        self.plots['game_length'] = BasePlot(
            self.ax_game_length,
            title='Average Game Length',
            xlabel='Games Played',
            ylabel='Moves'
        )
        self.plots['training_speed'] = BasePlot(
            self.ax_training_speed,
            title='Training Speed',
            xlabel='Games Played',
            ylabel='Games/Second'
        )
        self.plots['avg_mcts_visits'] = BasePlot(
            self.ax_avg_mcts_visits,
            title='Average MCTS Visits',
            xlabel='Games Played',
            ylabel='Visits'
        )
        self.ax4 = None
        self.ax5 = None

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
        if self.games_played:
            total_games_so_far = self.games_played[-1] + total_games
            total_wins = self.wins[-1] + wins
            total_losses = self.losses[-1] + losses
            total_draws = self.draws[-1] + draws
            total_avg_mcts_visits = (self.avg_mcts_visits[-1] + avg_visits) / 2
        else:
            total_games_so_far = total_games
            total_wins = wins
            total_losses = losses
            total_draws = draws
            total_avg_mcts_visits = avg_visits
        self.games_played.append(total_games_so_far)
        self.wins.append(total_wins)
        self.losses.append(total_losses)
        self.draws.append(total_draws)
        self.win_rates.append(100 * total_wins / total_games_so_far if total_games_so_far else 0)
        self.loss_rates.append(100 * total_losses / total_games_so_far if total_games_so_far else 0)
        self.draw_rates.append(100 * total_draws / total_games_so_far if total_games_so_far else 0)
        self.avg_game_lengths.append(avg_game_length)
        self.game_lengths_all.extend([avg_game_length] * total_games)
        elapsed_time = current_time - self.start_time
        games_per_second = total_games_so_far / elapsed_time if elapsed_time > 0 else 0
        self.games_per_second.append(games_per_second)
        self.avg_mcts_visits.append(total_avg_mcts_visits)
        self.update_visualization()
        self.last_update_time = current_time

    def update_visualization(self):
        self.clear_axis('game_outcomes')
        if self.games_played:
            self.ax_game_outcomes.plot(self.games_played, self.win_rates, label='Win Rate', color='green', alpha=0.8)
            self.ax_game_outcomes.plot(self.games_played, self.draw_rates, label='Draw Rate', color='blue', alpha=0.8)
            self.ax_game_outcomes.plot(self.games_played, self.loss_rates, label='Loss Rate', color='red', alpha=0.8)
            self.ax_game_outcomes.legend(frameon=False, fontsize=10)
            self.ax_game_outcomes.set_ylim(0, 100)
        else:
            self.add_text_to_axis('game_outcomes', 'No Data')
        self.clear_axis('game_length')
        if self.games_played:
            self.ax_game_length.plot(self.games_played, self.avg_game_lengths, color='purple', alpha=0.8)
        else:
            self.add_text_to_axis('game_length', 'No Data')
        self.clear_axis('training_speed')
        if self.games_played and self.games_per_second:
            self.ax_training_speed.plot(self.games_played, self.games_per_second, color='orange', alpha=0.8)
            self.ax_training_speed.set_ylim(bottom=0)
        else:
            self.add_text_to_axis('training_speed', 'No Data')
        self.clear_axis('avg_mcts_visits')
        if self.games_played and self.avg_mcts_visits:
            self.ax_avg_mcts_visits.plot(self.games_played, self.avg_mcts_visits, color='magenta', alpha=0.8)
            self.ax_avg_mcts_visits.set_ylim(bottom=0)
        else:
            self.add_text_to_axis('avg_mcts_visits', 'No Data')
        self.canvas.draw_idle()