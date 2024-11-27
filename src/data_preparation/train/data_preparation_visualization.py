import numpy as np, time
from src.base.base_visualization import BasePlot, BaseVisualizationWidget


class DataPreparationVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reset_visualizations()

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        self.ax_game_results = self.figure.add_subplot(gs[0, 0])
        self.ax_games_processed = self.figure.add_subplot(gs[0, 1])
        self.ax_game_lengths = self.figure.add_subplot(gs[1, 0])
        self.ax_player_ratings = self.figure.add_subplot(gs[1, 1])

        self.plots['game_results'] = BasePlot(self.ax_game_results, title='Game Results Distribution')
        self.plots['games_processed'] = BasePlot(self.ax_games_processed, title='Games Processed Over Time', xlabel='Time (s)', ylabel='Total Games Processed')
        self.plots['game_lengths'] = BasePlot(self.ax_game_lengths, title='Game Length Distribution', xlabel='Number of Moves', ylabel='Frequency')
        self.plots['player_ratings'] = BasePlot(self.ax_player_ratings, title='Player Rating Distribution', xlabel='Player Rating', ylabel='Frequency')

    def update_data_visualization(self, stats):
        self.game_results = stats.get('game_results_counter', self.game_results)
        total_games = stats.get('total_games_processed', 0)
        if self.start_time is None:
            self.start_time = time.time()
            self.processing_times = [0]
        else:
            self.processing_times.append(time.time() - self.start_time)
        self.total_games_processed.append(total_games)
        self.game_length_bins = stats.get('game_length_bins', self.game_length_bins)
        self.game_length_histogram = stats.get('game_length_histogram', self.game_length_histogram)
        self.player_rating_bins = stats.get('player_rating_bins', self.player_rating_bins)
        self.player_rating_histogram = stats.get('player_rating_histogram', self.player_rating_histogram)

        self.update_game_results_plot()
        self.update_games_processed_plot()
        self.update_game_lengths_plot()
        self.update_player_ratings_plot()
        self.update_visualization()

    def update_game_results_plot(self):
        ax = self.plots['game_results'].ax
        ax.clear()
        self.plots['game_results'].apply_settings()
        results = [self.game_results.get(val, 0) for val in [1.0, -1.0, 0.0]]
        total = sum(results)
        if total > 0:
            percentages = [(r / total) * 100 for r in results]
            labels = ['White Wins', 'Black Wins', 'Draws']
            colors = ['#4CAF50', '#F44336', '#FFC107']
            explode = (0.05, 0.05, 0.05)
            ax.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=140,
                   colors=colors, explode=explode, shadow=True)
            ax.axis('equal')
        else:
            ax.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=12)

    def update_games_processed_plot(self):
        ax = self.plots['games_processed'].ax
        ax.clear()
        self.plots['games_processed'].apply_settings()
        if self.total_games_processed and self.processing_times:
            ax.plot(self.processing_times, self.total_games_processed,
                    marker='o', color='#2196F3')
            ax.relim()
            ax.autoscale_view()
        else:
            ax.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=12)

    def update_game_lengths_plot(self):
        ax = self.plots['game_lengths'].ax
        ax.clear()
        self.plots['game_lengths'].apply_settings()
        if self.game_length_histogram is not None and np.sum(self.game_length_histogram) > 0:
            ax.bar(self.game_length_bins[:-1], self.game_length_histogram,
                   width=np.diff(self.game_length_bins), align='edge',
                   color='#9C27B0', edgecolor='black', alpha=0.7)
            ax.relim()
            ax.autoscale_view()
        else:
            ax.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=12)

    def update_player_ratings_plot(self):
        ax = self.plots['player_ratings'].ax
        ax.clear()
        self.plots['player_ratings'].apply_settings()
        if self.player_rating_histogram is not None and np.sum(self.player_rating_histogram) > 0:
            ax.bar(self.player_rating_bins[:-1], self.player_rating_histogram,
                   width=np.diff(self.player_rating_bins), align='edge',
                   color='#FF5722', edgecolor='black', alpha=0.7)
            ax.relim()
            ax.autoscale_view()
        else:
            ax.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=12)

    def reset_visualizations(self):
        self.game_results = {1.0: 0, -1.0: 0, 0.0: 0}
        self.total_games_processed = []
        self.processing_times = []
        self.game_length_bins = np.arange(0, 200, 5)
        self.game_length_histogram = np.zeros(len(self.game_length_bins) - 1, dtype=int)
        self.player_rating_bins = np.arange(1000, 3000, 50)
        self.player_rating_histogram = np.zeros(len(self.player_rating_bins) - 1, dtype=int)
        self.start_time = None
        self.reset_visualization()