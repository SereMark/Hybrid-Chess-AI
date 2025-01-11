from src.base.base_visualization import BasePlot, BaseVisualizationWidget
import numpy as np
import time

class DataPreparationVisualization(BaseVisualizationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reset_visualization()

    def init_visualization(self):
        grid_spec = self.figure.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

        # Define subplots
        self.ax_game_results = self.figure.add_subplot(grid_spec[0, 0])
        self.ax_games_processed = self.figure.add_subplot(grid_spec[0, 1])
        self.ax_game_lengths = self.figure.add_subplot(grid_spec[1, 0])
        self.ax_player_ratings = self.figure.add_subplot(grid_spec[1, 1])

        # Define plots
        self.plots = {
            'game_results': BasePlot(self.ax_game_results, title='Game Results Distribution'),
            'games_processed': BasePlot(self.ax_games_processed, title='Games Processed Over Time', xlabel='Time (s)', ylabel='Total Games Processed'),
            'game_lengths': BasePlot(self.ax_game_lengths, title='Game Length Distribution', xlabel='Number of Moves', ylabel='Frequency'),
            'player_ratings': BasePlot(self.ax_player_ratings, title='Player Rating Distribution', xlabel='Player Rating', ylabel='Frequency'),
        }

    def update_data_visualization(self, stats):
        self.game_results = stats.get('game_results_counter', self.game_results)
        total_games = stats.get('total_games_processed', 0)

        # Update processing time
        if self.start_time is None:
            self.start_time = time.time()
            self.processing_times = [0]
        else:
            self.processing_times.append(time.time() - self.start_time)

        self.total_games_processed.append(total_games)
        self.game_length_bins = np.array(stats.get('game_length_bins', self.game_length_bins))
        self.game_length_histogram = np.array(stats.get('game_length_histogram', self.game_length_histogram))
        self.player_rating_bins = np.array(stats.get('player_rating_bins', self.player_rating_bins))
        self.player_rating_histogram = np.array(stats.get('player_rating_histogram', self.player_rating_histogram))

        # Update game results plot
        self.clear_axis('game_results')
        results = [self.game_results.get(val, 0) for val in [1.0, -1.0, 0.0]]
        total = sum(results)
        if total > 0:
            percentages = [(count / total) * 100 for count in results]
            labels = ['White Wins', 'Black Wins', 'Draws']
            colors = ['#4CAF50', '#F44336', '#FFC107']
            explode = (0.05, 0.05, 0.05)

            self.ax_game_results.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode, shadow=False, textprops={'color': '#333333', 'fontsize': 10})
            self.ax_game_results.axis('equal')
        else:
            self.add_text_to_axis('game_results', 'No Data Yet')

        # Update games processed plot
        self.clear_axis('games_processed')
        if self.total_games_processed and self.processing_times:
            self.ax_games_processed.plot(self.processing_times, self.total_games_processed, marker='o', color='#2196F3', alpha=0.8, linewidth=1.5)
            self.ax_games_processed.relim()
            self.ax_games_processed.autoscale_view()
        else:
            self.add_text_to_axis('games_processed', 'No Data Yet')

        # Update game lengths plot
        self.clear_axis('game_lengths')
        if self.game_length_histogram is not None and np.any(self.game_length_histogram > 0):
            self.ax_game_lengths.bar(self.game_length_bins[:-1], self.game_length_histogram, width=np.diff(self.game_length_bins), align='edge', color='#9C27B0', edgecolor='black', alpha=0.7)
            self.ax_game_lengths.relim()
            self.ax_game_lengths.autoscale_view()
        else:
            self.add_text_to_axis('game_lengths', 'No Data Yet')

        # Update player ratings plot
        self.clear_axis('player_ratings')
        if self.player_rating_histogram is not None and np.any(self.player_rating_histogram > 0):
            self.ax_player_ratings.bar(self.player_rating_bins[:-1], self.player_rating_histogram, width=np.diff(self.player_rating_bins), align='edge', color='#FF5722', edgecolor='black', alpha=0.7)
            self.ax_player_ratings.relim()
            self.ax_player_ratings.autoscale_view()
        else:
            self.add_text_to_axis('player_ratings', 'No Data Yet')

        # Refresh visualization
        self.update_visualization()

    def reset_visualization(self):
        self.game_results = {1.0: 0, -1.0: 0, 0.0: 0}
        self.total_games_processed = []
        self.processing_times = []
        self.game_length_bins = np.arange(0, 200, 5)
        self.game_length_histogram = np.zeros(len(self.game_length_bins) - 1, dtype=int)
        self.player_rating_bins = np.arange(1000, 3000, 50)
        self.player_rating_histogram = np.zeros(len(self.player_rating_bins) - 1, dtype=int)
        self.start_time = None
        super().reset_visualization()