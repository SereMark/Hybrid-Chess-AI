import numpy as np
import time
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DataPreparationVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.game_results = {1.0: 0, -1.0: 0, 0.0: 0}
        self.total_games_processed = []
        self.processing_times = []
        self.game_length_bins = None
        self.game_length_histogram = None
        self.player_rating_bins = None
        self.player_rating_histogram = None
        self.init_ui()
        self.start_time = None

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.init_visualization()

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2)
        self.ax_game_results = self.figure.add_subplot(gs[0, 0])
        self.ax_games_processed = self.figure.add_subplot(gs[0, 1])
        self.ax_game_lengths = self.figure.add_subplot(gs[1, 0])
        self.ax_player_ratings = self.figure.add_subplot(gs[1, 1])

        self.ax_game_results.set_title('Game Results Distribution')
        self.ax_games_processed.set_title('Games Processed Over Time')
        self.ax_game_lengths.set_title('Game Length Distribution')
        self.ax_player_ratings.set_title('Player Rating Distribution')

        self.canvas.draw()

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

        self.init_visualization()

        # Game Results Pie Chart
        self.ax_game_results.clear()
        results = [self.game_results.get(val, 0) for val in [1.0, -1.0, 0.0]]
        total = sum(results)
        if total > 0:
            percentages = [(r / total) * 100 for r in results]
            self.ax_game_results.pie(percentages, labels=['White Wins', 'Black Wins', 'Draws'],
                                     autopct='%1.1f%%', startangle=140)
        else:
            self.ax_game_results.text(0.5, 0.5, 'No Data Yet', ha='center', va='center')

        # Games Processed Over Time
        self.ax_games_processed.clear()
        if self.total_games_processed and self.processing_times:
            self.ax_games_processed.plot(self.processing_times, self.total_games_processed, marker='o')
            self.ax_games_processed.set_xlabel('Time (s)')
            self.ax_games_processed.set_ylabel('Total Games Processed')
        else:
            self.ax_games_processed.text(0.5, 0.5, 'No Data Yet', ha='center', va='center')

        # Game Length Distribution Histogram
        self.ax_game_lengths.clear()
        if self.game_length_histogram is not None:
            self.ax_game_lengths.bar(self.game_length_bins[:-1], self.game_length_histogram,
                                     width=np.diff(self.game_length_bins), align='edge', color='green', edgecolor='black')
            self.ax_game_lengths.set_xlabel('Number of Moves')
            self.ax_game_lengths.set_ylabel('Frequency')
        else:
            self.ax_game_lengths.text(0.5, 0.5, 'No Data Yet', ha='center', va='center')

        # Player Rating Distribution Histogram
        self.ax_player_ratings.clear()
        if self.player_rating_histogram is not None:
            self.ax_player_ratings.bar(self.player_rating_bins[:-1], self.player_rating_histogram,
                                       width=np.diff(self.player_rating_bins), align='edge', color='orange', edgecolor='black')
            self.ax_player_ratings.set_xlabel('Rating')
            self.ax_player_ratings.set_ylabel('Frequency')
        else:
            self.ax_player_ratings.text(0.5, 0.5, 'No Data Yet', ha='center', va='center')

        self.canvas.draw()

    def reset_visualizations(self):
        self.game_results = {1.0: 0, -1.0: 0, 0.0: 0}
        self.total_games_processed = []
        self.processing_times = []
        self.game_length_bins = None
        self.game_length_histogram = None
        self.player_rating_bins = None
        self.player_rating_histogram = None
        self.start_time = None
        self.init_visualization()