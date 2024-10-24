# src\gui\visualizations\data_preparation_visualization.py

import numpy as np, time
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
        self.start_time = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(10, 8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.init_visualization()

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        self.ax_game_results = self.figure.add_subplot(gs[0, 0])
        self.ax_games_processed = self.figure.add_subplot(gs[0, 1])
        self.ax_game_lengths = self.figure.add_subplot(gs[1, 0])
        self.ax_player_ratings = self.figure.add_subplot(gs[1, 1])

        self.ax_game_results.set_title('Game Results Distribution', fontsize=12, fontweight='bold')
        self.ax_games_processed.set_title('Games Processed Over Time', fontsize=12, fontweight='bold')
        self.ax_game_lengths.set_title('Game Length Distribution', fontsize=12, fontweight='bold')
        self.ax_player_ratings.set_title('Player Rating Distribution', fontsize=12, fontweight='bold')

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

        self.update_game_results_plot()
        self.update_games_processed_plot()
        self.update_game_lengths_plot()
        self.update_player_ratings_plot()
        self.canvas.draw()

    def update_game_results_plot(self):
        self.ax_game_results.clear()
        results = [self.game_results.get(val, 0) for val in [1.0, -1.0, 0.0]]
        total = sum(results)
        if total > 0:
            percentages = [(r / total) * 100 for r in results]
            labels = ['White Wins', 'Black Wins', 'Draws']
            colors = ['#4CAF50', '#F44336', '#FFC107']
            explode = (0.05, 0.05, 0.05)
            self.ax_game_results.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=140,
                                     colors=colors, explode=explode, shadow=True)
            self.ax_game_results.axis('equal')
        else:
            self.ax_game_results.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=12)

    def update_games_processed_plot(self):
        self.ax_games_processed.clear()
        if self.total_games_processed and self.processing_times:
            self.ax_games_processed.plot(self.processing_times, self.total_games_processed, marker='o', color='#2196F3')
            self.ax_games_processed.set_xlabel('Time (s)', fontsize=10)
            self.ax_games_processed.set_ylabel('Total Games Processed', fontsize=10)
            self.ax_games_processed.grid(True, linestyle='--', alpha=0.7)
        else:
            self.ax_games_processed.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=12)

    def update_game_lengths_plot(self):
        self.ax_game_lengths.clear()
        if self.game_length_histogram is not None and np.sum(self.game_length_histogram) > 0:
            self.ax_game_lengths.bar(self.game_length_bins[:-1], self.game_length_histogram,
                                     width=np.diff(self.game_length_bins), align='edge',
                                     color='#9C27B0', edgecolor='black', alpha=0.7)
            self.ax_game_lengths.set_xlabel('Number of Moves', fontsize=10)
            self.ax_game_lengths.set_ylabel('Frequency', fontsize=10)
            self.ax_game_lengths.grid(True, linestyle='--', alpha=0.7)
        else:
            self.ax_game_lengths.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=12)

    def update_player_ratings_plot(self):
        self.ax_player_ratings.clear()
        if self.player_rating_histogram is not None and np.sum(self.player_rating_histogram) > 0:
            self.ax_player_ratings.bar(self.player_rating_bins[:-1], self.player_rating_histogram,
                                       width=np.diff(self.player_rating_bins), align='edge',
                                       color='#FF5722', edgecolor='black', alpha=0.7)
            self.ax_player_ratings.set_xlabel('Player Rating', fontsize=10)
            self.ax_player_ratings.set_ylabel('Frequency', fontsize=10)
            self.ax_player_ratings.grid(True, linestyle='--', alpha=0.7)
        else:
            self.ax_player_ratings.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=12)

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