import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class BasePlot:
    def __init__(self, ax, title, xlabel='', ylabel='', invert_y=False):
        self.ax = ax
        self.ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        self.ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
        self.ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
        self.ax.tick_params(axis='both', which='major', labelsize=9)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        if invert_y:
            self.ax.invert_yaxis()

class DataPreparationVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.game_results = {1.0: 0, -1.0: 0, 0.0: 0}
        self.total_moves = 0
        self.move_frequencies = {}
        self.game_lengths = []
        self.player_ratings = []
        self.init_ui()

    def init_ui(self):
        self.figure = Figure(figsize=(14, 10), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.init_visualization()

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 3, wspace=0.3, hspace=0.4)
        self.ax_game_results = self.figure.add_subplot(gs[0, 0])
        self.ax_total_moves = self.figure.add_subplot(gs[0, 1])
        self.ax_move_freq = self.figure.add_subplot(gs[0, 2])
        self.ax_game_lengths = self.figure.add_subplot(gs[1, 0])
        self.ax_player_ratings = self.figure.add_subplot(gs[1, 1])

        BasePlot(self.ax_game_results, title='Game Results Distribution', xlabel='', ylabel='Percentage')
        BasePlot(self.ax_total_moves, title='Total Moves Processed', xlabel='', ylabel='Number of Moves')
        BasePlot(self.ax_move_freq, title='Top 10 Most Frequent Moves', xlabel='Move', ylabel='Frequency')
        BasePlot(self.ax_game_lengths, title='Game Length Distribution', xlabel='Number of Moves', ylabel='Frequency')
        BasePlot(self.ax_player_ratings, title='Player Ratings Distribution', xlabel='Rating', ylabel='Frequency')

        for ax in [self.ax_game_results, self.ax_total_moves, self.ax_move_freq,
                   self.ax_game_lengths, self.ax_player_ratings]:
            ax.text(0.5, 0.5, 'No Data Yet', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, fontweight='bold', color='gray')

        self.canvas.draw()

    def update_data_visualization(self, stats):
        self.game_results = stats.get('game_results_counter', self.game_results)
        self.total_moves = stats.get('total_moves_processed', self.total_moves)
        new_move_freq = stats.get('move_frequencies', {})
        for move, freq in new_move_freq.items():
            self.move_frequencies[move] = self.move_frequencies.get(move, 0) + freq
        self.game_lengths = stats.get('game_lengths', self.game_lengths)
        self.player_ratings = stats.get('player_ratings', self.player_ratings)

        self.figure.clear()
        gs = self.figure.add_gridspec(2, 3, wspace=0.3, hspace=0.4)
        self.ax_game_results = self.figure.add_subplot(gs[0, 0])
        self.ax_total_moves = self.figure.add_subplot(gs[0, 1])
        self.ax_move_freq = self.figure.add_subplot(gs[0, 2])
        self.ax_game_lengths = self.figure.add_subplot(gs[1, 0])
        self.ax_player_ratings = self.figure.add_subplot(gs[1, 1])

        BasePlot(self.ax_game_results, title='Game Results Distribution', xlabel='', ylabel='Percentage')
        BasePlot(self.ax_total_moves, title='Total Moves Processed', xlabel='', ylabel='Number of Moves')
        BasePlot(self.ax_move_freq, title='Top 10 Most Frequent Moves', xlabel='Move', ylabel='Frequency')
        BasePlot(self.ax_game_lengths, title='Game Length Distribution', xlabel='Number of Moves', ylabel='Frequency')
        BasePlot(self.ax_player_ratings, title='Player Ratings Distribution', xlabel='Rating', ylabel='Frequency')

        results = [self.game_results.get(val, 0) for val in [1.0, -1.0, 0.0]]
        total = sum(results)
        if total > 0:
            percentages = [(r / total) * 100 for r in results]
            colors = ['#4CAF50', '#F44336', '#FFC107']
            self.ax_game_results.pie(percentages, labels=['White Wins', 'Black Wins', 'Draws'],
                                     autopct='%1.1f%%', startangle=140, colors=colors,
                                     textprops={'fontsize': 10})
        else:
            self.ax_game_results.text(0.5, 0.5, 'No Data Yet',
                                      ha='center', va='center',
                                      transform=self.ax_game_results.transAxes,
                                      fontsize=12, fontweight='bold', color='gray')

        self.ax_total_moves.clear()
        BasePlot(self.ax_total_moves, title='Total Moves Processed', xlabel='', ylabel='Number of Moves')
        self.ax_total_moves.bar(['Total Moves'], [self.total_moves], color='#2196F3')
        self.ax_total_moves.set_ylim(0, self.total_moves * 1.2 if self.total_moves > 0 else 1)
        self.ax_total_moves.text(0, self.total_moves + self.total_moves * 0.02,
                                 f"{self.total_moves}", ha='center', va='bottom',
                                 fontsize=10, fontweight='bold')

        self.ax_move_freq.clear()
        BasePlot(self.ax_move_freq, title='Top 10 Most Frequent Moves', xlabel='Move', ylabel='Frequency')
        top_moves = sorted(self.move_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_moves:
            moves, freqs = zip(*top_moves)
            y_pos = np.arange(len(moves))
            bars = self.ax_move_freq.barh(y_pos, freqs, color='#673AB7', edgecolor='black')
            self.ax_move_freq.set_yticks(y_pos)
            self.ax_move_freq.set_yticklabels(moves, fontsize=9)
            self.ax_move_freq.invert_yaxis()
            for bar, freq in zip(bars, freqs):
                self.ax_move_freq.text(bar.get_width() + max(freqs)*0.01, bar.get_y() + bar.get_height()/2,
                                       f'{freq}', va='center', fontsize=8, fontweight='bold')
        else:
            self.ax_move_freq.text(0.5, 0.5, 'No Moves Yet',
                                   ha='center', va='center',
                                   transform=self.ax_move_freq.transAxes,
                                   fontsize=12, fontweight='bold', color='gray')

        self.ax_game_lengths.clear()
        BasePlot(self.ax_game_lengths, title='Game Length Distribution', xlabel='Number of Moves', ylabel='Frequency')
        if self.game_lengths:
            bins = range(min(self.game_lengths), max(self.game_lengths) + 2)
            self.ax_game_lengths.hist(self.game_lengths, bins=bins, color='#FF5722', edgecolor='black')
        else:
            self.ax_game_lengths.text(0.5, 0.5, 'No Game Lengths Yet',
                                      ha='center', va='center',
                                      transform=self.ax_game_lengths.transAxes,
                                      fontsize=12, fontweight='bold', color='gray')

        self.ax_player_ratings.clear()
        BasePlot(self.ax_player_ratings, title='Player Ratings Distribution', xlabel='Rating', ylabel='Frequency')
        if self.player_ratings:
            self.ax_player_ratings.hist(self.player_ratings, bins=20, color='#009688', edgecolor='black')
        else:
            self.ax_player_ratings.text(0.5, 0.5, 'No Ratings Yet',
                                        ha='center', va='center',
                                        transform=self.ax_player_ratings.transAxes,
                                        fontsize=12, fontweight='bold', color='gray')

        self.canvas.draw()

    def reset_visualizations(self):
        self.game_results = {1.0: 0, -1.0: 0, 0.0: 0}
        self.total_moves = 0
        self.move_frequencies = {}
        self.game_lengths = []
        self.player_ratings = []
        self.init_visualization()