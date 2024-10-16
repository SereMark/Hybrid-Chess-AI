from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.init_visualization()

    def init_visualization(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 3)
        self.ax_game_results = self.figure.add_subplot(gs[0, 0])
        self.ax_total_moves = self.figure.add_subplot(gs[0, 1])
        self.ax_move_freq = self.figure.add_subplot(gs[0, 2])
        self.ax_game_lengths = self.figure.add_subplot(gs[1, 0])
        self.ax_player_ratings = self.figure.add_subplot(gs[1, 1])

        self.ax_game_results.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=14, fontweight='bold')
        self.ax_total_moves.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=14, fontweight='bold')
        self.ax_move_freq.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=14, fontweight='bold')
        self.ax_game_lengths.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=14, fontweight='bold')
        self.ax_player_ratings.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', fontsize=14, fontweight='bold')

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
        gs = self.figure.add_gridspec(2, 3)
        self.ax_game_results = self.figure.add_subplot(gs[0, 0])
        self.ax_total_moves = self.figure.add_subplot(gs[0, 1])
        self.ax_move_freq = self.figure.add_subplot(gs[0, 2])
        self.ax_game_lengths = self.figure.add_subplot(gs[1, 0])
        self.ax_player_ratings = self.figure.add_subplot(gs[1, 1])

        results = [self.game_results.get(val, 0) for val in [1.0, -1.0, 0.0]]
        self.ax_game_results.clear()
        self.ax_game_results.pie(results, labels=['White Wins', 'Black Wins', 'Draws'], autopct='%1.1f%%', startangle=140)
        self.ax_game_results.set_title('Game Results Distribution')

        self.ax_total_moves.clear()
        self.ax_total_moves.bar(['Total Moves'], [self.total_moves], color='skyblue')
        self.ax_total_moves.set_title('Total Moves Processed')
        self.ax_total_moves.set_ylabel('Number of Moves')
        self.ax_total_moves.text(0, self.total_moves + self.total_moves * 0.01, f"{self.total_moves}", ha='center')

        self.ax_move_freq.clear()
        top_moves = sorted(self.move_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_moves:
            moves, freqs = zip(*top_moves)
            self.ax_move_freq.bar(moves, freqs)
            self.ax_move_freq.set_title('Top 10 Most Frequent Moves')
            self.ax_move_freq.set_xticklabels(moves, rotation=45)
            self.ax_move_freq.set_ylabel('Frequency')
        else:
            self.ax_move_freq.text(0.5, 0.5, 'No Moves Yet',
                                   ha='center', va='center', transform=self.ax_move_freq.transAxes,
                                   fontsize=14, fontweight='bold')

        self.ax_game_lengths.clear()
        if self.game_lengths:
            self.ax_game_lengths.hist(self.game_lengths, bins=range(1, max(self.game_lengths) + 2), edgecolor='black')
            self.ax_game_lengths.set_title('Game Length Distribution')
            self.ax_game_lengths.set_xlabel('Number of Moves')
            self.ax_game_lengths.set_ylabel('Frequency')
        else:
            self.ax_game_lengths.text(0.5, 0.5, 'No Game Lengths Yet',
                                      ha='center', va='center', transform=self.ax_game_lengths.transAxes,
                                      fontsize=14, fontweight='bold')

        self.ax_player_ratings.clear()
        if self.player_ratings:
            self.ax_player_ratings.hist(self.player_ratings, bins=20, edgecolor='black')
            self.ax_player_ratings.set_title('Player Ratings Distribution')
            self.ax_player_ratings.set_xlabel('Rating')
            self.ax_player_ratings.set_ylabel('Frequency')
        else:
            self.ax_player_ratings.text(0.5, 0.5, 'No Ratings Yet',
                                        ha='center', va='center', transform=self.ax_player_ratings.transAxes,
                                        fontsize=14, fontweight='bold')

        self.figure.tight_layout()
        self.canvas.draw()