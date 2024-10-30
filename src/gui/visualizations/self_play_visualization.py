from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class BasePlot:
    def __init__(self, ax, title, xlabel='', ylabel=''):
        self.ax = ax
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel(ylabel, fontsize=10)
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)


class SelfPlayVisualization(QWidget):
    def __init__(self, parent=None):
        super(SelfPlayVisualization, self).__init__(parent)
        
        self.layout = QVBoxLayout(self)
        
        self.figure = Figure(figsize=(5, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        self.layout.addWidget(self.canvas)
        
        self.ax1 = self.figure.add_subplot(211)
        self.plot1 = BasePlot(self.ax1, 'Game Results Over Time', 'Games Played', 'Number of Games')
        
        self.ax2 = self.figure.add_subplot(212)
        self.plot2 = BasePlot(self.ax2, 'Average Game Length Over Time', 'Games Played', 'Average Moves')
        
        self.setLayout(self.layout)
        
        self.games_played = []
        self.wins = []
        self.losses = []
        self.draws = []
        self.avg_game_lengths = []
    
    def update_stats(self, stats):
        games_played = stats['games_played']
        wins = stats['wins']
        losses = stats['losses']
        draws = stats['draws']
        avg_game_length = stats['average_game_length']
        
        self.games_played.append(games_played)
        self.wins.append(wins)
        self.losses.append(losses)
        self.draws.append(draws)
        self.avg_game_lengths.append(avg_game_length)
        
        self.ax1.clear()
        self.plot1 = BasePlot(self.ax1, 'Game Results Over Time', 'Games Played', 'Number of Games')
        self.ax1.plot(self.games_played, self.wins, label='Wins', color='green')
        self.ax1.plot(self.games_played, self.losses, label='Losses', color='red')
        self.ax1.plot(self.games_played, self.draws, label='Draws', color='blue')
        self.ax1.legend()
        
        self.ax2.clear()
        self.plot2 = BasePlot(self.ax2, 'Average Game Length Over Time', 'Games Played', 'Average Moves')
        self.ax2.plot(self.games_played, self.avg_game_lengths, color='purple')
        
        self.canvas.draw()
    
    def reset_visualization(self):
        self.games_played = []
        self.wins = []
        self.losses = []
        self.draws = []
        self.avg_game_lengths = []
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()