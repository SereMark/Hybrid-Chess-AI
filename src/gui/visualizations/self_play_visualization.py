from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np, time


class BasePlot:
    def __init__(self, ax, title, xlabel='', ylabel=''):
        self.ax = ax
        self.ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
        self.ax.set_xlabel(xlabel, fontsize=9)
        self.ax.set_ylabel(ylabel, fontsize=9)
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        self.ax.tick_params(labelsize=8)
        

class SelfPlayVisualization(QWidget):
    def __init__(self, parent=None):
        super(SelfPlayVisualization, self).__init__(parent)
        self.setup_ui()
        self.initialize_data_storage()
        self.last_update_time = time.time()
        self.update_interval = 0.5

    def setup_ui(self):
        self.layout = QVBoxLayout(self)
        
        self.figure = Figure(figsize=(8, 10))
        self.figure.subplots_adjust(hspace=0.4)
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        gs = self.figure.add_gridspec(3, 2)
        
        self.ax1 = self.figure.add_subplot(gs[0, :])
        self.plot1 = BasePlot(self.ax1, 'Game Outcomes', 'Games Played', 'Percentage')
        
        self.ax2 = self.figure.add_subplot(gs[1, 0])
        self.plot2 = BasePlot(self.ax2, 'Game Length Distribution', 'Number of Moves', 'Frequency')
        
        self.ax3 = self.figure.add_subplot(gs[1, 1])
        self.plot3 = BasePlot(self.ax3, 'Training Speed', 'Games Played', 'Games/Second')
        
        self.ax4 = self.figure.add_subplot(gs[2, :])
        self.plot4 = BasePlot(self.ax4, 'Training Progress Metrics', 'Games Played', 'Value')
        
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

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
        self.min_lengths = []
        self.max_lengths = []
        
    def reset_visualization(self):
        self.initialize_data_storage()
        self.last_update_time = time.time()
        
        self.ax1.clear()
        self.plot1 = BasePlot(self.ax1, 'Game Outcomes Over Time', 'Games Played', 'Percentage (%)')
        self.ax1.set_ylim(0, 100)
        
        self.ax2.clear()
        self.plot2 = BasePlot(self.ax2, 'Game Length Range', 'Games Played', 'Number of Moves')
        
        self.ax3.clear()
        self.plot3 = BasePlot(self.ax3, 'Training Speed', 'Games Played', 'Games/Second')
        
        self.ax4.clear()
        self.plot4 = BasePlot(self.ax4, 'Training Progress', 'Games Played', 'Value')
        self.ax4.set_ylim(0, 1.1)
        
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def update_stats(self, stats):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        self.last_update_time = current_time

        games_played = stats['games_played']
        wins = stats['wins']
        losses = stats['losses']
        draws = stats['draws']
        avg_length = stats['average_game_length']
        min_length = stats['min_game_length']
        max_length = stats['max_game_length']
        speed = stats['games_per_second']
        
        total_games = wins + losses + draws
        if total_games > 0:
            win_rate = (wins / total_games) * 100
            loss_rate = (losses / total_games) * 100
            draw_rate = (draws / total_games) * 100
        else:
            win_rate = loss_rate = draw_rate = 0

        self.games_played.append(games_played)
        self.win_rates.append(win_rate)
        self.loss_rates.append(loss_rate)
        self.draw_rates.append(draw_rate)
        self.avg_game_lengths.append(avg_length)
        self.min_lengths.append(min_length)
        self.max_lengths.append(max_length)
        self.games_per_second.append(speed)
        
        self.ax1.clear()
        self.plot1 = BasePlot(self.ax1, 'Game Outcomes Over Time', 'Games Played', 'Percentage (%)')
        self.ax1.fill_between(self.games_played, self.win_rates, 
                            label='Wins', color='#2ecc71', alpha=0.7)
        self.ax1.fill_between(self.games_played, self.draw_rates, 
                            label='Draws', color='#3498db', alpha=0.7)
        self.ax1.fill_between(self.games_played, self.loss_rates, 
                            label='Losses', color='#e74c3c', alpha=0.7)
        self.ax1.legend(loc='upper left', fontsize=8)
        self.ax1.set_ylim(0, 100)
        if self.games_played:
            self.ax1.set_xlim(0, max(self.games_played) * 1.1)

        self.ax2.clear()
        self.plot2 = BasePlot(self.ax2, 'Game Length Range', 'Games Played', 'Number of Moves')
        self.ax2.fill_between(self.games_played, self.max_lengths, self.min_lengths, 
                            color='#9b59b6', alpha=0.3, label='Game Length Range')
        self.ax2.plot(self.games_played, self.avg_game_lengths, 
                    color='#8e44ad', label='Average', linewidth=2)
        self.ax2.legend(fontsize=8)
        if self.games_played:
            self.ax2.set_xlim(0, max(self.games_played) * 1.1)
            if self.min_lengths:
                self.ax2.set_ylim(min(self.min_lengths) * 0.9, max(self.max_lengths) * 1.1)

        self.ax3.clear()
        self.plot3 = BasePlot(self.ax3, 'Training Speed', 'Games Played', 'Games/Second')
        self.ax3.plot(self.games_played, self.games_per_second, 
                    color='#f1c40f', linewidth=2, label='Speed')
        self.ax3.legend(fontsize=8)
        if self.games_played:
            self.ax3.set_xlim(0, max(self.games_played) * 1.1)
            if self.games_per_second:
                self.ax3.set_ylim(0, max(self.games_per_second) * 1.1)
        
        self.ax4.clear()
        self.plot4 = BasePlot(self.ax4, 'Training Progress', 'Games Played', 'Value')
        
        norm_win_rate = np.array(self.win_rates) / 100
        norm_speed = np.array(self.games_per_second) / max(self.games_per_second) if self.games_per_second else np.array([0])
        norm_avg_length = np.array(self.avg_game_lengths) / max(self.avg_game_lengths) if self.avg_game_lengths else np.array([0])
        
        self.ax4.plot(self.games_played, norm_win_rate, 
                    label='Win Rate', color='#2ecc71', linewidth=2)
        self.ax4.plot(self.games_played, norm_speed, 
                    label='Relative Speed', color='#f1c40f', linewidth=2)
        self.ax4.plot(self.games_played, norm_avg_length, 
                    label='Relative Game Length', color='#9b59b6', linewidth=2)
        self.ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        if self.games_played:
            self.ax4.set_xlim(0, max(self.games_played) * 1.1)
        self.ax4.set_ylim(0, 1.1)
        
        self.figure.tight_layout()
        self.canvas.draw_idle()