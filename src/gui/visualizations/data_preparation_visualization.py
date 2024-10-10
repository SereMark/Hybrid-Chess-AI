import os
import h5py
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class DataPreparationVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.init_visualization()
    
    def init_visualization(self):
        self.figure.clear()
        for i in range(1, 3):
            ax = self.figure.add_subplot(1, 2, i)
            ax.text(0.5, 0.5, f'{"Data Visualization" if i == 1 else "Data Statistics"}\nNo Data Yet',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, fontweight='bold')
        self.canvas.draw()
    
    def update_data_visualization(self, dataset_path):
        if not os.path.exists(dataset_path):
            return
    
        try:
            with h5py.File(dataset_path, 'r') as h5_file:
                policy_targets = h5_file['policy_targets'][:]
                value_targets = h5_file['value_targets'][:]
        except Exception as e:
            return
    
        game_results = value_targets
        results = [np.sum(game_results == val) for val in [1.0, -1.0, 0.0]]
        total_games = sum(results)
    
        if total_games == 0:
            return
    
        move_frequencies = {}
        for move_idx in policy_targets:
            move_frequencies[move_idx] = move_frequencies.get(move_idx, 0) + 1
    
        self.figure.clear()
    
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax1.pie(results, labels=['White Wins', 'Black Wins', 'Draws'], autopct='%1.1f%%', startangle=140)
        ax1.set_title('Game Results Distribution')
    
        ax2 = self.figure.add_subplot(1, 2, 2)
        sorted_moves = sorted(move_frequencies.items(), key=lambda item: item[1], reverse=True)[:10]
        move_indices, frequencies = zip(*sorted_moves) if sorted_moves else ([], [])
        ax2.bar([f'Move {idx}' for idx in move_indices], frequencies)
        ax2.set_title('Top 10 Most Frequent Moves')
        ax2.set_xlabel('Move Index')
        ax2.set_ylabel('Frequency')
    
        self.canvas.draw()