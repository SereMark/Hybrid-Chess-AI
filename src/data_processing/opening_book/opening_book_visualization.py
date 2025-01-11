from typing import Dict, Any
from collections import defaultdict
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from src.base.base_visualization import BaseVisualizationWidget, BasePlot

class OpeningBookVisualization(BaseVisualizationWidget):
    def __init__(self, parent: QWidget = None) -> None:
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.top_n = 10
        self.plot_key = 'main_plot'
        super().__init__(parent)

    def init_visualization(self) -> None:
        self._setup_ui()
        self._initialize_plot()

    def _setup_ui(self) -> None:
        title_label = QLabel("Most Common Openings")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        current_layout = self.layout()
        if current_layout is None:
            current_layout = QVBoxLayout()
            self.setLayout(current_layout)
        
        current_layout.insertWidget(0, title_label)

    def _initialize_plot(self) -> None:
        ax = self.figure.add_subplot(111)
        self.plots[self.plot_key] = BasePlot(ax=ax, title='Most Common Openings', xlabel='Opening Name', ylabel='Number of Games', title_fontsize=16, label_fontsize=12, tick_labelsize=10, grid_alpha=0.7, grid_color='#cccccc', font_family='sans-serif')
        self.update_graph()

    def reset_visualization(self) -> None:
        self.positions.clear()
        self.update_graph()

    def update_opening_book(self, data: Dict[str, Any]) -> None:
        self.positions = data.get('positions', {})
        self.update_graph()

    def update_graph(self) -> None:
        # Aggregate opening names and their occurrence counts
        opening_counts: Dict[str, int] = defaultdict(int)
        for fen, moves in self.positions.items():
            for move, stats in moves.items():
                opening_name = stats.get('name', 'Unknown')
                opening_counts[opening_name] += (stats.get('win', 0) + stats.get('draw', 0) + stats.get('loss', 0))

        if not opening_counts:
            self._display_no_data()
            return

        # Sort openings by count in descending order
        sorted_openings = sorted(opening_counts.items(), key=lambda item: item[1], reverse=True)
        top_openings = sorted_openings[:self.top_n]
        openings, counts = zip(*top_openings) if top_openings else ([], [])

        # Clear the previous plot
        self.plots[self.plot_key].ax.clear()
        self.plots[self.plot_key].apply_settings()

        # Create a bar chart
        bars = self.plots[self.plot_key].ax.bar(openings, counts, color='skyblue')
        self.plots[self.plot_key].ax.set_xticklabels(openings, rotation=45, ha='right')

        # Add count labels on top of each bar
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            self.plots[self.plot_key].ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        self.canvas.draw()

    def _display_no_data(self) -> None:
        self.plots[self.plot_key].ax.clear()
        self.plots[self.plot_key].ax.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center', fontsize=16, color='red', transform=self.plots[self.plot_key].ax.transAxes)
        self.plots[self.plot_key].ax.axis('off')
        self.canvas.draw()