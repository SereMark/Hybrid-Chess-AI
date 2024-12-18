from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class BasePlot:
    def __init__(self, ax, title='', xlabel='', ylabel='', invert_y=False,
                 title_fontsize=12, label_fontsize=10, tick_labelsize=8, grid_alpha=0.7):
        self.ax = ax
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.invert_y = invert_y
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_labelsize = tick_labelsize
        self.grid_alpha = grid_alpha
        self.apply_settings()

    def apply_settings(self):
        self.ax.set_facecolor('#2a2a2a')
        self.ax.title.set_color('#eeeeee')
        self.ax.set_xlabel(self.xlabel, fontsize=self.label_fontsize, labelpad=8, color='#eeeeee')
        self.ax.set_ylabel(self.ylabel, fontsize=self.label_fontsize, labelpad=8, color='#eeeeee')
        self.ax.tick_params(axis='both', which='major', labelsize=self.tick_labelsize, colors='#eeeeee')
        for spine in self.ax.spines.values():
            spine.set_color('#eeeeee')
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=self.grid_alpha, color='#444444')
        if self.invert_y:
            self.ax.invert_yaxis()

class BaseVisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 8), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.plots = {}
        self.init_visualization()

    def init_visualization(self):
        pass

    def update_visualization(self):
        self.canvas.draw_idle()

    def reset_visualization(self):
        self.figure.clear()
        self.plots = {}
        self.init_visualization()
        self.canvas.draw_idle()

    def clear_axis(self, plot_key):
        ax = self.plots[plot_key].ax
        ax.clear()
        self.plots[plot_key].apply_settings()

    def add_text_to_axis(self, plot_key, text):
        ax = self.plots[plot_key].ax
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, fontweight='bold', color='#eeeeee', transform=ax.transAxes)