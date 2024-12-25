from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class BasePlot:
    def __init__(self, ax, title='', xlabel='', ylabel='', invert_y=False,
                 title_fontsize=14, label_fontsize=12, tick_labelsize=10, grid_alpha=0.7,
                 grid_color='#cccccc', line_width=1.5, font_family='sans-serif'):
        self.ax = ax
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.invert_y = invert_y
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_labelsize = tick_labelsize
        self.grid_alpha = grid_alpha
        self.grid_color = grid_color
        self.line_width = line_width
        self.font_family = font_family
        self.apply_settings()

    def apply_settings(self):
        self.ax.figure.set_facecolor('#fafafa')
        self.ax.set_facecolor('#ffffff')

        self.ax.set_title(self.title, fontsize=self.title_fontsize, weight='bold', pad=15, color='#333333', family=self.font_family)
        self.ax.set_xlabel(self.xlabel, fontsize=self.label_fontsize, labelpad=10, color='#333333', family=self.font_family)
        self.ax.set_ylabel(self.ylabel, fontsize=self.label_fontsize, labelpad=10, color='#333333', family=self.font_family)

        self.ax.tick_params(axis='both', which='major', labelsize=self.tick_labelsize, colors='#333333')

        for spine in self.ax.spines.values():
            spine.set_color('#333333')
            spine.set_linewidth(1.2)

        self.ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=self.grid_alpha, color=self.grid_color)

        if self.invert_y:
            self.ax.invert_yaxis()

        self.ax.figure.tight_layout()

    def plot_line(self, x, y, label='', color='#007acc', linestyle='-', linewidth=None, alpha=0.9):
        self.ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth or self.line_width, alpha=alpha)
        if label:
            self.ax.legend(fontsize=self.tick_labelsize, frameon=False, loc='best')

    def scatter(self, x, y, label='', color='#ff6600', marker='o', s=50, alpha=0.8):
        self.ax.scatter(x, y, label=label, color=color, marker=marker, s=s, alpha=alpha)
        if label:
            self.ax.legend(fontsize=self.tick_labelsize, frameon=False, loc='best')

    def bar(self, x, height, label='', color='#32a852', alpha=0.8, width=0.8):
        self.ax.bar(x, height, label=label, color=color, alpha=alpha, width=width)
        if label:
            self.ax.legend(fontsize=self.tick_labelsize, frameon=False, loc='best')


class BaseVisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 8), facecolor='#fafafa')
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.plots = {}
        self.init_visualization()

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
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, fontweight='bold', color='#555555', transform=ax.transAxes)