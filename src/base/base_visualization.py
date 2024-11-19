from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout


class BasePlot:
    def __init__(self, ax, title='', xlabel='', ylabel='', invert_y=False,
                 title_fontsize=14, label_fontsize=12, tick_labelsize=10, grid_alpha=0.7):
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
        self.ax.set_title(self.title, fontsize=self.title_fontsize, fontweight='bold', pad=10)
        self.ax.set_xlabel(self.xlabel, fontsize=self.label_fontsize, labelpad=10)
        self.ax.set_ylabel(self.ylabel, fontsize=self.label_fontsize, labelpad=10)
        self.ax.tick_params(axis='both', which='major', labelsize=self.tick_labelsize)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=self.grid_alpha)
        if self.invert_y:
            self.ax.invert_yaxis()


class BasePlotWidget(QWidget):
    def __init__(self, title='', xlabel='', ylabel='', invert_y=False, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.base_plot = BasePlot(self.ax, title, xlabel, ylabel, invert_y)

    def update_plot(self):
        self.canvas.draw_idle()

    def reset_axis(self):
        self.ax.clear()
        self.base_plot.apply_settings()


class BaseVisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
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