from src.gui.visualizations.self_play_visualization import SelfPlayVisualization
from PyQt5.QtWidgets import QWidget, QVBoxLayout

class SelfPlayTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.training_visualization = SelfPlayVisualization()
        layout.addWidget(self.training_visualization)
        self.setLayout(layout)

    def update_training_visualization(self, losses, accuracies):
        self.training_visualization.update_training_visualization(losses, accuracies)