# src\gui\main.py

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from src.gui.tabs.game_tab import ChessGameTab
from src.gui.tabs.data_preparation_tab import DataPreparationTab
from src.gui.tabs.training_tab import TrainingTab
from src.gui.tabs.evaluation_tab import EvaluationTab

class ChessMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Chess AI")
        self.setGeometry(100, 100, 1600, 800)
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)
        for widget, name in [
            (ChessGameTab, "Gameplay"),
            (DataPreparationTab, "Data Preparation"),
            (TrainingTab, "Training"),
            (EvaluationTab, "Evaluation")
        ]:
            self.tab_widget.addTab(widget(), name)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ChessMainWindow()
    main_window.show()
    sys.exit(app.exec_())