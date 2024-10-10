import sys
import ctypes
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5.QtCore import Qt
from src.gui.game_tab import ChessGameTab
from src.gui.data_preparation_tab import DataPreparationTab
from src.gui.training_tab import TrainingTab
from src.gui.evaluation_tab import EvaluationTab

class ChessMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Chess AI")
        self.setGeometry(100, 100, 1200, 800)
        self.init_tabs()

    def init_tabs(self):
        self.tabs = QTabWidget()
        [self.tabs.addTab(widget(), name) for widget, name in [
            (ChessGameTab, "Gameplay"),
            (DataPreparationTab, "Data Preparation"),
            (TrainingTab, "Training"),
            (EvaluationTab, "Evaluation")
        ]]
        self.setCentralWidget(self.tabs)

if __name__ == "__main__":
    try: ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except: pass

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    main_window = ChessMainWindow()
    main_window.show()
    sys.exit(app.exec_())