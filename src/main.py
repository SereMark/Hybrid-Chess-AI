import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from src.data_preparation.data_preparation_tab import DataPreparationTab
from src.supervised.supervised_training_tab import SupervisedTab
from src.reinforcement.reinforcement_training_tab import ReinforcementTab
from src.evaluation.evaluation_tab import EvaluationTab

class ChessMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Chess AI")
        self.setGeometry(100, 100, 1600, 800)
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)
        self.tab_widget.addTab(DataPreparationTab(), "Data Preparation")
        self.tab_widget.addTab(SupervisedTab(), "Supervised Training")
        self.tab_widget.addTab(ReinforcementTab(), "Reinforcement Training")
        self.tab_widget.addTab(EvaluationTab(), "Evaluation")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ChessMainWindow()
    main_window.show()
    sys.exit(app.exec_())