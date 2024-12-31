import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from src.data_preparation.data_preparation_subtab import DataPreparationSubTab
from src.data_preparation.opening_book_subtab import OpeningBookSubTab
from src.training.supervised_training_subtab import SupervisedTrainingSubTab
from src.training.reinforcement_training_subtab import ReinforcementTrainingSubTab
from src.analysis.evaluation_subtab import EvaluationSubTab
from src.analysis.benchmark_subtab import BenchmarkSubTab
from src.style import stylesheet

class TabWidget(QWidget):
    def __init__(self, tabs, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        tab_widget = QTabWidget(self)
        for tab, title in tabs:
            tab_widget.addTab(tab, title)
        layout.addWidget(tab_widget)

class ChessMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Chess AI")
        self.setGeometry(100, 100, 1600, 800)
        main_tabs = QTabWidget(self)

        tab_data = [
            ([(DataPreparationSubTab(), "Data Preparation"), (OpeningBookSubTab(), "Opening Book")], "Data Preparation"),
            ([(SupervisedTrainingSubTab(), "Supervised Training"), (ReinforcementTrainingSubTab(), "Reinforcement Training")], "Training"),
            ([(EvaluationSubTab(), "Model Evaluation"), (BenchmarkSubTab(), "Benchmarking")], "Analysis")
        ]

        for tabs, title in tab_data:
            main_tabs.addTab(TabWidget(tabs), title)

        self.setCentralWidget(main_tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(stylesheet)
    main_window = ChessMainWindow()
    main_window.show()
    sys.exit(app.exec_())