import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from src.data_preparation.data_preparation_subtab import DataPreparationSubTab
from src.data_preparation.opening_book_subtab import OpeningBookSubTab
from src.training.supervised_training_subtab import SupervisedTrainingSubTab
from src.training.reinforcement_training_subtab import ReinforcementTrainingSubTab
from src.evaluation.evaluation_subtab import EvaluationSubTab
from src.evaluation.benchmark_subtab import BenchmarkSubTab

class TabWidget(QWidget):
    def __init__(self, tabs, tooltips=None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        tab_widget = QTabWidget(self)
        for i, (tab, title) in enumerate(tabs):
            tab_widget.addTab(tab, title)
            if tooltips and i < len(tooltips): tab_widget.setTabToolTip(i, tooltips[i])
        layout.addWidget(tab_widget)
        self.setLayout(layout)

class ChessMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Chess AI")
        self.setGeometry(100, 100, 1600, 800)
        main_tabs = QTabWidget(self)
        main_tabs.addTab(TabWidget([(DataPreparationSubTab(), "Data Preparation"), (OpeningBookSubTab(), "Opening Book")], ["Convert raw chess data (e.g., PGN files) into a processed format suitable for training.", "Generate an opening book by analyzing processed chess game data."]), "Data Preparation")
        main_tabs.addTab(TabWidget([(SupervisedTrainingSubTab(), "Supervised Training"), (ReinforcementTrainingSubTab(), "Reinforcement Training")], ["Train the AI using labeled data from historical chess games.", "Enhance the AI through self-play reinforcement learning."]), "Training")
        main_tabs.addTab(TabWidget([(EvaluationSubTab(), "Model Evaluation"), (BenchmarkSubTab(), "Benchmarking")], ["Evaluate the AI's performance on unseen test data.", "Benchmark the AI against other models or predefined baselines."]), "Evaluation")
        self.setCentralWidget(main_tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ChessMainWindow()
    main_window.show()
    sys.exit(app.exec_())