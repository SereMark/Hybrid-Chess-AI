import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QStatusBar, QListWidget,
                             QSplitter, QWidget, QVBoxLayout, QAction, QFileDialog,
                             QMessageBox)
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from board_widget import ChessBoardWidget
from visualization import ChessAIVisualization
import chess.pgn

# Define the dark theme stylesheet
dark_stylesheet = """
QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
}
QPushButton {
    background-color: #3c3f41;
    color: #ffffff;
    border: 1px solid #4b4b4b;
    padding: 5px;
}
QPushButton:hover {
    background-color: #4b4b4b;
}
QLineEdit, QTextEdit, QListWidget {
    background-color: #3c3f41;
    color: #ffffff;
    border: 1px solid #4b4b4b;
}
QLabel {
    color: #ffffff;
}
QMenuBar, QMenu {
    background-color: #2b2b2b;
    color: #ffffff;
}
QMenuBar::item:selected, QMenu::item:selected {
    background-color: #4b4b4b;
}
QStatusBar {
    background-color: #2b2b2b;
    color: #ffffff;
}
"""

class ChessMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Chess AI")
        self.setGeometry(100, 100, 1200, 800)
        self.move_list = []

        self.board_widget = ChessBoardWidget(self)
        self.board_widget.move_made.connect(self.update_status_bar)
        self.board_widget.show_hint_signal.connect(self.show_hint)

        self.move_history = QListWidget()
        self.move_history.setFixedWidth(200)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.board_widget)
        self.splitter.addWidget(self.move_history)

        self.setCentralWidget(self.splitter)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.ai_visualization = ChessAIVisualization()
        self.visualization_canvas = self.ai_visualization.canvas
        self.splitter.addWidget(self.ai_visualization)

        # Add menu bar
        self.create_menus()

    def create_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        view_menu = menubar.addMenu("View")
        settings_menu = menubar.addMenu("Settings")
        help_menu = menubar.addMenu("Help")

        # File menu actions
        new_game_action = QAction("New Game", self)
        new_game_action.triggered.connect(self.board_widget.reset_board)
        save_game_action = QAction("Save Game", self)
        save_game_action.triggered.connect(self.save_game)
        load_game_action = QAction("Load Game", self)
        load_game_action.triggered.connect(self.load_game)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(new_game_action)
        file_menu.addAction(save_game_action)
        file_menu.addAction(load_game_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # View menu actions
        flip_board_action = QAction("Flip Board", self)
        flip_board_action.triggered.connect(self.board_widget.flip_board)
        view_menu.addAction(flip_board_action)

        # Settings menu actions
        ai_difficulty_action = QAction("Set AI Difficulty", self)
        ai_difficulty_action.triggered.connect(self.set_ai_difficulty)
        settings_menu.addAction(ai_difficulty_action)

        # Help menu actions
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def save_game(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Game", "", "Portable Game Notation (*.pgn)", options=options)
        if filename:
            try:
                with open(filename, 'w') as f:
                    exporter = chess.pgn.FileExporter(f)
                    game = chess.pgn.Game.from_board(self.board_widget.board_state)
                    game.accept(exporter)
                QMessageBox.information(self, "Save Game", "Game saved successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Save Game", f"Failed to save game: {e}")

    def load_game(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Game", "", "Portable Game Notation (*.pgn)", options=options)
        if filename:
            try:
                with open(filename, 'r') as f:
                    game = chess.pgn.read_game(f)
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                    self.board_widget.board_state = board
                    self.board_widget.update_board()
                    self.move_list.clear()
                    self.move_history.clear()
                    QMessageBox.information(self, "Load Game", "Game loaded successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Load Game", f"Failed to load game: {e}")

    def set_ai_difficulty(self):
        # TODO: Implement AI difficulty setting
        QMessageBox.information(self, "AI Difficulty", "Feature not implemented yet.")

    def show_about_dialog(self):
        QMessageBox.information(
            self, "About",
            "Hybrid Chess AI\nVersion 1.0\nDeveloped with PyQt5 and python-chess library.")

    def update_status_bar(self, message):
        self.status_bar.showMessage(message)
        if message.startswith("AI moved:"):
            move = message.split(': ')[1]
            self.move_list.append(move)
            self.update_move_history()
            self.update_visualization()
        elif message == "Board reset" or message == "Move undone":
            self.move_list.clear()
            self.move_history.clear()
            self.ai_visualization.clear_visualization()
        else:
            move = message
            self.move_list.append(move)
            self.update_move_history()

    def update_move_history(self):
        self.move_history.clear()
        for idx, move in enumerate(self.move_list):
            if idx % 2 == 0:
                item_text = f"{(idx // 2) + 1}. {move}"
            else:
                item_text = f"{move}"
            self.move_history.addItem(item_text)

    def show_hint(self):
        self.board_widget.show_hint()

    def update_visualization(self):
        # TODO: Implement AI evaluation visualization
        import random
        evaluation = random.uniform(-1, 1)
        self.ai_visualization.move_evaluations.append(evaluation)
        self.ai_visualization.visualize_evaluation(self.ai_visualization.move_evaluations)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(dark_stylesheet)
    main_window = ChessMainWindow()
    main_window.show()
    sys.exit(app.exec_())