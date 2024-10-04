import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QListWidget,
    QWidget, QVBoxLayout, QAction, QFileDialog,
    QMessageBox, QHBoxLayout, QLabel, QInputDialog, QSplitter, QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QFont
import matplotlib
matplotlib.use('Qt5Agg')
from board_widget import ChessBoardWidget
from visualization import ChessAIVisualization
import chess.pgn

dark_stylesheet = """
QMainWindow {
    background-color: #121212;
}
QWidget {
    background-color: #121212;
    color: #E0E0E0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 12pt;
}
QPushButton {
    background-color: #1E1E1E;
    color: #E0E0E0;
    border: none;
    padding: 8px 12px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #333333;
}
QLineEdit, QTextEdit, QListWidget {
    background-color: #1E1E1E;
    color: #E0E0E0;
    border: 1px solid #333333;
    border-radius: 4px;
}
QLabel {
    color: #E0E0E0;
}
QMenuBar, QMenu {
    background-color: #121212;
    color: #E0E0E0;
}
QMenuBar::item:selected, QMenu::item:selected {
    background-color: #333333;
}
QStatusBar {
    background-color: #121212;
    color: #E0E0E0;
}
QToolTip {
    background-color: #333333;
    color: #E0E0E0;
    border: none;
}
QMessageBox {
    background-color: #121212;
    color: #E0E0E0;
}
QScrollBar:vertical {
    background: #1E1E1E;
    width: 12px;
    margin: 15px 3px 15px 3px;
    border: 1px solid #333333;
}
QScrollBar::handle:vertical {
    background: #333333;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    background: none;
    height: 15px;
}
"""

class ChessMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Chess AI")
        self.setGeometry(100, 100, 1200, 800)
        self.player_color = chess.WHITE

        self.board_widget = ChessBoardWidget(self)
        self.board_widget.move_made.connect(self.update_status_bar)
        self.board_widget.show_hint_signal.connect(self.show_hint)
        self.board_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.move_history = QListWidget()
        self.move_history.setStyleSheet("border: none;")
        self.move_history_font = QFont('Segoe UI', 10)
        self.move_history.setFont(self.move_history_font)
        self.move_history.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.ai_visualization = ChessAIVisualization()

        self.white_time_label = QLabel("White: 10:00")
        self.black_time_label = QLabel("Black: 10:00")
        self.white_time_label.setStyleSheet("font-size: 12pt;")
        self.black_time_label.setStyleSheet("font-size: 12pt;")
        self.timer_layout = QHBoxLayout()
        self.timer_layout.addWidget(self.white_time_label)
        self.timer_layout.addStretch()
        self.timer_layout.addWidget(self.black_time_label)

        self.undo_button = QPushButton("Undo Move")
        self.undo_button.setToolTip("Undo the last move")
        self.help_button = QPushButton("Hint")
        self.help_button.setToolTip("Show a hint for your next move")
        self.undo_button.clicked.connect(self.board_widget.undo_move)
        self.help_button.clicked.connect(self.show_hint)
        button_style = """
        QPushButton {
            background-color: #1E1E1E;
            color: #E0E0E0;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #333333;
        }
        """
        self.undo_button.setStyleSheet(button_style)
        self.help_button.setStyleSheet(button_style)
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.undo_button)
        self.button_layout.addWidget(self.help_button)

        self.right_layout = QVBoxLayout()
        self.right_layout.addLayout(self.timer_layout)
        self.right_layout.addLayout(self.button_layout)

        self.board_history_splitter = QSplitter(Qt.Vertical)
        self.board_history_splitter.addWidget(self.board_widget)
        self.board_history_splitter.addWidget(self.move_history)
        self.board_history_splitter.setStretchFactor(0, 3)
        self.board_history_splitter.setStretchFactor(1, 1)

        self.right_layout.addWidget(self.board_history_splitter)
        self.right_widget = QWidget()
        self.right_widget.setLayout(self.right_layout)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(1)
        self.main_splitter.setStyleSheet("QSplitter::handle { background-color: #333333; }")
        self.main_splitter.addWidget(self.ai_visualization)
        self.main_splitter.addWidget(self.right_widget)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 1)

        self.central_widget = QWidget()
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.main_splitter)
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        self.time_control = 10 * 60
        self.white_time = self.time_control
        self.black_time = self.time_control
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_clock)
        self.current_player = chess.WHITE
        self.timer_running = False

        self.create_menus()

        self.setWindowIcon(QIcon(os.path.join('src/gui/assets', 'chess_icon.png')))

    def create_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        settings_menu = menubar.addMenu("Settings")
        help_menu = menubar.addMenu("Help")

        new_game_action = QAction("New Game", self)
        new_game_action.setShortcut("Ctrl+N")
        new_game_action.triggered.connect(self.new_game)
        save_game_action = QAction("Save Game", self)
        save_game_action.setShortcut("Ctrl+S")
        save_game_action.triggered.connect(self.save_game)
        load_game_action = QAction("Load Game", self)
        load_game_action.setShortcut("Ctrl+O")
        load_game_action.triggered.connect(self.load_game)
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        file_menu.addAction(new_game_action)
        file_menu.addAction(save_game_action)
        file_menu.addAction(load_game_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        ai_difficulty_action = QAction("Set AI Difficulty", self)
        ai_difficulty_action.triggered.connect(self.set_ai_difficulty)
        time_control_action = QAction("Set Time Control", self)
        time_control_action.triggered.connect(self.set_time_control)
        settings_menu.addAction(ai_difficulty_action)
        settings_menu.addAction(time_control_action)

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
                    self.board_widget.game_over = self.board_widget.board_state.is_game_over()
                    self.update_status_bar("Game loaded")
                QMessageBox.information(self, "Load Game", "Game loaded successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Load Game", f"Failed to load game: {e}")

    def set_ai_difficulty(self):
        QMessageBox.information(self, "AI Difficulty", "Feature not implemented yet.")

    def set_time_control(self):
        minutes, ok = QInputDialog.getInt(self, "Set Time Control", "Enter minutes per player:", min=1, max=180)
        if ok:
            self.time_control = minutes * 60
            self.white_time = self.time_control
            self.black_time = self.time_control
            self.update_clock_labels()
            self.timer.start(1000)

    def show_about_dialog(self):
        QMessageBox.information(
            self, "About",
            "Hybrid Chess AI\nVersion 1.0\nDeveloped with PyQt5 and python-chess library.")

    def update_status_bar(self, message):
        if message.startswith("AI moved:") or message.startswith("Move made:") or message == "Game loaded":
            self.update_move_history()
            self.update_visualization()
            self.current_player = self.board_widget.board_state.turn
            if not self.timer_running and not self.board_widget.game_over:
                self.timer.start(1000)
                self.timer_running = True
        elif message == "Board reset" or message == "Move undone":
            self.update_move_history()
            self.ai_visualization.clear_visualization()
            self.white_time = self.time_control
            self.black_time = self.time_control
            self.current_player = self.board_widget.board_state.turn
            self.update_clock_labels()
            self.timer_running = False
            self.timer.stop()

    def update_move_history(self):
        self.move_history.clear()
        board = chess.Board()
        move_stack = self.board_widget.board_state.move_stack.copy()
        for idx, move in enumerate(move_stack):
            san = board.san(move)
            if idx % 2 == 0:
                item_text = f"{(idx // 2) + 1}. {san}"
            else:
                item_text = f"{san}"
            self.move_history.addItem(item_text)
            board.push(move)
        self.move_history.scrollToBottom()

    def show_hint(self):
        self.board_widget.show_hint()

    def update_visualization(self):
        import random
        evaluation = random.uniform(-1, 1)
        self.ai_visualization.move_evaluations.append(evaluation)

        legal_moves = list(self.board_widget.board_state.legal_moves)

        if legal_moves:
            policy_output = {move.uci(): random.random() for move in legal_moves}
            total = sum(policy_output.values())
            for move in policy_output:
                policy_output[move] /= total

            mcts_stats = {
                'simulations': random.randint(100, 500),
                'nodes_explored': random.randint(50, 200),
                'best_move': max(policy_output, key=policy_output.get)
            }
        else:
            mcts_stats = {
                'simulations': 0,
                'nodes_explored': 0,
                'best_move': None
            }

        self.ai_visualization.visualize_evaluation(
            evaluations=self.ai_visualization.move_evaluations,
            policy_output=policy_output if legal_moves else {},
            mcts_stats=mcts_stats
        )

    def update_clock(self):
        if self.board_widget.board_state.is_game_over():
            self.timer.stop()
            self.timer_running = False
            return
        if self.current_player == chess.WHITE:
            self.white_time -= 1
        else:
            self.black_time -= 1
        self.update_clock_labels()
        if self.white_time <= 0 or self.black_time <= 0:
            self.timer.stop()
            winner = "Black" if self.white_time <= 0 else "White"
            QMessageBox.information(self, "Time Over", f"{winner} wins on time!")
            self.board_widget.game_over = True
            self.board_widget.update_board()

    def update_clock_labels(self):
        white_minutes = self.white_time // 60
        white_seconds = self.white_time % 60
        black_minutes = self.black_time // 60
        black_seconds = self.black_time % 60
        self.white_time_label.setText(f"White: {white_minutes:02d}:{white_seconds:02d}")
        self.black_time_label.setText(f"Black: {black_minutes:02d}:{black_seconds:02d}")

    def new_game(self):
        color_choice = QMessageBox.question(
            self, "Choose Color", "Do you want to play as White?", QMessageBox.Yes | QMessageBox.No)
        if color_choice == QMessageBox.Yes:
            self.player_color = chess.WHITE
        else:
            self.player_color = chess.BLACK
        self.board_widget.reset_board(self.player_color)
        self.white_time = self.time_control
        self.black_time = self.time_control
        self.update_clock_labels()
        self.timer_running = False
        self.timer.stop()
        if self.player_color == chess.BLACK:
            self.board_widget.ai_move()

if __name__ == "__main__":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setStyleSheet(dark_stylesheet)
    main_window = ChessMainWindow()
    main_window.show()
    sys.exit(app.exec_())