import random, chess
from PyQt5.QtWidgets import (
    QLabel, QWidget, QSizePolicy, QMessageBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget, QListWidget, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF
from PyQt5.QtGui import QPainter, QPixmap, QBrush, QColor
from PyQt5.QtSvg import QSvgRenderer

from src.common.base_tab import BaseTab
from src.common.common_widgets import create_labeled_combo, create_labeled_spinbox
from src.gui.visualizations.game_visualization import GameVisualization
from scripts.chess_utils import ChessEngine

class ChessSquareLabel(QLabel):
    square_clicked_signal = pyqtSignal(int)

    def __init__(self, parent, square_index):
        super().__init__(parent)
        self.square_index = square_index
        self.is_highlighted = False
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.square_clicked_signal.emit(self.square_index)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.is_highlighted:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(QColor("#FFD700")))
            painter.setPen(Qt.NoPen)
            radius = min(self.width(), self.height()) // 12
            painter.drawEllipse(self.rect().center(), radius, radius)
            painter.end()

class ChessBoardView(QWidget):
    move_made_signal = pyqtSignal(str)

    def __init__(self, engine: ChessEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.engine.move_made_signal.connect(self.on_move_made)
        self.engine.game_over_signal.connect(self.on_game_over)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout_grid = QGridLayout()
        self.layout_grid.setSpacing(0)
        self.layout_grid.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout_grid)
        self.square_labels = []
        self.selected_square = None
        self.is_orientation_flipped = True
        self.svg_renderer = QSvgRenderer('src/gui/assets/chess_pieces.svg')
        self.initialize_board()

    def initialize_board(self):
        for i in reversed(range(self.layout_grid.count())):
            widget = self.layout_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.square_labels = [ChessSquareLabel(self, self.calculate_square_index(row, col)) for row in range(8) for col in range(8)]
        for square in self.square_labels:
            square.square_clicked_signal.connect(self.on_square_clicked)
            row, col = divmod(square.square_index, 8)
            self.layout_grid.addWidget(square, row, col)
        for i in range(8):
            self.layout_grid.setRowStretch(i, 1)
            self.layout_grid.setColumnStretch(i, 1)
        self.apply_board_colors()
        self.refresh_board()

    def calculate_square_index(self, row, col):
        return (7 - row) * 8 + col if self.is_orientation_flipped else row * 8 + (7 - col)

    def apply_board_colors(self):
        for square in self.square_labels:
            row, col = square.square_index // 8, square.square_index % 8
            color = '#1E1E1E' if (row + col) % 2 == 0 else '#121212'
            square.setStyleSheet(f"background-color: {color};")

    def on_square_clicked(self, square_index):
        if self.engine.is_game_over:
            QMessageBox.information(self, "Game Over", "The game is over. Start a new game to continue.")
            return
        piece = self.engine.board.piece_at(square_index)
        if self.selected_square is None:
            if piece and piece.color == self.engine.player_color:
                self.selected_square = square_index
                self.highlight_possible_moves(square_index)
        else:
            if square_index != self.selected_square:
                if not self.engine.make_move(self.selected_square, square_index):
                    QMessageBox.warning(self, "Invalid Move", "That move is not legal.")
            self.remove_highlights()
            self.selected_square = None

    def on_move_made(self, message):
        self.move_made_signal.emit(message)
        self.refresh_board()

    def on_game_over(self, message):
        QMessageBox.information(self, "Game Over", message)
        self.move_made_signal.emit(message)
        self.refresh_board()

    def highlight_possible_moves(self, square_index):
        self.remove_highlights()
        legal_moves = [move.to_square for move in self.engine.board.legal_moves if move.from_square == square_index]
        for sq in self.square_labels:
            if sq.square_index in legal_moves:
                sq.is_highlighted = True
                sq.update()

    def remove_highlights(self):
        for sq in self.square_labels:
            if sq.is_highlighted:
                sq.is_highlighted = False
                sq.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_board()

    def refresh_board(self):
        if self.width() < 50 or self.height() < 50:
            return
        for square in self.square_labels:
            piece = self.engine.board.piece_at(square.square_index)
            square.setPixmap(self.generate_piece_pixmap(piece, square) if piece else QPixmap())

    def generate_piece_pixmap(self, piece, square_label):
        piece_names = {chess.PAWN: 'pawn', chess.KNIGHT: 'knight', chess.BISHOP: 'bishop',
                      chess.ROOK: 'rook', chess.QUEEN: 'queen', chess.KING: 'king'}
        pixmap = QPixmap(square_label.width(), square_label.height())
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        margin = min(pixmap.width(), pixmap.height()) * 0.1
        render_size = min(pixmap.width(), pixmap.height()) - 2 * margin
        if render_size <= 0:
            painter.end()
            return QPixmap()
        render_rect = QRectF((pixmap.width() - render_size) / 2, (pixmap.height() - render_size) / 2, render_size, render_size)
        self.svg_renderer.render(painter, piece_names[piece.piece_type], render_rect)
        painter.end()
        if piece.color == chess.WHITE:
            image = pixmap.toImage()
            for x in range(image.width()):
                for y in range(image.height()):
                    color = image.pixelColor(x, y)
                    if color.alpha() > 0:
                        color.setRgb(255 - color.red(), 255 - color.green(), 255 - color.blue(), color.alpha())
                        image.setPixelColor(x, y, color)
            return QPixmap.fromImage(image)
        return pixmap

    def start_new_game(self, player_color=chess.WHITE):
        self.engine.restart_game(player_color)
        self.refresh_board()
        self.clear_visualization()
        self.move_made_signal.emit("Board reset")

    def revert_move_action(self):
        if moves_undone := self.engine.revert_move():
            self.move_made_signal.emit(f"Move undone: {moves_undone}")

    def reapply_move_action(self):
        if moves_redone := self.engine.reapply_move():
            self.move_made_signal.emit(f"Move redone: {moves_redone}")

class ChessGameTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer_active = False
        self.time_limit = 600
        self.white_timer = self.black_timer = self.time_limit
        self.current_turn = chess.WHITE
        self.undone_evals = []
        self.move_evals = []
        self.engine = ChessEngine(player_color=chess.WHITE)
        self.engine.move_made_signal.connect(self.refresh_status)
        self.engine.game_over_signal.connect(self.handle_game_over)
        self.setup_ui_components()
        self.initialize_visualization()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.decrement_timer)

    def setup_ui_components(self):
        main_layout = QVBoxLayout()
        self.layout.addLayout(main_layout)
        self.tabs_widget = QTabWidget()
        main_layout.addWidget(self.tabs_widget)
        gameplay_tab = QWidget()
        gameplay_layout = QHBoxLayout()
        gameplay_tab.setLayout(gameplay_layout)
        self.tabs_widget.addTab(gameplay_tab, "Gameplay")
        left_section = QVBoxLayout()
        gameplay_layout.addLayout(left_section, 3)
        self.board_view = ChessBoardView(self.engine, self)
        self.board_view.move_made_signal.connect(self.refresh_status)
        self.board_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_section.addWidget(self.board_view)
        control_panel = QGroupBox("Game Controls")
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        left_section.addWidget(control_panel)
        new_game_btn = QPushButton("New Game")
        new_game_btn.clicked.connect(self.start_new_game)
        control_layout.addWidget(new_game_btn)
        _, self.ai_difficulty_dropdown, ai_layout = create_labeled_combo("AI Difficulty:", ["Easy", "Medium", "Hard"])
        self.ai_difficulty_dropdown.currentIndexChanged.connect(self.configure_ai_difficulty)
        control_layout.addLayout(ai_layout)
        _, self.time_control_spinbox, time_layout = create_labeled_spinbox("Time Control (min):", 1, 180, 10)
        self.time_control_spinbox.valueChanged.connect(self.configure_time_control)
        control_layout.addLayout(time_layout)
        undo_btn = QPushButton("Undo Move")
        undo_btn.clicked.connect(self.board_view.revert_move_action)
        control_layout.addWidget(undo_btn)
        self.status_label = QLabel()
        control_layout.addWidget(self.status_label)
        right_section = QVBoxLayout()
        gameplay_layout.addLayout(right_section, 1)
        right_section.addWidget(QLabel("Move History"))
        self.move_history_list = QListWidget()
        right_section.addWidget(self.move_history_list)
        timer_layout = QHBoxLayout()
        self.white_timer_label = QLabel("White: 10:00")
        self.black_timer_label = QLabel("Black: 10:00")
        timer_layout.addWidget(self.white_timer_label)
        timer_layout.addStretch()
        timer_layout.addWidget(self.black_timer_label)
        right_section.addLayout(timer_layout)
        self.visualization_widget = GameVisualization()
        viz_layout = QVBoxLayout()
        viz_layout.addWidget(self.visualization_widget)
        viz_tab = QWidget()
        viz_tab.setLayout(viz_layout)
        self.tabs_widget.addTab(viz_tab, "Visualizations")

    def initialize_visualization(self):
        pass

    def start_new_game(self):
        player_color = chess.WHITE if QMessageBox.question(self, "Choose Color", "Play as White?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes else chess.BLACK
        self.engine.restart_game(player_color)
        self.board_view.start_new_game(player_color)
        self.clear_visualization()
        self.white_timer = self.black_timer = self.time_limit
        self.refresh_timer_labels()
        self.timer_active = False
        self.timer.stop()
        self.undone_evals.clear()
        if self.engine.player_color == chess.BLACK:
            self.engine.make_ai_move()

    def configure_ai_difficulty(self):
        difficulty = self.ai_difficulty_dropdown.currentText()
        setattr(self.engine, 'ai_difficulty', difficulty)
        self.status_label.setText(f"AI difficulty set to {difficulty}.")

    def configure_time_control(self):
        self.time_limit = self.time_control_spinbox.value() * 60
        self.white_timer = self.black_timer = self.time_limit
        self.refresh_timer_labels()
        if not self.timer_active and not self.engine.is_game_over:
            self.timer.start(1000)
            self.timer_active = True
        self.status_label.setText(f"Time control set to {self.time_control_spinbox.value()} minutes per player.")

    def refresh_status(self, message):
        if message.startswith(("AI moved:", "Move made:", "Game loaded")):
            self.refresh_move_history()
            self.refresh_visualization(True)
            self.current_turn = self.engine.board.turn
            if not self.timer_active and not self.engine.is_game_over:
                self.timer.start(1000)
                self.timer_active = True
        elif message == "Board reset":
            self.refresh_move_history()
            self.clear_visualization()
            self.white_timer = self.black_timer = self.time_limit
            self.refresh_timer_labels()
            self.timer_active = False
            self.timer.stop()
            self.undone_evals.clear()
            if self.engine.player_color == chess.BLACK:
                self.engine.make_ai_move()
            message = "Board has been reset."
        elif message.startswith("Move undone"):
            self.refresh_move_history()
            moves = int(message.split(':')[1]) if ':' in message else 2
            self.refresh_visualization(False, moves)
            message = f"{moves} move(s) undone."
        elif message.startswith("Move redone"):
            self.refresh_move_history()
            moves = int(message.split(':')[1]) if ':' in message else 2
            self.refresh_visualization(True, moves)
            message = f"{moves} move(s) redone."
        elif message.startswith("Game over"):
            self.timer.stop()
            self.timer_active = False
        self.status_label.setText(message)

    def decrement_timer(self):
        if self.engine.board.is_game_over():
            self.timer.stop()
            self.timer_active = False
            return
        if self.current_turn == chess.WHITE:
            self.white_timer -= 1
        else:
            self.black_timer -= 1
        self.refresh_timer_labels()
        if self.white_timer <= 0 or self.black_timer <= 0:
            self.timer.stop()
            self.timer_active = False
            winner = "Black" if self.white_timer <= 0 else "White"
            QMessageBox.information(self, "Time Over", f"{winner} wins on time!")
            self.engine.is_game_over = True
            self.engine.board.clear_stack()
            self.engine.move_made_signal.emit(f"Game over: {winner} wins on time!")

    def refresh_timer_labels(self):
        self.white_timer_label.setText(f"White: {self.format_seconds(self.white_timer)}")
        self.black_timer_label.setText(f"Black: {self.format_seconds(self.black_timer)}")

    @staticmethod
    def format_seconds(seconds):
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.board_view.revert_move_action()
        elif event.key() == Qt.Key_Right:
            self.board_view.reapply_move_action()
        else:
            super().keyPressEvent(event)

    def handle_game_over(self, message):
        self.status_label.setText(message)

    def refresh_visualization(self, append=True, count=1):
        if append:
            self.undone_evals.clear()
            self.move_evals += [random.uniform(-1, 1) for _ in range(count)]
        else:
            self.undone_evals += [self.move_evals.pop() for _ in range(count) if self.move_evals]
        legal_moves = list(self.engine.board.legal_moves)
        if legal_moves:
            policy = {move.uci(): random.random() for move in legal_moves}
            total = sum(policy.values())
            policy = {move: value / total for move, value in policy.items()}
            mcts = {
                'simulations': random.randint(100, 500),
                'nodes_explored': random.randint(50, 200),
                'best_move': max(policy, key=policy.get)
            }
        else:
            policy, mcts = {}, {'simulations': 0, 'nodes_explored': 0, 'best_move': None}
        self.visualization_widget.update_value_evaluation(self.move_evals)
        self.visualization_widget.update_policy_output(policy)
        self.visualization_widget.update_mcts_statistics(mcts)

    def refresh_move_history(self):
        self.move_history_list.clear()
        board = chess.Board()
        for idx, move in enumerate(self.engine.board.move_stack):
            text = f"{(idx // 2) + 1}. {board.san(move)}" if idx % 2 == 0 else f"{board.san(move)}"
            self.move_history_list.addItem(text)
            board.push(move)
        self.move_history_list.scrollToBottom()

    def clear_visualization(self):
        self.move_evals = []
        self.visualization_widget.update_value_evaluation(self.move_evals)
        self.visualization_widget.update_policy_output({})
        self.visualization_widget.update_mcts_statistics({})