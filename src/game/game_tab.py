import os, chess
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF, QSize
from PyQt5.QtGui import QFont, QPainter, QColor, QPixmap, QBrush
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSplitter, QDialog,
    QGridLayout, QPushButton, QSizePolicy, QMessageBox
)
from src.game.game_engine import GameEngine
from src.game.game_visualization import GameVisualization
from src.base.base_tab import BaseTab
from src.utils.chess_utils import initialize_move_mappings


class ChessBoardView(QWidget):
    move_made_signal = pyqtSignal(str)
    status_message = pyqtSignal(str)
    promotion_requested = pyqtSignal(int, int)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.selected_square = None
        self.highlighted_squares = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_svg_renderer()
        self._connect_signals()
        self.policy_output = {}
        self.square_probs = {}
        self.last_move = None

    def _setup_svg_renderer(self):
        svg_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'chess_pieces.svg')
        svg_path = os.path.abspath(svg_path)
        if not os.path.exists(svg_path):
            raise FileNotFoundError(f"SVG file not found at path: {svg_path}")
        self.svg_renderer = QSvgRenderer(svg_path)
        if not self.svg_renderer.isValid():
            raise ValueError("Invalid SVG file for chess pieces.")

    def _connect_signals(self):
        self.connect_engine_signals()

    def connect_engine_signals(self):
        self.engine.move_made_signal.connect(self.on_move_made)
        self.engine.game_over_signal.connect(self.on_game_over)
        self.engine.policy_output_signal.connect(self.update_policy_output)

    def update_policy_output(self, policy_output):
        self.policy_output = policy_output
        self.square_probs = {}
        for move_uci, prob in policy_output.items():
            move = chess.Move.from_uci(move_uci)
            to_sq = move.to_square
            self.square_probs[to_sq] = self.square_probs.get(to_sq, 0) + prob
        self.update()

    def on_move_made(self, msg):
        self.move_made_signal.emit(msg)
        self.selected_square = None
        self.highlighted_squares = []
        if "Move made:" in msg or "AI moved:" in msg:
            move_str = msg.split(":")[-1].strip()
            move = chess.Move.from_uci(move_str)
            self.last_move = move
        else:
            self.last_move = None
        self.update()

    def on_game_over(self, msg):
        self.status_message.emit(msg)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        size = min(self.width(), self.height()) / 8
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_board(painter, size)
        self._draw_pieces_and_highlights(painter, size)

    def _draw_board(self, painter, size):
        for r in range(8):
            for c in range(8):
                color = QColor('#F0D9B5') if (r + c) % 2 == 0 else QColor('#B58863')
                painter.fillRect(int(c * size), int(r * size), int(size), int(size), color)

    def _get_square_coordinates(self, square, size):
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        x = col * size
        y = row * size
        return x, y

    def _draw_pieces_and_highlights(self, painter, size):
        if self.last_move:
            from_sq = self.last_move.from_square
            to_sq = self.last_move.to_square
            x_from, y_from = self._get_square_coordinates(from_sq, size)
            x_to, y_to = self._get_square_coordinates(to_sq, size)
            color = QColor(255, 255, 0, 100)
            painter.fillRect(int(x_from), int(y_from), int(size), int(size), color)
            painter.fillRect(int(x_to), int(y_to), int(size), int(size), color)

        if self.square_probs:
            max_prob = max(self.square_probs.values())
            for square, prob in self.square_probs.items():
                intensity = prob / max_prob if max_prob > 0 else 0
                color = QColor(255, 0, 0, int(255 * intensity * 0.3))
                x, y = self._get_square_coordinates(square, size)
                painter.fillRect(int(x), int(y), int(size), int(size), color)

        for square in chess.SQUARES:
            piece = self.engine.board.piece_at(square)
            if piece:
                x, y = self._get_square_coordinates(square, size)
                self._draw_piece(painter, piece, x, y, size)

        for square in self.highlighted_squares:
            x, y = self._get_square_coordinates(square, size)
            self._draw_highlight(painter, x, y, size)

    def _draw_piece(self, painter, piece, x, y, size):
        pad = size * 0.15
        ps = size * 0.7
        pixmap = QPixmap(QSize(int(ps), int(ps)))
        pixmap.fill(Qt.transparent)
        svg_painter = QPainter(pixmap)
        piece_name = {
            chess.PAWN: 'pawn',
            chess.KNIGHT: 'knight',
            chess.BISHOP: 'bishop',
            chess.ROOK: 'rook',
            chess.QUEEN: 'queen',
            chess.KING: 'king'
        }
        color_name = 'white' if piece.color == chess.WHITE else 'black'
        piece_type_name = piece_name[piece.piece_type]
        element_id = f"{piece_type_name}_{color_name}"

        if not self.svg_renderer.elementExists(element_id):
            element_id = f"{color_name}_{piece_type_name}"
            if not self.svg_renderer.elementExists(element_id):
                element_id = f"{piece_type_name}_{color_name}".capitalize()

        self.svg_renderer.render(svg_painter, element_id, QRectF(0, 0, ps, ps))
        svg_painter.end()
        painter.drawPixmap(int(x + pad), int(y + pad), pixmap)

    def _draw_highlight(self, painter, x, y, size):
        r = size / 8
        painter.setBrush(QBrush(QColor(0, 255, 0, 180), Qt.SolidPattern))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            int(x + size / 2 - r),
            int(y + size / 2 - r),
            int(2 * r),
            int(2 * r)
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            size = min(self.width(), self.height()) / 8
            x = event.x()
            y = event.y()
            col = int(x // size)
            row = int(y // size)
            square = self._get_square(row, col)
            self.on_square_clicked(square)

    def on_square_clicked(self, sq):
        if self.engine.is_game_over:
            self.status_message.emit("The game is over.")
            return

        piece = self.engine.board.piece_at(sq)
        if self.selected_square is None:
            if piece and piece.color == self.engine.board.turn:
                self.selected_square = sq
                self.highlighted_squares = [
                    m.to_square for m in self.engine.board.legal_moves if m.from_square == sq
                ]
            else:
                self.status_message.emit("Select a piece to move.")
        else:
            if sq == self.selected_square:
                self.selected_square = None
                self.highlighted_squares = []
            elif piece and piece.color == self.engine.board.turn:
                self.selected_square = sq
                self.highlighted_squares = [
                    m.to_square for m in self.engine.board.legal_moves if m.from_square == sq
                ]
            else:
                moving_piece = self.engine.board.piece_at(self.selected_square)
                if moving_piece is None:
                    self.status_message.emit("No piece selected.")
                    self.selected_square = None
                    self.highlighted_squares = []
                    self.update()
                    return
                is_pawn_promotion_move = (
                    moving_piece.piece_type == chess.PAWN and
                    chess.square_rank(sq) in [0, 7]
                )
                if is_pawn_promotion_move:
                    self.promotion_requested.emit(self.selected_square, sq)
                    self.update()
                    return
                else:
                    move = chess.Move(self.selected_square, sq)
                    if move in self.engine.board.legal_moves:
                        if not self.engine.make_move(self.selected_square, sq):
                            self.status_message.emit("Invalid Move.")
                    else:
                        self.status_message.emit("Invalid Move.")
                    self.selected_square = None
                    self.highlighted_squares = []

        self.update()

    def _get_square(self, row, col):
        square = chess.square(col, 7 - row)
        return square

    def reset_view(self):
        self.selected_square = None
        self.highlighted_squares = []
        self.update()


class ChessGameTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.time_limit = 600
        self.white_timer = self.time_limit
        self.black_timer = self.time_limit
        self.game_active = False
        initialize_move_mappings()
        self.visual = GameVisualization()
        self._setup_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.decrement_timer)
        self.initialize_game(chess.WHITE, 'cnn', self.time_limit)

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        self.splitter = QSplitter(Qt.Horizontal)

        self.game_container = QWidget()
        game_layout = QVBoxLayout(self.game_container)

        self.board_container = QWidget()
        self.board_container.setLayout(QVBoxLayout())
        game_layout.addWidget(self.board_container)

        control_layout = QHBoxLayout()
        self.restart_button = QPushButton("Restart Game")
        self.restart_button.setToolTip("Restart the game with the same settings")
        self.restart_button.setEnabled(True)
        self.restart_button.clicked.connect(self.restart_game)
        control_layout.addWidget(self.restart_button)
        control_layout.addStretch()

        timers_layout = self._create_timer_layout()
        control_layout.addLayout(timers_layout)
        game_layout.addLayout(control_layout)

        self.status = QLabel("Game started!", alignment=Qt.AlignCenter)
        self.mode_label = QLabel("", alignment=Qt.AlignCenter)
        mode_font = QFont('Arial', 12, QFont.Bold)
        self.mode_label.setFont(mode_font)

        info_layout = QVBoxLayout()
        info_layout.addWidget(self.status)
        info_layout.addWidget(self.mode_label)
        game_layout.addLayout(info_layout)

        self.splitter.addWidget(self.game_container)
        self.splitter.addWidget(self.visual)
        self.splitter.setSizes([int(self.width() * 0.5), int(self.width() * 0.5)])
        main_layout.addWidget(self.splitter)

    def _create_timer_layout(self):
        font_bold = QFont('Arial', 14, QFont.Bold)
        self.white_label = QLabel(self._get_timer_text(chess.WHITE), alignment=Qt.AlignCenter)
        self.black_label = QLabel(self._get_timer_text(chess.BLACK), alignment=Qt.AlignCenter)
        self.white_label.setFont(font_bold)
        self.black_label.setFont(font_bold)

        timers_layout = QHBoxLayout()
        timers_layout.addWidget(self.white_label)
        timers_layout.addWidget(self.black_label)
        return timers_layout

    def initialize_game(self, player_color, opponent_type, time_limit):
        self.timer.stop()
        if hasattr(self, 'engine') and self.engine:
            self._disconnect_signals()
        self._clear_board_view()

        self.engine = GameEngine(player_color=player_color, opponent_type=opponent_type)
        self.board_view = ChessBoardView(self.engine, self)
        self.board_container.layout().addWidget(self.board_view)

        self.time_limit = time_limit
        self.white_timer = self.black_timer = self.time_limit

        self._connect_board_view_signals()
        self._connect_engine_signals()

        self.game_active = True
        self.restart_button.setEnabled(True)
        self.refresh_labels()
        self.visual.reset_visualizations()
        self.timer.start(1000)
        self.status.setText("Game started!")
        self.mode_label.setText(f"{'CNN AI' if self.engine.opponent_type == 'cnn' else 'Random'} AI Opponent")
        if self.engine.board.turn != self.engine.player_color:
            QTimer.singleShot(500, self.engine.make_ai_move)

    def _clear_board_view(self):
        if hasattr(self, 'board_view') and self.board_view:
            if self.board_view.parent():
                self.board_view.setParent(None)
            self.board_view.deleteLater()
            self.board_view = None

    def _connect_board_view_signals(self):
        self.board_view.move_made_signal.connect(self.refresh_status)
        self.board_view.status_message.connect(self.status.setText)
        self.board_view.promotion_requested.connect(self.handle_promotion)

    def _connect_engine_signals(self):
        self.engine.game_over_signal.connect(self.handle_game_over)
        self.engine.value_evaluation_signal.connect(self.visual.update_value_evaluation)
        self.engine.material_balance_signal.connect(self.visual.update_material_balance)
        self.engine.move_made_signal.connect(self.status.setText)
        self.engine.move_made_signal.connect(self.board_view.on_move_made)
        self.engine.policy_output_signal.connect(self.board_view.update_policy_output)

    def _disconnect_signals(self):
        try:
            self.engine.move_made_signal.disconnect(self.status.setText)
            self.engine.move_made_signal.disconnect(self.board_view.on_move_made)
            self.engine.game_over_signal.disconnect(self.handle_game_over)
            self.engine.value_evaluation_signal.disconnect(self.visual.update_value_evaluation)
            self.engine.material_balance_signal.disconnect(self.visual.update_material_balance)
            self.engine.policy_output_signal.disconnect(self.board_view.update_policy_output)
            self.board_view.move_made_signal.disconnect(self.refresh_status)
            self.board_view.status_message.disconnect(self.status.setText)
            self.board_view.promotion_requested.disconnect(self.handle_promotion)
        except Exception:
            pass

    def refresh_status(self, msg):
        if "Move made:" in msg or "AI moved:" in msg:
            self.refresh_labels()
        elif "Game over" in msg:
            self.timer.stop()
        if "Move made:" not in msg and "AI moved:" not in msg:
            self.status.setText(msg)

    def decrement_timer(self):
        if self.engine.is_game_over:
            self.timer.stop()
            return
        if self.engine.board.turn == chess.WHITE:
            self.white_timer = max(0, self.white_timer - 1)
            if self.white_timer == 0:
                self.engine.is_game_over = True
                self.engine.game_over_signal.emit("Game over: Black wins on time!")
                self.timer.stop()
                return
        else:
            self.black_timer = max(0, self.black_timer - 1)
            if self.black_timer == 0:
                self.engine.is_game_over = True
                self.engine.game_over_signal.emit("Game over: White wins on time!")
                self.timer.stop()
                return
        self.refresh_labels()

    def refresh_labels(self):
        self.white_label.setText(self._get_timer_text(chess.WHITE))
        self.black_label.setText(self._get_timer_text(chess.BLACK))

    def _get_timer_text(self, color):
        timer = self.white_timer if color == chess.WHITE else self.black_timer
        minutes, seconds = divmod(timer, 60)
        return f"{'White' if color == chess.WHITE else 'Black'}: {minutes:02d}:{seconds:02d}"

    def handle_game_over(self, msg):
        self.status.setText(msg)
        self.timer.stop()
        self.game_active = False
        self.restart_button.setEnabled(True)

    def handle_promotion(self, from_sq, to_sq):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Promotion Piece")
        layout = QGridLayout(dialog)
        selected_piece = {'piece': None}
        piece_types = ['Queen', 'Rook', 'Bishop', 'Knight']
        for i, piece in enumerate(piece_types):
            btn = QPushButton(piece)
            btn.clicked.connect(self.create_promotion_handler(piece, dialog, selected_piece))
            layout.addWidget(btn, 0, i)
        dialog.setLayout(layout)
        dialog.exec_()
        if selected_piece['piece']:
            if not self.engine.make_move(
                from_sq, to_sq, promotion=selected_piece['piece']
            ):
                self.status.setText("Invalid Move.")
            else:
                self.board_view.highlighted_squares = []
                self.board_view.selected_square = None
                self.board_view.update()
        else:
            self.status.setText("Promotion cancelled.")

    def create_promotion_handler(self, piece, dialog, selected_piece):
        def handler():
            selected_piece['piece'] = piece
            dialog.accept()
        return handler

    def restart_game(self):
        self.initialize_game(chess.WHITE, 'cnn', self.time_limit)

    def closeEvent(self, event):
        try:
            self.timer.stop()
            if hasattr(self, 'engine') and self.engine:
                self.engine.close()
            if self.visual:
                self.visual.close()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error during cleanup: {str(e)}")
        finally:
            super().closeEvent(event)