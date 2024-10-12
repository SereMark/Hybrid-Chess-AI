import os, chess
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QLabel, QPushButton, QDialog, QGridLayout, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF
from PyQt5.QtGui import QPainter, QPixmap, QBrush, QColor, QFont
from PyQt5.QtSvg import QSvgRenderer
from src.gui.visualizations.game_visualization import GameVisualization
from scripts.chess_engine import ChessEngine

class ChessBoardView(QWidget):
    move_made_signal, status_message, promotion_requested = pyqtSignal(str), pyqtSignal(str), pyqtSignal(int, int)

    def __init__(self, engine: ChessEngine, parent=None):
        super().__init__(parent)
        self.engine, self.selected_square, self.is_orientation_flipped, self.highlighted_squares = engine, None, False, []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_svg_renderer()
        self._connect_signals()

    def _setup_svg_renderer(self):
        svg_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'chess_pieces.svg')
        self.svg_renderer = QSvgRenderer(svg_path)
        if not self.svg_renderer.isValid(): raise ValueError("Invalid SVG file for chess pieces.")

    def _connect_signals(self):
        self.engine.move_made_signal.connect(self.on_move_made)
        self.engine.game_over_signal.connect(self.on_game_over)

    def on_move_made(self, msg):
        self.move_made_signal.emit(msg)
        self.update()

    def on_game_over(self, msg):
        self.status_message.emit(msg)
        self.update()

    def paintEvent(self, event):
        painter, size = QPainter(self), min(self.width(), self.height()) / 8
        painter.setRenderHint(QPainter.Antialiasing)
        if self.is_orientation_flipped:
            painter.translate(self.width(), self.height())
            painter.rotate(180)
        self._draw_board(painter, size)
        self._draw_pieces_and_highlights(painter, size)

    def _draw_board(self, painter, size):
        for r in range(8):
            for c in range(8):
                painter.fillRect(int(c * size), int(r * size), int(size), int(size), QColor('#F0D9B5') if (r + c) % 2 == 0 else QColor('#B58863'))

    def _draw_pieces_and_highlights(self, painter, size):
        for r in range(8):
            for c in range(8):
                square = chess.square(c, 7 - r if not self.is_orientation_flipped else r)
                piece = self.engine.board.piece_at(square)
                if piece: self._draw_piece(painter, piece, c * size, r * size, size)
                if square in self.highlighted_squares: self._draw_highlight(painter, c * size, r * size, size)

    def _draw_piece(self, painter, piece, x, y, size):
        pad, ps = size * 0.15, size * 0.7
        pixmap = QPixmap(int(ps), int(ps))
        pixmap.fill(Qt.transparent)
        svg_painter = QPainter(pixmap)
        piece_name = {chess.PAWN: 'pawn', chess.KNIGHT: 'knight', chess.BISHOP: 'bishop', chess.ROOK: 'rook', chess.QUEEN: 'queen', chess.KING: 'king'}
        piece_group = f"{piece_name[piece.piece_type]}_{'white' if piece.color == chess.WHITE else 'black'}"
        self.svg_renderer.render(svg_painter, piece_group, QRectF(0, 0, ps, ps))
        svg_painter.end()
        painter.drawPixmap(int(x + pad), int(y + pad), pixmap)

    def _draw_highlight(self, painter, x, y, size):
        r = size / 8
        painter.setBrush(QBrush(QColor(255, 255, 0, 100)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(x + size / 2 - r), int(y + size / 2 - r), int(2 * r), int(2 * r))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            size = min(self.width(), self.height()) / 8
            square = self._get_square(int(event.y() // size), int(event.x() // size))
            self.on_square_clicked(square)

    def on_square_clicked(self, sq):
        if self.engine.is_game_over:
            self.status_message.emit("The game is over.")
            return
        piece = self.engine.board.piece_at(sq)
        if self.selected_square is None and piece and piece.color == self.engine.player_color:
            self.selected_square, self.highlighted_squares = sq, [m.to_square for m in self.engine.board.legal_moves if m.from_square == sq]
        else:
            if self.selected_square is not None:
                move = chess.Move(self.selected_square, sq)
                if move in self.engine.board.legal_moves and (chess.square_rank(sq) == 0 or chess.square_rank(sq) == 7):
                    self.promotion_requested.emit(self.selected_square, sq)
                elif not self.engine.make_move(self.selected_square, sq):
                    self.status_message.emit("Invalid Move.")
            else:
                self.status_message.emit("No piece selected to move.")
            self.selected_square, self.highlighted_squares = None, []
        self.update()

    def _get_square(self, row, col):
        return chess.square(col, 7 - row if not self.is_orientation_flipped else row)

    def reset_view(self):
        self.selected_square, self.highlighted_squares = None, []
        self.update()

class ChessGameTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.time_limit = self.white_timer = self.black_timer = 600
        self.engine = ChessEngine(player_color=chess.WHITE)
        self._setup_ui()
        self._connect_signals()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.decrement_timer)
        self.timer.start(1000)

    def _setup_ui(self):
        main_layout, splitter = QHBoxLayout(self), QSplitter(Qt.Horizontal)
        self.board_view, self.visual = ChessBoardView(self.engine, self), GameVisualization()
        timers_layout = self._create_timer_layout()
        self.status = QLabel("", alignment=Qt.AlignCenter)
        left_layout = QVBoxLayout()
        left_layout.addLayout(timers_layout)
        left_layout.addWidget(self.board_view)
        left_layout.addWidget(self.status)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.visual)
        main_layout.addWidget(splitter)

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

    def _connect_signals(self):
        self.board_view.move_made_signal.connect(self.refresh_status)
        self.board_view.status_message.connect(self.status.setText)
        self.board_view.promotion_requested.connect(self.handle_promotion)
        self.engine.game_over_signal.connect(self.handle_game_over)
        self.engine.value_evaluation_signal.connect(self.visual.update_value_evaluation)
        self.engine.policy_output_signal.connect(self.visual.update_policy_output)
        self.engine.mcts_statistics_signal.connect(self.visual.update_mcts_statistics)

    def refresh_status(self, msg):
        if "Move made:" in msg or "AI moved:" in msg: self.refresh_labels()
        elif "Game over" in msg: self.timer.stop()
        self.status.setText(msg)

    def decrement_timer(self):
        if self.engine.is_game_over:
            self.timer.stop()
            return
        current_timer = self.white_timer if self.engine.board.turn == chess.WHITE else self.black_timer
        current_timer = max(0, current_timer - 1)
        if current_timer == 0:
            self.engine.is_game_over = True
            winner = "Black" if self.engine.board.turn == chess.WHITE else "White"
            self.engine.move_made_signal.emit(f"Game over: {winner} wins on time!")
        if self.engine.board.turn == chess.WHITE:
            self.white_timer = current_timer
        else:
            self.black_timer = current_timer
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

    def handle_promotion(self, from_sq, to_sq):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Promotion Piece")
        layout = QGridLayout(dialog)
        selected_piece = {'piece': None}
        for i, piece in enumerate(['Queen', 'Rook', 'Bishop', 'Knight']):
            btn = QPushButton(piece)
            btn.clicked.connect(lambda _, p=piece: self._handle_promotion_selection(p, dialog, selected_piece))
            layout.addWidget(btn, 0, i)
        dialog.setLayout(layout)
        dialog.exec_()
        if selected_piece['piece'] and self.engine.make_move(from_sq, to_sq, promotion=selected_piece['piece']):
            self.board_view.highlighted_squares, self.board_view.selected_square = [], None
            self.board_view.update()
        else:
            self.status.setText("Invalid Move.")

    def _handle_promotion_selection(self, piece, dialog, selected_piece):
        selected_piece['piece'] = piece
        dialog.accept()