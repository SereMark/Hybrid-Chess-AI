import os, chess
from PyQt5.QtWidgets import QWidget, QSizePolicy, QHBoxLayout, QVBoxLayout, QSplitter, QLabel, QPushButton, QDialog, QGridLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QRectF
from PyQt5.QtGui import QPainter, QPixmap, QBrush, QColor, QFont
from PyQt5.QtSvg import QSvgRenderer
from src.gui.visualizations.game_visualization import GameVisualization
from scripts.chess_utils import ChessEngine, format_seconds

class PromotionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Promotion Piece")
        self.selected_piece = None
        layout = QGridLayout(self)
        pieces = ['Queen','Rook','Bishop','Knight']
        for i, p in enumerate(pieces):
            btn = QPushButton(p)
            btn.setFixedSize(100,50)
            btn.clicked.connect(lambda _, piece=p: self.select_piece(piece))
            layout.addWidget(btn,0,i)
        self.setLayout(layout)
    def select_piece(self, piece):
        self.selected_piece = piece
        self.accept()
    def get_selected_piece(self):
        return self.selected_piece

class ChessBoardView(QWidget):
    move_made_signal = pyqtSignal(str)
    status_message = pyqtSignal(str)
    promotion_requested = pyqtSignal(int, int)

    def __init__(self, engine: ChessEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.engine.move_made_signal.connect(self.on_move_made)
        self.engine.game_over_signal.connect(self.on_game_over)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.selected_square, self.is_orientation_flipped, self.highlighted_squares = None, False, []
        svg_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'chess_pieces.svg')
        if not os.path.exists(svg_path): raise FileNotFoundError(f"SVG file not found at {svg_path}")
        self.svg_renderer = QSvgRenderer(svg_path)
        if not self.svg_renderer.isValid(): raise ValueError(f"Invalid SVG file: {svg_path}")
    def on_move_made(self, msg):
        self.move_made_signal.emit(msg)
        self.update()
    def on_game_over(self, msg):
        self.move_made_signal.emit(msg)
        self.update()
    def sizeHint(self):
        return QSize(400,400)
    def minimumSizeHint(self):
        return QSize(200,200)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        size = min(self.width(), self.height()) /8
        if self.is_orientation_flipped:
            painter.translate(self.width(), self.height())
            painter.rotate(180)
        for r in range(8):
            for c in range(8):
                x, y = c*size, r*size
                color = QColor('#F0D9B5') if (r+c)%2==0 else QColor('#B58863')
                painter.fillRect(int(x), int(y), int(size), int(size), color)
                rank = 7 - r if not self.is_orientation_flipped else r
                sq = chess.square(c, rank)
                piece = self.engine.board.piece_at(sq)
                if piece: self.draw_piece(painter, piece, x, y, size)
                if sq in self.highlighted_squares: self.draw_highlight(painter, x, y, size)
    def draw_piece(self, painter, piece, x, y, size):
        pad, ps = size*0.15, size*0.7
        px, py = x + pad, y + pad
        names = {chess.PAWN:'pawn', chess.KNIGHT:'knight', chess.BISHOP:'bishop', chess.ROOK:'rook', chess.QUEEN:'queen', chess.KING:'king'}
        name = names.get(piece.piece_type, 'pawn')
        pix = QPixmap(int(ps), int(ps))
        pix.fill(Qt.transparent)
        svg_painter = QPainter(pix)
        svg_painter.setRenderHint(QPainter.Antialiasing)
        grp = f"{name}_white" if piece.color == chess.WHITE else f"{name}_black"
        self.svg_renderer.render(svg_painter, grp, QRectF(0,0,ps,ps))
        svg_painter.end()
        painter.drawPixmap(int(px), int(py), pix)
    def draw_highlight(self, painter, x, y, size):
        r = size/8
        painter.setBrush(QBrush(QColor(255,255,0,100)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(x + size/2 - r), int(y + size/2 - r), int(2*r), int(2*r))
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            size = min(self.width(), self.height()) /8
            c, r = int(event.x()//size), int(event.y()//size)
            sq = self.calculate_square_index(r, c)
            self.on_square_clicked(sq)
    def on_square_clicked(self, sq):
        if self.engine.is_game_over:
            self.status_message.emit("The game is over.")
            return
        piece = self.engine.board.piece_at(sq)
        if self.selected_square is None and piece and piece.color == self.engine.player_color:
            self.selected_square = sq
            self.highlight_possible_moves(sq)
        else:
            if self.selected_square is not None:
                move = chess.Move(self.selected_square, sq)
                if self.is_promotion_move(move):
                    self.promotion_requested.emit(self.selected_square, sq)
                    return
                if self.engine.make_move(self.selected_square, sq, promotion=None):
                    pass
                else:
                    self.status_message.emit("Invalid Move.")
            else:
                self.status_message.emit("No piece selected to move.")
            self.remove_highlights()
            self.selected_square = None
        self.update()
    def is_promotion_move(self, move):
        piece = self.engine.board.piece_at(self.selected_square)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(move.to_square)
            return (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0)
        return False
    def highlight_possible_moves(self, sq):
        self.remove_highlights()
        self.highlighted_squares = [m.to_square for m in self.engine.board.legal_moves if m.from_square == sq]
        self.update()
    def remove_highlights(self):
        self.highlighted_squares = []
        self.update()
    def calculate_square_index(self, r, c):
        rank = 7 - r if not self.is_orientation_flipped else r
        return chess.square(c, rank)
    def reset_view(self):
        self.selected_square, self.highlighted_squares = None, []
        self.update()

class ChessGameTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.time_limit = 600
        self.white_timer = self.black_timer = self.time_limit
        self.current_turn = chess.WHITE
        self.engine = ChessEngine(player_color=chess.WHITE)
        self.evaluations = []
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10,10,10,10)
        left_layout.setSpacing(10)
        timers_layout = QHBoxLayout()
        timers_layout.setSpacing(20)
        font_bold = QFont('Arial',14, QFont.Bold)
        self.white_label = QLabel(self.get_timer_text(chess.WHITE))
        self.white_label.setFont(font_bold)
        self.white_label.setAlignment(Qt.AlignCenter)
        self.white_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.black_label = QLabel(self.get_timer_text(chess.BLACK))
        self.black_label.setFont(font_bold)
        self.black_label.setAlignment(Qt.AlignCenter)
        self.black_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        timers_layout.addWidget(self.white_label)
        timers_layout.addWidget(self.black_label)
        self.board_view = ChessBoardView(self.engine, self)
        self.board_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.status = QLabel("")
        self.status.setFont(QFont('Arial',12))
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        left_layout.addLayout(timers_layout)
        left_layout.addWidget(self.board_view)
        left_layout.addWidget(self.status)
        self.visual = GameVisualization()
        self.visual.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(left)
        splitter.addWidget(self.visual)
        splitter.setStretchFactor(0,1)
        splitter.setStretchFactor(1,1)
        main_layout.addWidget(splitter)
        self.board_view.status_message.connect(self.update_status)
        self.board_view.promotion_requested.connect(self.handle_promotion)
        self.engine.move_made_signal.connect(self.refresh_status)
        self.engine.game_over_signal.connect(self.handle_game_over)
        self.engine.evaluation_signal.connect(self.update_visualizations)
        self.engine.policy_output_signal.connect(self.update_visualizations)
        self.engine.mcts_stats_signal.connect(self.update_visualizations)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.decrement_timer)
        self.timer.start(1000)
        self.timer_active = True

    def start_new_game(self, player_color=chess.WHITE):
        self.engine.restart_game(player_color)
        self.board_view.is_orientation_flipped = player_color == chess.BLACK
        self.board_view.reset_view()
        self.white_timer = self.black_timer = self.time_limit
        self.current_turn = self.engine.board.turn
        self.evaluations = []
        self.visual.update_value_evaluation(self.evaluations)
        self.refresh_labels()
        self.update_status("New game started.")
        if self.engine.player_color == chess.BLACK:
            self.engine.make_ai_move()

    def refresh_status(self, msg):
        if msg.startswith(("AI moved:", "Move made:", "Board reset")):
            self.current_turn = self.engine.board.turn
            self.refresh_labels()
            self.update_status(msg)
        if msg.startswith("Game over"):
            self.timer.stop()
            self.timer_active = False
            self.update_status(msg)

    def decrement_timer(self):
        if self.engine.board.is_game_over():
            self.timer.stop()
            return
        if self.current_turn == chess.WHITE:
            self.white_timer = max(0, self.white_timer -1)
            if self.white_timer ==0:
                self.timer.stop()
                self.engine.is_game_over = True
                self.engine.move_made_signal.emit("Game over: Black wins on time!")
        else:
            self.black_timer = max(0, self.black_timer -1)
            if self.black_timer ==0:
                self.timer.stop()
                self.engine.is_game_over = True
                self.engine.move_made_signal.emit("Game over: White wins on time!")
        self.refresh_labels()

    def refresh_labels(self):
        self.white_label.setText(f"White: {format_seconds(self.white_timer)}")
        self.black_label.setText(f"Black: {format_seconds(self.black_timer)}")

    def get_timer_text(self, color):
        return f"{'White' if color == chess.WHITE else 'Black'}: {format_seconds(self.white_timer if color == chess.WHITE else self.black_timer)}"

    def handle_game_over(self, msg):
        self.update_status(msg)
        self.timer.stop()

    def update_visualizations(self, data):
        if isinstance(data, int):
            self.evaluations.append(data)
            self.visual.update_value_evaluation(self.evaluations)
        elif isinstance(data, dict):
            if 'policy_output' in data:
                self.visual.update_policy_output(data['policy_output'])
            if 'mcts_stats' in data:
                self.visual.update_mcts_statistics(data['mcts_stats'])

    def update_status(self, msg):
        self.status.setText(msg)

    def handle_promotion(self, from_sq, to_sq):
        piece = self.get_promotion_piece()
        if piece is None:
            self.update_status("Promotion canceled.")
            self.board_view.remove_highlights()
            self.board_view.selected_square = None
            self.board_view.update()
            return
        promo = self.engine.promotion_str_to_type(piece) or chess.QUEEN
        if self.engine.make_move(from_sq, to_sq, promotion=promo):
            pass
        else:
            self.update_status("Invalid Move.")
        self.board_view.remove_highlights()
        self.board_view.selected_square = None
        self.board_view.update()

    def get_promotion_piece(self):
        dialog = PromotionDialog(self)
        dialog.exec_()
        return dialog.get_selected_piece()