import os
import chess
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QLabel, QMessageBox, QDialog,
    QDialogButtonBox, QComboBox, QSizePolicy, QVBoxLayout
)
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QSize
from PyQt5.QtSvg import QSvgRenderer
from ..common.config import ASSETS_DIR

class SquareLabel(QLabel):
    square_clicked = pyqtSignal(int)

    def __init__(self, parent, square_index):
        super().__init__(parent)
        self.square_index = square_index
        self.setAlignment(Qt.AlignCenter)
        self.piece = None
        self.highlighted = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy = self.sizePolicy()
        size_policy.setHeightForWidth(True)
        self.setSizePolicy(size_policy)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.square_clicked.emit(self.square_index)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.highlighted:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            radius = min(self.width(), self.height()) // 6
            painter.setBrush(QBrush(QColor("#FFD700")))
            center = self.rect().center()
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, radius, radius)

class ChessBoardWidget(QWidget):
    move_made = pyqtSignal(str)
    show_hint_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy = self.sizePolicy()
        size_policy.setHeightForWidth(True)
        self.setSizePolicy(size_policy)

        self.board_layout = QGridLayout()
        self.board_layout.setSpacing(0)
        self.board_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.board_layout)

        self.squares = []
        self.selected_square = None
        self.selected_square_label = None
        self.board_state = chess.Board()
        self.board_orientation = True
        self.player_color = chess.WHITE
        self.game_over = False
        self.undone_moves = []

        self.svg_renderer = QSvgRenderer(os.path.join(ASSETS_DIR, 'chess_pieces.svg'))

        self.init_board()

    def init_board(self):
        for i in reversed(range(self.board_layout.count())):
            self.board_layout.itemAt(i).widget().setParent(None)
        self.squares = []

        for row in range(8):
            for col in range(8):
                square_index = self.get_square_index(row, col)
                label = SquareLabel(self, square_index)
                label.square_clicked.connect(self.square_clicked)
                self.board_layout.addWidget(label, row, col)
                self.squares.append(label)

        for i in range(8):
            self.board_layout.setRowStretch(i, 1)
            self.board_layout.setColumnStretch(i, 1)

        self.set_background()
        self.update_board()

    def get_square_index(self, row, col):
        if self.board_orientation:
            return (7 - row) * 8 + col
        else:
            return row * 8 + (7 - col)

    def flip_board(self):
        self.board_orientation = not self.board_orientation
        self.init_board()

    def set_background(self):
        for idx, square in enumerate(self.squares):
            square.setStyleSheet(self.get_square_style(square.square_index))

    def get_square_style(self, square_index):
        idx = square_index
        row = idx // 8
        col = idx % 8
        if (row + col) % 2 == 0:
            return "background-color: #1E1E1E;"
        else:
            return "background-color: #121212;"

    def square_clicked(self, square_index):
        if self.game_over:
            QMessageBox.information(self, "Game Over", "The game is over. Start a new game to continue.")
            return

        piece = self.board_state.piece_at(square_index)
        if self.selected_square is None:
            if piece and piece.color == self.board_state.turn and piece.color == self.player_color:
                self.selected_square = square_index
                self.highlight_legal_moves(square_index)
        else:
            if square_index != self.selected_square:
                self.handle_move(self.selected_square, square_index)
            self.clear_highlights()
            self.selected_square = None

    def handle_move(self, from_square, to_square):
        move = chess.Move(from_square, to_square)
        promotion = None

        if chess.PAWN == self.board_state.piece_type_at(from_square):
            if (self.board_state.turn == chess.WHITE and chess.square_rank(to_square) == 7) or \
               (self.board_state.turn == chess.BLACK and chess.square_rank(to_square) == 0):
                promotion = self.choose_promotion_piece()
                if promotion is None:
                    self.clear_highlights()
                    return
                move.promotion = promotion

        if move in self.board_state.legal_moves:
            self.undone_moves.clear()
            self.board_state.push(move)
            self.update_board()
            self.move_made.emit(f"Move made: {self.board_state.peek().uci()}")

            if self.board_state.is_game_over():
                result = self.board_state.result()
                QMessageBox.information(self, "Game Over", f"Game over: {result}")
                self.game_over = True
                return

            if self.board_state.turn != self.player_color:
                self.ai_move()

    def choose_promotion_piece(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Choose Promotion Piece")
        layout = QVBoxLayout()

        combo = QComboBox()
        combo.addItem("Queen", chess.QUEEN)
        combo.addItem("Rook", chess.ROOK)
        combo.addItem("Bishop", chess.BISHOP)
        combo.addItem("Knight", chess.KNIGHT)
        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            return combo.currentData()
        else:
            return None

    def highlight_legal_moves(self, square_index):
        self.clear_highlights()
        legal_moves = [
            move.to_square for move in self.board_state.legal_moves
            if move.from_square == square_index
        ]
        for sq in self.squares:
            if sq.square_index in legal_moves:
                sq.highlighted = True
                sq.update()

    def clear_highlights(self):
        for sq in self.squares:
            sq.highlighted = False
            sq.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_board()

    def update_board(self):
            if self.width() < 50 or self.height() < 50:
                return

            piece_names = {
                chess.PAWN: 'pawn',
                chess.KNIGHT: 'knight',
                chess.BISHOP: 'bishop',
                chess.ROOK: 'rook',
                chess.QUEEN: 'queen',
                chess.KING: 'king'
            }

            for idx, label in enumerate(self.squares):
                square_index = label.square_index
                piece = self.board_state.piece_at(square_index)
                if piece:
                    pixmap_width = label.width()
                    pixmap_height = label.height()
                    if pixmap_width <= 0 or pixmap_height <= 0:
                        label.clear()
                        continue

                    element_id = piece_names[piece.piece_type]
                    pixmap = QPixmap(pixmap_width, pixmap_height)
                    pixmap.fill(Qt.transparent)
                    painter = QPainter(pixmap)
                    painter.setRenderHint(QPainter.Antialiasing)
                    painter.setRenderHint(QPainter.SmoothPixmapTransform)

                    margin_ratio = 0.1
                    margin = min(pixmap_width, pixmap_height) * margin_ratio
                    render_size = min(pixmap_width, pixmap_height) - 2 * margin
                    if render_size <= 0:
                        label.clear()
                        painter.end()
                        continue

                    render_rect = QRectF(
                        (pixmap_width - render_size) / 2,
                        (pixmap_height - render_size) / 2,
                        render_size,
                        render_size
                    )

                    self.svg_renderer.render(painter, element_id, render_rect)
                    painter.end()

                    if piece.color == chess.WHITE:
                        image = pixmap.toImage()
                        image.invertPixels()
                        pixmap = QPixmap.fromImage(image)

                    label.setPixmap(pixmap)
                else:
                    label.clear()

    def reset_board(self, player_color=chess.WHITE):
        self.board_state.reset()
        self.player_color = player_color
        self.board_orientation = (player_color == chess.WHITE)
        self.init_board()
        self.update_board()
        self.move_made.emit("Board reset")
        self.game_over = False
        self.undone_moves.clear()

    def undo_move(self):
        moves_undone = 0
        if len(self.board_state.move_stack) >= 1:
            move = self.board_state.pop()
            self.undone_moves.append(move)
            moves_undone += 1
            if self.board_state.turn != self.player_color and len(self.board_state.move_stack) >= 1:
                move = self.board_state.pop()
                self.undone_moves.append(move)
                moves_undone += 1
            self.game_over = self.board_state.is_game_over()
        self.update_board()
        self.move_made.emit(f"Move undone:{moves_undone}")

    def redo_move(self):
        moves_redone = 0
        if self.undone_moves:
            move = self.undone_moves.pop()
            self.board_state.push(move)
            moves_redone += 1
            if self.board_state.turn != self.player_color and self.undone_moves:
                move = self.undone_moves.pop()
                self.board_state.push(move)
                moves_redone += 1
            self.game_over = self.board_state.is_game_over()
            self.update_board()
            self.move_made.emit(f"Move redone:{moves_redone}")

    def show_hint(self):
        if self.board_state.is_game_over():
            QMessageBox.information(self, "Game Over", "No more hints available, the game is over.")
            return

        legal_moves = list(self.board_state.legal_moves)
        if legal_moves:
            hint_move = legal_moves[0]
            QMessageBox.information(self, "Hint", f"Consider the move: {hint_move.uci()}")
        else:
            QMessageBox.information(self, "Hint", "No legal moves available.")

    def ai_move(self):
        if self.board_state.is_game_over():
            result = self.board_state.result()
            QMessageBox.information(self, "Game Over", f"Game over: {result}")
            self.game_over = True
            return
        self.undone_moves.clear()
        move = self.get_ai_move()
        self.board_state.push(move)
        self.update_board()
        self.move_made.emit(f"AI moved: {move.uci()}")
        if self.board_state.is_game_over():
            result = self.board_state.result()
            QMessageBox.information(self, "Game Over", f"Game over: {result}")
            self.game_over = True

    def get_ai_move(self):
        return list(self.board_state.legal_moves)[0]