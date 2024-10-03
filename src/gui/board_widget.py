import os
import chess
from PyQt5.QtWidgets import (QWidget, QGridLayout, QLabel, QVBoxLayout,
                             QPushButton, QHBoxLayout, QMessageBox)
from PyQt5.QtGui import QPixmap, QDrag
from PyQt5.QtCore import Qt, pyqtSignal, QMimeData

class SquareLabel(QLabel):
    square_clicked = pyqtSignal(int)

    def __init__(self, parent, square_index):
        super().__init__(parent)
        self.square_index = square_index
        self.setFixedSize(70, 70)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.piece = None

    def mousePressEvent(self, event):
        if self.pixmap():
            self.square_clicked.emit(self.square_index)
            if event.button() == Qt.LeftButton:
                self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        if not self.pixmap():
            return
        if (event.pos() - self.drag_start_position).manhattanLength() < 20:
            return
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(str(self.square_index))
        drag.setMimeData(mime_data)
        pixmap = self.pixmap()
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos())
        drag.exec_(Qt.MoveAction)

    def dragEnterEvent(self, event):
        event.accept()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        source_square = int(event.mimeData().text())
        target_square = self.square_index
        self.parent().handle_move(source_square, target_square)
        event.accept()

class ChessBoardWidget(QWidget):
    move_made = pyqtSignal(str)
    show_hint_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 600)
        self.layout = QVBoxLayout()
        self.board_layout = QGridLayout()
        self.layout.addLayout(self.board_layout)
        self.setLayout(self.layout)
        self.squares = []
        self.selected_square = None
        self.board_state = chess.Board()
        self.board_orientation = True # True for white at bottom, False for black at bottom
        self.init_board()
        self.add_control_buttons()

    def init_board(self):
        for i in reversed(range(self.board_layout.count())):
            self.board_layout.itemAt(i).widget().setParent(None)
        self.squares = []
        for row in range(8):
            for col in range(8):
                square_index = self.get_square_index(row, col)
                label = SquareLabel(self, square_index)
                label.square_clicked.connect(self.square_clicked)
                if self.board_orientation:
                    self.board_layout.addWidget(label, 7 - row, col)
                else:
                    self.board_layout.addWidget(label, row, 7 - col)
                self.squares.append(label)
        self.set_background()
        self.update_board()

    def get_square_index(self, row, col):
        if self.board_orientation:
            return row * 8 + col
        else:
            return (7 - row) * 8 + (7 - col)

    def flip_board(self):
        self.board_orientation = not self.board_orientation
        self.init_board()

    def set_background(self):
        for idx, square in enumerate(self.squares):
            row = idx // 8
            col = idx % 8
            if (row + col) % 2 == 0:
                square.setStyleSheet(
                    "background-color: #3c3f41; border: 1px solid #4b4b4b;")
            else:
                square.setStyleSheet(
                    "background-color: #2b2b2b; border: 1px solid #4b4b4b;")

    def square_clicked(self, square_index):
        if self.selected_square is None:
            piece = self.board_state.piece_at(square_index)
            if piece and piece.color == self.board_state.turn:
                self.selected_square = square_index
                self.highlight_legal_moves(square_index)
        else:
            if square_index != self.selected_square:
                self.handle_move(self.selected_square, square_index)
            self.clear_highlights()
            self.selected_square = None

    def handle_move(self, from_square, to_square):
        move = chess.Move(from_square, to_square)
        if move in self.board_state.legal_moves:
            self.board_state.push(move)
            self.update_board()
            self.move_made.emit(self.board_state.peek().uci())
            if self.board_state.turn == chess.BLACK:
                self.ai_move()
        else:
            QMessageBox.information(self, "Illegal Move", "That move is not legal.")

    def highlight_legal_moves(self, square_index):
        self.clear_highlights()
        legal_moves = [
            move.to_square for move in self.board_state.legal_moves
            if move.from_square == square_index]
        for idx, sq in enumerate(self.squares):
            if sq.square_index in legal_moves:
                sq.setStyleSheet("background-color: #5e81ac; border: 1px solid #4b4b4b;")

    def clear_highlights(self):
        self.set_background()

    def update_board(self):
        asset_path = os.path.join(os.path.dirname(__file__), 'assets') # TODO: Mention source of chess piece images
        piece_to_asset = {
            chess.PAWN:   {True: 'white_pawn.png', False: 'black_pawn.png'},
            chess.KNIGHT: {True: 'white_knight.png', False: 'black_knight.png'},
            chess.BISHOP: {True: 'white_bishop.png', False: 'black_bishop.png'},
            chess.ROOK:   {True: 'white_rook.png', False: 'black_rook.png'},
            chess.QUEEN:  {True: 'white_queen.png', False: 'black_queen.png'},
            chess.KING:   {True: 'white_king.png', False: 'black_king.png'}
        }

        for idx, label in enumerate(self.squares):
            square_index = label.square_index
            piece = self.board_state.piece_at(square_index)
            if piece:
                pixmap = QPixmap(os.path.join(
                    asset_path, piece_to_asset[piece.piece_type][piece.color]))
                label.setPixmap(pixmap)
            else:
                label.clear()

    def add_control_buttons(self):
        button_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo Move")
        self.help_button = QPushButton("Show Hint")
        button_layout.addWidget(self.undo_button)
        button_layout.addWidget(self.help_button)
        self.layout.addLayout(button_layout)
        self.undo_button.clicked.connect(self.undo_move)
        self.help_button.clicked.connect(self.show_hint_signal)
        button_style = """
        QPushButton {
            background-color: #3c3f41;
            color: #ffffff;
            border: 1px solid #4b4b4b;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #4b4b4b;
        }
        """
        self.undo_button.setStyleSheet(button_style)
        self.help_button.setStyleSheet(button_style)

    def reset_board(self):
        self.board_state.reset()
        self.update_board()
        self.move_made.emit("Board reset")

    def undo_move(self):
        if len(self.board_state.move_stack) >= 1:
            self.board_state.pop()
            if self.board_state.turn == chess.BLACK:
                self.board_state.pop()
        self.update_board()
        self.move_made.emit("Move undone")

    def show_hint(self):
        # TODO: Implement hint generation
        hint_move = list(self.board_state.legal_moves)[0]
        QMessageBox.information(self, "Hint", f"Consider the move: {hint_move.uci()}")

    def ai_move(self):
        if self.board_state.is_game_over():
            QMessageBox.information(self, "Game Over", "The game is over.")
            return
        move = self.get_ai_move()
        self.board_state.push(move)
        self.update_board()
        self.move_made.emit(f"AI moved: {move.uci()}")

    def get_ai_move(self):
        # TODO: Implement AI move generation
        return list(self.board_state.legal_moves)[0]

    def get_current_opening(self):
        # TODO: Implement opening book usage
        return "Mock Opening"

    def indicate_opening_book_usage(self):
        opening_name = self.get_current_opening()
        QMessageBox.information(self, "Opening Book", f"Current Opening: {opening_name}")