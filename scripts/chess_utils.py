import random, chess
from PyQt5.QtCore import QObject, pyqtSignal

class ChessEngine(QObject):
    move_made_signal = pyqtSignal(str)
    game_over_signal = pyqtSignal(str)

    def __init__(self, player_color=chess.WHITE):
        super().__init__()
        self.board = chess.Board()
        self.player_color = player_color
        self.is_game_over = False
        self.undone_moves = []

    def restart_game(self, player_color=chess.WHITE):
        self.board.reset()
        self.player_color = player_color
        self.is_game_over = False
        self.undone_moves.clear()
        self.move_made_signal.emit("Board reset")

    def make_move(self, from_square, to_square):
        move = chess.Move(from_square, to_square)
        if self.board.piece_type_at(from_square) == chess.PAWN and (
            (self.board.turn == chess.WHITE and chess.square_rank(to_square) == 7) or
            (self.board.turn == chess.BLACK and chess.square_rank(to_square) == 0)
        ):
            move.promotion = chess.QUEEN
        if move in self.board.legal_moves:
            self.undone_moves.clear()
            self.board.push(move)
            self.move_made_signal.emit(f"Move made: {self.board.peek().uci()}")
            if self.board.is_game_over():
                self.is_game_over = True
                self.game_over_signal.emit(f"Game over: {self.board.result()}")
                return True
            if self.board.turn != self.player_color:
                self.make_ai_move()
            return True
        return False

    def make_ai_move(self):
        if self.board.is_game_over():
            self.is_game_over = True
            self.game_over_signal.emit(f"Game over: {self.board.result()}")
            return
        self.undone_moves.clear()
        move = self.choose_ai_move()
        if not move:
            self.is_game_over = True
            self.game_over_signal.emit("No legal moves available for AI.")
            return
        if move not in self.board.legal_moves:
            self.is_game_over = True
            self.game_over_signal.emit(f"AI attempted an illegal move: {move}")
            return
        self.board.push(move)
        self.move_made_signal.emit(f"AI moved: {move.uci()}")
        if self.board.is_game_over():
            self.is_game_over = True
            self.game_over_signal.emit(f"Game over: {self.board.result()}")

    def choose_ai_move(self):
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None

    def revert_move(self):
        moves_undone = 0
        if self.board.move_stack:
            move = self.board.pop()
            self.undone_moves.append(move)
            moves_undone += 1
            if self.board.turn != self.player_color and self.board.move_stack:
                move = self.board.pop()
                self.undone_moves.append(move)
                moves_undone += 1
            self.is_game_over = self.board.is_game_over()
            self.move_made_signal.emit(f"Move undone: {moves_undone}")
            return moves_undone
        return 0

    def reapply_move(self):
        moves_redone = 0
        while self.undone_moves and moves_redone < 2:
            move = self.undone_moves.pop()
            self.board.push(move)
            moves_redone += 1
        if moves_redone:
            self.is_game_over = self.board.is_game_over()
            self.move_made_signal.emit(f"Move redone: {moves_redone}")
        return moves_redone