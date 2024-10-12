from PyQt5.QtCore import QObject, pyqtSignal
import chess
import random

class ChessEngine(QObject):
    move_made_signal = pyqtSignal(str)
    game_over_signal = pyqtSignal(str)
    value_evaluation_signal = pyqtSignal(list)
    policy_output_signal = pyqtSignal(dict)
    mcts_statistics_signal = pyqtSignal(dict)

    def __init__(self, player_color=chess.WHITE):
        super().__init__()
        self.player_color, self.board = player_color, chess.Board()
        self.is_game_over, self.move_history = False, []

    def make_move(self, from_sq, to_sq, promotion=None):
        move = chess.Move(from_sq, to_sq, promotion={'Queen': chess.QUEEN, 'Rook': chess.ROOK, 'Bishop': chess.BISHOP, 'Knight': chess.KNIGHT}.get(promotion))
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            self.move_made_signal.emit(f"Move made: {move.uci()}")
            self._check_game_over()
            self._emit_ai_data()
            if self.board.turn != self.player_color:
                self.make_ai_move()
            return True
        self.move_made_signal.emit("Invalid Move.")
        return False

    def make_ai_move(self):
        if not self.is_game_over:
            move = random.choice(list(self.board.legal_moves))
            self.board.push(move)
            self.move_history.append(move)
            self.move_made_signal.emit(f"AI moved: {move.uci()}")
            self._check_game_over()
            self._emit_ai_data()
        else:
            self.game_over_signal.emit(self._get_game_over_message())

    def _check_game_over(self):
        if self.board.is_game_over():
            self.is_game_over = True
            self.game_over_signal.emit(self._get_game_over_message())

    def _emit_ai_data(self):
        self.value_evaluation_signal.emit([random.uniform(-1, 1) for _ in range(len(self.move_history))])
        self.policy_output_signal.emit({move.uci(): random.random() for move in self.board.legal_moves})
        self.mcts_statistics_signal.emit({'simulations': random.randint(1000, 10000), 'nodes_explored': random.randint(500, 5000), 'best_move': random.choice(list(self.board.legal_moves)).uci() if self.board.legal_moves else None})

    def _get_game_over_message(self):
        if self.board.is_checkmate(): return f"Game over: Checkmate! {'White' if self.board.turn == chess.BLACK else 'Black'} wins!"
        if self.board.is_stalemate(): return "Game over: Stalemate!"
        if self.board.is_insufficient_material(): return "Game over: Insufficient material!"
        return "Game over!"