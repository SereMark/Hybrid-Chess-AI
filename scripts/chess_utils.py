import random, chess
from PyQt5.QtCore import QObject, pyqtSignal

class ChessEngine(QObject):
    move_made_signal = pyqtSignal(str)
    game_over_signal = pyqtSignal(str)
    value_evaluation_signal = pyqtSignal(list)
    policy_output_signal = pyqtSignal(dict)
    mcts_statistics_signal = pyqtSignal(dict)

    def __init__(self, player_color=chess.WHITE):
        super().__init__()
        self.player_color = player_color
        self.board = chess.Board()
        self.is_game_over = False
        self.move_history = []

    def restart_game(self, player_color=chess.WHITE):
        self.board.reset()
        self.player_color = player_color
        self.is_game_over = False
        self.move_history = []
        self.move_made_signal.emit("Board reset")
        self._check_game_over()
        self._update_graphs()

    def make_move(self, from_sq, to_sq, promotion=None):
        if not from_sq or not to_sq:
            self.move_made_signal.emit("Invalid move parameters.")
            return False

        move = chess.Move(from_sq, to_sq, promotion=self._get_promotion_code(promotion))
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            self.move_made_signal.emit(f"Move made: {move.uci()}")
            self._check_game_over()
            self._update_graphs()
            if self.board.turn != self.player_color:
                self.make_ai_move()
            return True
        else:
            self.move_made_signal.emit("Invalid Move.")
            return False

    def _get_promotion_code(self, promotion):
        return {'Queen': chess.QUEEN, 'Rook': chess.ROOK, 'Bishop': chess.BISHOP, 'Knight': chess.KNIGHT}.get(promotion)

    def make_ai_move(self):
        if self.is_game_over:
            self.game_over_signal.emit(self._get_game_over_message())
        else:
            move = random.choice(list(self.board.legal_moves))
            self.board.push(move)
            self.move_history.append(move)
            self.move_made_signal.emit(f"AI moved: {move.uci()}")
            self._check_game_over()
            self._update_graphs()

    def _update_graphs(self):
        self.value_evaluation_signal.emit([random.uniform(-1, 1) for _ in range(len(self.move_history))])
        self.policy_output_signal.emit({move.uci(): random.random() for move in self.board.legal_moves})
        self.mcts_statistics_signal.emit({
            'simulations': random.randint(1000, 10000),
            'nodes_explored': random.randint(500, 5000),
            'best_move': random.choice(list(self.board.legal_moves)).uci() if self.board.legal_moves else None
        })

    def _check_game_over(self):
        if self.board.is_game_over():
            self.is_game_over = True
            self.game_over_signal.emit(self._get_game_over_message())

    def _get_game_over_message(self):
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            return f"Game over: Checkmate! {winner} wins!"
        if self.board.is_stalemate():
            return "Game over: Stalemate!"
        if self.board.is_insufficient_material():
            return "Game over: Insufficient material!"
        return "Game over!"

    def is_promotion_move(self, move):
        piece = self.board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE and chess.square_rank(move.to_square) == 7:
                return True
            if piece.color == chess.BLACK and chess.square_rank(move.to_square) == 0:
                return True
        return False