import random, chess
from PyQt5.QtCore import QObject, pyqtSignal

def format_seconds(seconds):
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"

class ChessEngine(QObject):
    move_made_signal = pyqtSignal(str)
    game_over_signal = pyqtSignal(str)
    evaluation_signal = pyqtSignal(int)
    policy_output_signal = pyqtSignal(dict)
    mcts_stats_signal = pyqtSignal(dict)

    def __init__(self, player_color=chess.WHITE):
        super().__init__()
        self.player_color = player_color
        self.board = chess.Board()
        self.is_game_over = False

    def restart_game(self, player_color=chess.WHITE):
        self.board.reset()
        self.player_color = player_color
        self.is_game_over = False
        self.move_made_signal.emit("Board reset")
        self.evaluate_position()
        self.update_policy_output()
        self.update_mcts_stats()

    def make_move(self, from_sq, to_sq, promotion=None):
        if from_sq is None or to_sq is None:
            self.move_made_signal.emit("Invalid move parameters.")
            return False
        if promotion:
            promotion = self.promotion_str_to_type(promotion)
        move = chess.Move(from_sq, to_sq, promotion=promotion)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_made_signal.emit(f"Move made: {move.uci()}")
            self.evaluate_position()
            self.update_policy_output()
            self.update_mcts_stats()
            if self.board.is_game_over():
                self.is_game_over = True
                self.game_over_signal.emit(self.get_game_over_message())
            elif self.board.turn != self.player_color:
                self.make_ai_move()
            return True
        else:
            self.move_made_signal.emit("Invalid Move.")
            return False

    def make_ai_move(self):
        if self.board.is_game_over():
            self.is_game_over = True
            self.game_over_signal.emit(self.get_game_over_message())
        else:
            move = self.choose_ai_move()
            if move:
                self.board.push(move)
                self.move_made_signal.emit(f"AI moved: {move.uci()}")
                self.evaluate_position()
                self.update_policy_output()
                self.update_mcts_stats()
                if self.board.is_game_over():
                    self.is_game_over = True
                    self.game_over_signal.emit(self.get_game_over_message())
            else:
                self.is_game_over = True
                self.game_over_signal.emit("Game over: No legal moves available for AI.")

    def choose_ai_move(self):
        legal = list(self.board.legal_moves)
        if not legal: return None
        move = random.choice(legal)
        if self.is_promotion_move(move) and move.promotion is None:
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        return move

    def get_game_over_message(self):
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            return f"Game over: Checkmate! {winner} wins!"
        if self.board.is_stalemate(): return "Game over: Stalemate!"
        if self.board.is_insufficient_material(): return "Game over: Insufficient material!"
        if self.board.is_seventyfive_moves(): return "Game over: Draw by 75-move rule!"
        if self.board.is_fivefold_repetition(): return "Game over: Draw by fivefold repetition!"
        return "Game over!"

    def evaluate_position(self):
        self.evaluation_signal.emit(self.evaluate_board())

    def update_policy_output(self):
        legal = list(self.board.legal_moves)
        policy = {m.uci(): random.uniform(0,1) for m in legal}
        total = sum(policy.values())
        if total > 0:
            policy = {m: p/total for m, p in policy.items()}
        self.policy_output_signal.emit({'policy_output': policy})

    def update_mcts_stats(self):
        legal = list(self.board.legal_moves)
        stats = {
            'simulations': random.randint(100,1000),
            'nodes_explored': random.randint(50,500),
            'best_move': random.choice([m.uci() for m in legal]) if legal else None
        }
        self.mcts_stats_signal.emit({'mcts_stats': stats})

    def evaluate_board(self):
        if self.board.is_checkmate():
            return -10000 if self.board.turn == self.player_color else 10000
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        values = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:20000}
        score = sum(values.get(p.piece_type,0) * (1 if p.color == self.player_color else -1) for p in self.board.piece_map().values())
        return score

    def promotion_str_to_type(self, promo):
        mapping = {'Queen':chess.QUEEN, 'Rook':chess.ROOK, 'Bishop':chess.BISHOP, 'Knight':chess.KNIGHT}
        return mapping.get(promo, chess.QUEEN)

    def is_promotion_move(self, move):
        p = self.board.piece_at(move.from_square)
        if p and p.piece_type == chess.PAWN:
            rank = chess.square_rank(move.to_square)
            return (p.color == chess.WHITE and rank == 7) or (p.color == chess.BLACK and rank == 0)
        return False