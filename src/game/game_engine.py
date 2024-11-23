import os, random, torch, chess
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from src.models.model import ChessModel
from src.utils import chess_utils
from src.utils.common_utils import log_message


class GameEngine(QObject):
    move_made_signal = pyqtSignal(str)
    game_over_signal = pyqtSignal(str)
    value_evaluation_signal = pyqtSignal(list)
    policy_output_signal = pyqtSignal(dict)
    material_balance_signal = pyqtSignal(list)

    def __init__(self, player_color=chess.WHITE, opponent_type='random'):
        super().__init__()
        self.player_color = player_color
        self.opponent_type = opponent_type
        self.board = chess.Board()
        self.is_game_over = False
        self.move_history = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        if self.opponent_type == 'cnn':
            self._load_model()

    def _load_model(self):
        model_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'models', 'saved_models', 'final.model'
        )
        model_path = os.path.abspath(model_path)
        self.model = self.load_model(model_path, self.device)
        if self.model is None:
            self.opponent_type = 'random'

    def make_move(self, from_sq, to_sq, promotion=None):
        promotion_piece_type = None
        if promotion:
            promotion_piece_type = {
                'queen': chess.QUEEN,
                'rook': chess.ROOK,
                'bishop': chess.BISHOP,
                'knight': chess.KNIGHT
            }.get(promotion.lower())
        move = chess.Move(from_sq, to_sq, promotion=promotion_piece_type)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            self.move_made_signal.emit(f"Move made: {move.uci()}")
            self._check_game_over()
            balance = self.compute_material_balance(self.board)
            self.material_balance_signal.emit([balance])
            self._compute_and_emit_ai_data()
            if not self.is_game_over and self.board.turn != self.player_color:
                QTimer.singleShot(500, self.make_ai_move)
            return True
        else:
            self.move_made_signal.emit("Invalid Move.")
            return False

    def make_ai_move(self):
        if self.is_game_over:
            self.game_over_signal.emit(self.get_game_over_message(self.board))
            return

        if self.board.turn == self.player_color:
            return

        move = None
        if self.opponent_type == 'random':
            move = random.choice(list(self.board.legal_moves))
        elif self.opponent_type == 'cnn' and self.model:
            move = self._select_move_with_cnn()
            if move is None:
                self.move_made_signal.emit("AI could not select a move. Reverting to random move.")
                move = random.choice(list(self.board.legal_moves))
        else:
            move = random.choice(list(self.board.legal_moves))

        self.board.push(move)
        self.move_history.append(move)
        self.move_made_signal.emit(f"AI moved: {move.uci()}")
        self._check_game_over()
        balance = self.compute_material_balance(self.board)
        self.material_balance_signal.emit([balance])
        self._compute_and_emit_ai_data()

    def _select_move_with_cnn(self):
        if not self.model:
            return None
        move, move_probs, value_estimate = self.compute_ai_move(self.board, self.model, self.device)
        self.policy_output_signal.emit(move_probs)
        self.value_evaluation_signal.emit([value_estimate])
        return move

    def _compute_and_emit_ai_data(self):
        if not self.model:
            value_estimate = random.uniform(-1, 1)
            move_probs = {move.uci(): random.random() for move in self.board.legal_moves}
            total_prob = sum(move_probs.values())
            move_probs = {k: v / total_prob for k, v in move_probs.items()}
            self.value_evaluation_signal.emit([value_estimate])
            self.policy_output_signal.emit(move_probs)
            return
        move, move_probs, value_estimate = self.compute_ai_move(self.board, self.model, self.device)
        self.value_evaluation_signal.emit([value_estimate])
        self.policy_output_signal.emit(move_probs)

    def _check_game_over(self):
        if self.board.is_game_over():
            self.is_game_over = True
            self.game_over_signal.emit(self.get_game_over_message(self.board))

    def compute_material_balance(self, board):
        material = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        white_material = sum(
            material[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE
        )
        black_material = sum(
            material[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK
        )
        return white_material - black_material

    def get_game_over_message(self, board):
        if board.is_checkmate():
            return f"Game over: Checkmate! {'White' if board.turn == chess.BLACK else 'Black'} wins!"
        if board.is_stalemate():
            return "Game over: Stalemate!"
        if board.is_insufficient_material():
            return "Game over: Insufficient material!"
        if board.can_claim_threefold_repetition():
            return "Game over: Draw by threefold repetition!"
        if board.can_claim_fifty_moves():
            return "Game over: Draw by fifty-move rule!"
        return "Game over!"

    def load_model(self, model_path, device, log_callback=None):
        if not model_path or not os.path.exists(model_path):
            log_message("Model file not found. Switching to random opponent.", log_callback)
            return None
        model = ChessModel()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model

    def compute_ai_move(self, board, model, device):
        input_tensor = chess_utils.convert_board_to_tensor(board)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_output, value_output = model(input_tensor)
            policy_probs = torch.softmax(policy_output, dim=1).cpu().numpy()[0]
        move_probs = {}
        for move in board.legal_moves:
            idx = chess_utils.move_mapping.INDEX_MAPPING.get(move)
            if idx is not None:
                move_probs[move.uci()] = policy_probs[idx]
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {k: v / total_prob for k, v in move_probs.items()}
        else:
            move_probs = {k: 1.0 / len(move_probs) for k in move_probs.keys()}
        if move_probs:
            best_move_uci = max(move_probs, key=move_probs.get)
            best_move = chess.Move.from_uci(best_move_uci)
            value_estimate = value_output.item()
            return best_move, move_probs, value_estimate
        else:
            move = random.choice(list(board.legal_moves))
            return move, {}, value_output.item()

    def close(self):
        pass