import os, random, torch, chess
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from src.models.model import ChessModel
import src.utils.chess_utils as chess_utils
from src.game.opening_book import OpeningBook

class GameEngine(QObject):
    move_made_signal = pyqtSignal(str)
    game_over_signal = pyqtSignal(str)
    value_evaluation_signal = pyqtSignal(list)
    policy_output_signal = pyqtSignal(dict)
    material_balance_signal = pyqtSignal(list)
    opening_book_status_signal = pyqtSignal(str)
    opening_name_signal = pyqtSignal(str)

    def __init__(self, player_color=chess.WHITE, opponent_type='random'):
        super().__init__()
        self.player_color = player_color
        self.opponent_type = opponent_type
        self.board = chess.Board()
        self.is_game_over = False
        self.move_history = []
        self.opening_book = None
        self.current_opening_name = ""

        chess_utils.initialize_move_mappings()

        if self.opponent_type == 'cnn':
            self._load_model()

    def _load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved_models', 'final_model.pth')
        model_path = os.path.abspath(model_path)
        if os.path.exists(model_path):
            self.model = ChessModel()
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
        else:
            print("Model file not found. Switching to random opponent.")
            self.model = None
            self.opponent_type = 'random'

    def initialize_opening_book(self):
        self.opening_book = OpeningBook()
        data_file = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'opening_book', 'processed', 'opening_book.bin'
        )
        data_file = os.path.abspath(data_file)
        if os.path.exists(data_file):
            self.opening_book.start_loading(data_file)
            self.opening_book_status_signal.emit("Opening book loading started...")
        else:
            print("Opening book file not found. Proceeding without it.")

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
            balance = self._compute_material_balance()
            self.material_balance_signal.emit([balance])
            self._compute_and_emit_ai_data()

            if self.opening_book and self.opening_book.is_loaded():
                opening_name = self.opening_book.get_opening_name(self.board)
                self.current_opening_name = opening_name
                self.opening_name_signal.emit(self.current_opening_name)
            else:
                self.current_opening_name = ""
                self.opening_name_signal.emit(self.current_opening_name)

            if not self.is_game_over and self.board.turn != self.player_color:
                QTimer.singleShot(500, self.make_ai_move)
            return True
        else:
            self.move_made_signal.emit("Invalid Move.")
            return False

    def make_ai_move(self):
        if self.is_game_over:
            self.game_over_signal.emit(self._get_game_over_message())
            return

        opening_moves = []
        if self.opening_book and self.opening_book.is_loaded():
            opening_moves = self.opening_book.get_opening_moves(self.board)

        if opening_moves:
            move = random.choice(opening_moves)
            self.move_made_signal.emit(f"AI selected opening book move: {move.uci()}")
        else:
            if self.opponent_type == 'random':
                move = random.choice(list(self.board.legal_moves))
            elif self.opponent_type == 'cnn':
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
        balance = self._compute_material_balance()
        self.material_balance_signal.emit([balance])
        self._compute_and_emit_ai_data()

        if self.opening_book and self.opening_book.is_loaded():
            opening_name = self.opening_book.get_opening_name(self.board)
            self.current_opening_name = opening_name
            self.opening_name_signal.emit(self.current_opening_name)
        else:
            self.current_opening_name = ""
            self.opening_name_signal.emit(self.current_opening_name)

    def _select_move_with_cnn(self):
        if not hasattr(self, 'model') or self.model is None:
            return None

        input_tensor = chess_utils.convert_board_to_tensor(self.board)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_output, value_output = self.model(input_tensor)
            policy_probs = torch.softmax(policy_output, dim=1).cpu().numpy()[0]

        move_probs = {}
        for move in self.board.legal_moves:
            idx = chess_utils.INDEX_MAPPING.get(move)
            if idx is not None:
                move_probs[move.uci()] = policy_probs[idx]
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {k: v / total_prob for k, v in move_probs.items()}
        else:
            move_probs = {k: 1.0 / len(move_probs) for k in move_probs.keys()}

        self.policy_output_signal.emit(move_probs)

        best_move_uci = max(move_probs, key=move_probs.get)
        best_move = chess.Move.from_uci(best_move_uci)
        return best_move

    def _compute_and_emit_ai_data(self):
        if not hasattr(self, 'model') or self.model is None:
            value_estimate = random.uniform(-1, 1)
            move_probs = {move.uci(): random.random() for move in self.board.legal_moves}
            total_prob = sum(move_probs.values())
            move_probs = {k: v / total_prob for k, v in move_probs.items()}
            self.value_evaluation_signal.emit([value_estimate])
            self.policy_output_signal.emit(move_probs)
            return

        input_tensor = chess_utils.convert_board_to_tensor(self.board)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_output, value_output = self.model(input_tensor)
            policy_probs = torch.softmax(policy_output, dim=1).cpu().numpy()[0]
            value_estimate = value_output.item()

        self.value_evaluation_signal.emit([value_estimate])

        move_probs = {}
        for move in self.board.legal_moves:
            idx = chess_utils.INDEX_MAPPING.get(move)
            if idx is not None:
                move_probs[move.uci()] = policy_probs[idx]
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {k: v / total_prob for k, v in move_probs.items()}
        else:
            move_probs = {k: 1.0 / len(move_probs) for k in move_probs.keys()}

        self.policy_output_signal.emit(move_probs)

    def _compute_material_balance(self):
        material = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        white_material = sum(
            material[p.piece_type] for p in self.board.piece_map().values() if p.color == chess.WHITE
        )
        black_material = sum(
            material[p.piece_type] for p in self.board.piece_map().values() if p.color == chess.BLACK
        )
        balance = white_material - black_material
        return balance

    def _check_game_over(self):
        if self.board.is_game_over():
            self.is_game_over = True
            self.game_over_signal.emit(self._get_game_over_message())

    def _get_game_over_message(self):
        if self.board.is_checkmate():
            return f"Game over: Checkmate! {'White' if self.board.turn == chess.BLACK else 'Black'} wins!"
        if self.board.is_stalemate():
            return "Game over: Stalemate!"
        if self.board.is_insufficient_material():
            return "Game over: Insufficient material!"
        if self.board.can_claim_threefold_repetition():
            return "Game over: Draw by threefold repetition!"
        if self.board.can_claim_fifty_moves():
            return "Game over: Draw by fifty-move rule!"
        return "Game over!"