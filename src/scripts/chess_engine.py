# src\scripts\chess_engine.py

from PyQt5.QtCore import QObject, pyqtSignal
import chess, random, torch, numpy as np
from src.scripts.data_pipeline import MOVE_MAPPING, INDEX_MAPPING, TOTAL_MOVES, convert_board_to_tensor


class ChessEngine(QObject):
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

        if self.opponent_type == 'cnn':
            self._load_model()

    def _load_model(self):
        from scripts.train import ChessModel
        import os

        self.MOVE_MAPPING = MOVE_MAPPING
        self.INDEX_MAPPING = INDEX_MAPPING

        model_path = os.path.join('models', 'saved_models', 'final_model.pth')
        if not os.path.exists(model_path):
            self.move_made_signal.emit("AI model not found. Reverting to random opponent.")
            self.opponent_type = 'random'
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessModel(num_moves=TOTAL_MOVES)
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model_state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(model_state_dict)
            self.model.to(device)
            self.model.eval()
            self.device = device
            self.move_made_signal.emit("AI model loaded successfully.")
        except Exception as e:
            self.move_made_signal.emit(f"Error loading AI model: {e}. Reverting to random opponent.")
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
            balance = self._compute_material_balance()
            self.material_balance_signal.emit([balance])
            self._compute_and_emit_ai_data()
            if self.board.turn != self.player_color:
                self.make_ai_move()
            return True
        else:
            self.move_made_signal.emit("Invalid Move.")
            return False

    def make_ai_move(self):
        if self.is_game_over:
            self.game_over_signal.emit(self._get_game_over_message())
            return

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

    def _select_move_with_cnn(self):
        if not hasattr(self, 'model'):
            return None

        input_tensor = convert_board_to_tensor(self.board)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_output, value_output = self.model(input_tensor)
            policy_probs = torch.softmax(policy_output, dim=1).cpu().numpy()[0]

        move_probs = {}
        for move in self.board.legal_moves:
            idx = self.INDEX_MAPPING.get(move)
            if idx is not None:
                move_probs[move.uci()] = policy_probs[idx]
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {k: v / total_prob for k, v in move_probs.items()}
        else:
            move_probs = {k: 1.0 / len(move_probs) for k in move_probs.items()}

        self.policy_output_signal.emit(move_probs)

        legal_moves = list(self.board.legal_moves)
        legal_indices = []
        legal_moves_dict = {}
        for move in legal_moves:
            idx = self.INDEX_MAPPING.get(move)
            if idx is not None:
                legal_indices.append(idx)
                legal_moves_dict[idx] = move
        if not legal_indices:
            return None

        legal_probs = policy_probs[legal_indices]
        best_idx = legal_indices[np.argmax(legal_probs)]
        best_move = legal_moves_dict[best_idx]
        return best_move

    def _compute_and_emit_ai_data(self):
        if not hasattr(self, 'model'):
            value_estimate = random.uniform(-1, 1)
            move_probs = {move.uci(): random.random() for move in self.board.legal_moves}
            total_prob = sum(move_probs.values())
            move_probs = {k: v / total_prob for k, v in move_probs.items()}
            self.value_evaluation_signal.emit([value_estimate])
            self.policy_output_signal.emit(move_probs)
            return

        input_tensor = convert_board_to_tensor(self.board)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_output, value_output = self.model(input_tensor)
            policy_probs = torch.softmax(policy_output, dim=1).cpu().numpy()[0]
            value_estimate = value_output.item()

        self.value_evaluation_signal.emit([value_estimate])

        move_probs = {}
        for move in self.board.legal_moves:
            idx = self.INDEX_MAPPING.get(move)
            if idx is not None:
                move_probs[move.uci()] = policy_probs[idx]
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {k: v / total_prob for k, v in move_probs.items()}
        else:
            move_probs = {k: 1.0 / len(move_probs) for k in move_probs.items()}

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
        white_material = sum(material[p.piece_type] for p in self.board.piece_map().values() if p.color == chess.WHITE)
        black_material = sum(material[p.piece_type] for p in self.board.piece_map().values() if p.color == chess.BLACK)
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
        return "Game over!"