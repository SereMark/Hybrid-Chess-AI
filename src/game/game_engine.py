import os, random, torch, chess
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from src.models.model import ChessModel
from src.utils import chess_utils
from src.utils.common_utils import log_message
from src.utils.mcts import MCTS


class GameEngine(QObject):
    move_made_signal = pyqtSignal(str)
    game_over_signal = pyqtSignal(str)
    value_evaluation_signal = pyqtSignal(list)
    policy_output_signal = pyqtSignal(dict)
    material_balance_signal = pyqtSignal(list)
    opening_info_signal = pyqtSignal(str, str)
    mcts_tree_signal = pyqtSignal(list, list)

    def __init__(self, player_color=chess.WHITE, opponent_type='random'):
        super().__init__()
        self.player_color = player_color
        self.opponent_type = opponent_type
        self.board = chess.Board()
        self.is_game_over = False
        self.move_history = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.opening_book = self.load_opening_book()

        if self.opponent_type == 'ai':
            self._load_model()
            if self.model:
                self.mcts = MCTS(self.policy_value_fn)
                self.mcts.set_root_node(self.board)
            else:
                log_message("Failed to load CNN model for AI. Switching to random opponent.")
                self.opponent_type = 'random'

    def _load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved_models', 'final_model.pth')
        model_path = os.path.abspath(model_path)
        self.model = self.load_model(model_path, self.device)
        if self.model is None:
            self.opponent_type = 'random'

    def policy_value_fn(self, board: chess.Board):
        if not self.model:
            move_probs = {move: 1.0 for move in board.legal_moves}
            total = len(move_probs)
            move_probs = {k: v / total for k, v in move_probs.items()}
            return move_probs, 0.0

        input_tensor = chess_utils.convert_board_to_tensor(board)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_output, value_output = self.model(input_tensor)
            policy_probs = torch.softmax(policy_output, dim=1).cpu().numpy()[0]
        move_probs = {}
        for move in board.legal_moves:
            idx = chess_utils.move_mapping.INDEX_MAPPING.get(move)
            if idx is not None and idx < len(policy_probs):
                move_probs[move] = policy_probs[idx]
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {k: v / total_prob for k, v in move_probs.items()}
        else:
            move_probs = {k: 1.0 / len(move_probs) for k in move_probs.keys()}
        value_estimate = value_output.item()
        return move_probs, value_estimate

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
            self.update_opening_info()
            self.move_made_signal.emit(f"Move made: {move.uci()}")
            self._check_game_over()
            balance = self.compute_material_balance(self.board)
            self.material_balance_signal.emit([balance])
            self._update_mcts_with_player_move(move)
            if not self.is_game_over and self.board.turn != self.player_color:
                QTimer.singleShot(500, self.make_ai_move)
            return True
        else:
            self.move_made_signal.emit("Invalid Move.")
            return False

    def _update_mcts_with_player_move(self, move):
        if self.opponent_type == 'ai' and hasattr(self, 'mcts'):
            self.mcts.update_with_move(move)

    def make_ai_move(self):
        if self.is_game_over:
            self.game_over_signal.emit(self.get_game_over_message(self.board))
            return

        if self.board.turn == self.player_color:
            return

        move = None
        move_source = None
        if self.opening_book:
            fen = ' '.join(self.board.fen().split(' ')[:4])
            if fen in self.opening_book:
                move_uci_counts = self.opening_book[fen]
                best_move_uci = max(move_uci_counts.items(), key=lambda item: item[1].get('win', 0) + item[1].get('draw', 0) + item[1].get('loss', 0))[0]
                move = chess.Move.from_uci(best_move_uci)
                move_source = 'opening_book'

        if move is None and self.opponent_type == 'ai':
            move = self._select_move_with_mcts()
            move_source = 'ai'

        if move is None and self.opponent_type == 'random':
            move = random.choice(list(self.board.legal_moves))
            move_source = 'random'

        if move is None:
            self.move_made_signal.emit("AI could not select a move. Reverting to random move.")
            move = random.choice(list(self.board.legal_moves))
            move_source = 'random'

        self.board.push(move)
        self.move_history.append(move)
        self.update_opening_info()
        if move_source == 'opening_book' or move_source == 'ai':
            self.move_made_signal.emit(f"AI moved: {move.uci()}")
        else:
            self.move_made_signal.emit(f"AI moved: {move.uci()}")
        self._check_game_over()
        balance = self.compute_material_balance(self.board)
        self.material_balance_signal.emit([balance])
        self._compute_and_emit_ai_data()
        self._update_mcts_with_ai_move(move)

    def _select_move_with_mcts(self):
        if not hasattr(self, 'mcts') or not self.mcts:
            return None
        self.mcts.set_root_node(self.board)
        for _ in range(self.mcts.n_simulations):
            self.mcts.simulate()
        move_probs = self.mcts.get_move_probs(temperature=1e-3)
        nodes, edges = self.mcts.get_tree_data(max_depth=3)
        self.mcts_tree_signal.emit(nodes, edges)
        if not move_probs:
            return None
        best_move = max(move_probs, key=move_probs.get)
        return chess.Move.from_uci(best_move)

    def _update_mcts_with_ai_move(self, move):
        if self.opponent_type == 'ai' and hasattr(self, 'mcts'):
            self.mcts.update_with_move(move)

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
        try:
            model = ChessModel(num_moves=chess_utils.get_total_moves())
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            log_message(f"Error loading model: {e}. Switching to random opponent.", log_callback)
            return None

    def compute_ai_move(self, board, model, device):
        input_tensor = chess_utils.convert_board_to_tensor(board)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_output, value_output = model(input_tensor)
            policy_probs = torch.softmax(policy_output, dim=1).cpu().numpy()[0]
        move_probs = {}
        for move in board.legal_moves:
            idx = chess_utils.move_mapping.INDEX_MAPPING.get(move)
            if idx is not None and idx < len(policy_probs):
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

    def load_opening_book(self):
        opening_book_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'opening_book.json')
        opening_book_file = os.path.abspath(opening_book_file)
        if os.path.exists(opening_book_file):
            import json
            with open(opening_book_file, 'r') as f:
                positions = json.load(f)
            return positions
        else:
            return None
    
    def update_opening_info(self):
        fen = ' '.join(self.board.fen().split(' ')[:4])
        if self.opening_book and fen in self.opening_book:
            moves = self.opening_book[fen]
            for move_data in moves.values():
                name = move_data.get('name', '')
                eco = move_data.get('eco', '')
                if name or eco:
                    self.opening_info_signal.emit(name, eco)
                    return
            self.opening_info_signal.emit('', '')
        else:
            self.opening_info_signal.emit('', '')
            
    def close(self):
        pass