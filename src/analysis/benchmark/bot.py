import chess
import torch
import chess.pgn
import numpy as np
from typing import Dict, Optional
from src.training.reinforcement.mcts import MCTS
from src.utils.chess_utils import get_total_moves, convert_board_to_tensor, get_move_mapping
from src.models.transformer import TransformerChessModel

class Bot:
    def __init__(self, path: str, use_mcts: bool, use_opening_book: bool):
        self.path = path
        self.use_mcts = use_mcts
        self.use_opening_book = use_opening_book
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TransformerChessModel(get_total_moves()).to(self.device)
        self.model.load_state_dict(torch.load(self.path, map_location=self.device)["model_state_dict"])
        self.model.eval()
        self.mcts: Optional[MCTS] = None
        if self.use_mcts:
            self.mcts = MCTS(model=self.model, device=torch.device(self.device), c_puct=1.4, n_simulations=100)

    def _get_board_action_probs(self, board: chess.Board) -> Dict[chess.Move, float]:
        board_tensor = convert_board_to_tensor(board)
        board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.model(board_tensor)
            policy_logits = policy_logits[0]
        policy = torch.softmax(policy_logits, dim=0).cpu().numpy()
        legal_moves = list(board.legal_moves)
        action_probs = {}
        total_prob = 0.0
        move_mapping = get_move_mapping()
        for move in legal_moves:
            idx = move_mapping.get_index_by_move(move)
            if idx is not None and idx < len(policy):
                prob = max(policy[idx], 1e-8)
                action_probs[move] = prob
                total_prob += prob
            else:
                action_probs[move] = 1e-8
                total_prob += 1e-8
        if total_prob > 0:
            for mv in action_probs:
                action_probs[mv] /= total_prob
        else:
            uniform_prob = 1.0 / len(legal_moves)
            for mv in action_probs:
                action_probs[mv] = uniform_prob
        return action_probs

    def get_move_pth(self, board: chess.Board) -> chess.Move:
        try:
            action_probs = self._get_board_action_probs(board)
            best_move = max(action_probs, key=action_probs.get)
            return best_move
        except Exception:
            return chess.Move.null()

    def get_move_mcts(self, board: chess.Board) -> chess.Move:
        self.mcts.set_root_node(board.copy())
        move_probs = self.mcts.get_move_probs(temperature=1e-3)
        if board.fullmove_number == 1 and board.turn == chess.WHITE:
            moves = list(move_probs.keys())
            noise = np.random.dirichlet([0.3] * len(moves))
            for i, mv in enumerate(moves):
                move_probs[mv] = 0.75 * move_probs[mv] + 0.25 * noise[i]
            total_prob = sum(move_probs.values())
            if total_prob > 1e-8:
                for mv in move_probs:
                    move_probs[mv] /= total_prob
        best_move = max(move_probs, key=move_probs.get)
        return best_move

    def get_opening_book_move(self, board: chess.Board, opening_book: Dict[str, Dict[str, Dict[str, int]]]) -> chess.Move:
        fen = board.fen()
        moves_data = opening_book.get(fen, {})
        best_move: Optional[chess.Move] = None
        best_score = -1.0
        for uci_move, stats in moves_data.items():
            total = stats.get("win", 0) + stats.get("draw", 0) + stats.get("loss", 0)
            if total == 0:
                continue
            score = (stats.get("win", 0) + 0.5 * stats.get("draw", 0)) / total
            move_candidate = chess.Move.from_uci(uci_move)
            if move_candidate in board.legal_moves and score > best_score:
                best_score = score
                best_move = move_candidate
        return best_move if best_move else chess.Move.null()

    def get_move(self, board: chess.Board, opening_book: Dict[str, Dict[str, Dict[str, int]]]) -> chess.Move:
        if self.use_mcts and self.use_opening_book:
            move = self.get_opening_book_move(board, opening_book)
            if move != chess.Move.null():
                return move
            return self.get_move_mcts(board)
        if self.use_mcts:
            return self.get_move_mcts(board)
        if self.use_opening_book:
            return self.get_opening_book_move(board, opening_book)
        return self.get_move_pth(board)