import os
import chess
import torch
import numpy as np
from src.training.reinforcement.mcts import MCTS
from src.utils.chess_utils import get_total_moves, convert_board_to_transformer_input, get_move_mapping
from src.models.transformer import TransformerCNNChessModel

class Bot:
    def __init__(self, path, use_mcts, use_opening_book):
        self.use_mcts = use_mcts
        self.use_opening_book = use_opening_book
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerCNNChessModel(get_total_moves()).to(self.device)
        self._load_model_checkpoint(path)
        self.mcts = MCTS(self.model, self.device, c_puct=1.4, n_simulations=100) if self.use_mcts else None
        self.move_map = get_move_mapping()

    def _load_model_checkpoint(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def get_move(self, board, opening_book):
        try:
            if self.use_opening_book:
                book_move = self._choose_book_move(board, opening_book)
                if book_move is not None:
                    return book_move
            if self.use_mcts and self.mcts:
                return self._choose_mcts_move(board)
            return self._choose_direct_policy_move(board)
        except:
            return chess.Move.null()

    def _choose_book_move(self, board, opening_book):
        data = opening_book.get(board.fen())
        if not data:
            return None
        best_move = None
        best_score = -1
        for uci, stats in data.items():
            total = stats.get("win", 0) + stats.get("draw", 0) + stats.get("loss", 0)
            if total > 0:
                score = (stats.get("win", 0) + 0.5 * stats.get("draw", 0)) / total
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves and score > best_score:
                    best_move = move
                    best_score = score
        return best_move

    def _choose_mcts_move(self, board):
        self.mcts.set_root_node(board.copy())
        probs = self.mcts.get_move_probs(temperature=1e-3)
        if board.fullmove_number == 1 and board.turn == chess.WHITE and len(probs) > 1:
            moves_list = list(probs.keys())
            arr = np.array(list(probs.values()), dtype=np.float32)
            arr /= arr.sum()
            probs = dict(zip(moves_list, arr))
        return max(probs, key=probs.get) if probs else chess.Move.null()

    def _choose_direct_policy_move(self, board):
        x = torch.from_numpy(convert_board_to_transformer_input(board)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
            policy = torch.softmax(logits[0], dim=0).cpu().numpy()
        legal = list(board.legal_moves)
        if not legal:
            return chess.Move.null()
        move_probs = {}
        for move in legal:
            idx = self.move_map.get_index_by_move(move)
            prob = policy[idx] if idx is not None else 1e-12
            move_probs[move] = max(prob, 1e-12)
        total = sum(move_probs.values())
        if total > 0:
            for move in move_probs:
                move_probs[move] /= total
        else:
            for move in move_probs:
                move_probs[move] = 1 / len(legal)
        return max(move_probs, key=move_probs.get)