import os
import chess
import torch
import numpy as np
from typing import Dict, Optional
from src.training.reinforcement.mcts import MCTS
from src.utils.chess_utils import get_total_moves, convert_board_to_tensor, get_move_mapping
from src.models.transformer import TransformerChessModel

class Bot:
    def __init__(self, path: str, use_mcts: bool, use_opening_book: bool):
        self.use_mcts = use_mcts
        self.use_opening_book = use_opening_book
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TransformerChessModel(get_total_moves()).to(self.device)
        self._load_model_checkpoint(path)

        self.mcts = MCTS(self.model, self.device, c_puct=1.4, n_simulations=100) if self.use_mcts else None

        self.move_map = get_move_mapping()

    def _load_model_checkpoint(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model from {path}: {str(e)}")

    def get_move(self, board: chess.Board, opening_book: Dict[str, Dict[str, Dict[str, int]]]) -> chess.Move:
        try:
            if self.use_opening_book:
                book_move = self._choose_book_move(board, opening_book)
                if book_move is not None:
                    return book_move

            if self.use_mcts and self.mcts is not None:
                return self._choose_mcts_move(board)

            return self._choose_direct_policy_move(board)

        except Exception as e:
            print(f"[Bot] Error in get_move: {e}")
            return chess.Move.null()

    def _choose_book_move(self, board: chess.Board, opening_book: Dict[str, Dict[str, Dict[str, int]]]) -> Optional[chess.Move]:
        position_data = opening_book.get(board.fen(), {})
        if not position_data:
            return None

        best_move = None
        best_score = -1.0

        for uci, stats in position_data.items():
            total = stats.get("win", 0) + stats.get("draw", 0) + stats.get("loss", 0)
            if total > 0:
                score = (stats.get("win", 0) + 0.5 * stats.get("draw", 0)) / total
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves and score > best_score:
                    best_move = move
                    best_score = score

        return best_move

    def _choose_mcts_move(self, board: chess.Board) -> chess.Move:
        self.mcts.set_root_node(board.copy())
        probs = self.mcts.get_move_probs(temperature=1e-3)

        if board.fullmove_number == 1 and board.turn == chess.WHITE and len(probs) > 1:
            moves, move_probs = list(probs.keys()), np.array(list(probs.values()), dtype=np.float32)
            noise = np.random.dirichlet([0.3] * len(moves))
            move_probs = 0.75 * move_probs + 0.25 * noise
            move_probs /= move_probs.sum()
            probs = dict(zip(moves, move_probs))

        return max(probs, key=probs.get) if probs else chess.Move.null()

    def _choose_direct_policy_move(self, board: chess.Board) -> chess.Move:
        tensor = torch.from_numpy(convert_board_to_tensor(board)).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(tensor)
            policy = torch.softmax(logits[0], dim=0).cpu().numpy()

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()

        action_probs = {}
        for mv in legal_moves:
            idx = self.move_map.get_index_by_move(mv)
            prob = max(policy[idx], 1e-12)
            action_probs[mv] = prob

        total_prob = sum(action_probs.values())
        if total_prob > 0:
            for mv in action_probs:
                action_probs[mv] /= total_prob
        else:
            for mv in action_probs:
                action_probs[mv] = 1.0 / len(legal_moves)

        return max(action_probs, key=action_probs.get)