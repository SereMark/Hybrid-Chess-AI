import os
import chess
import chess.pgn
import torch
import numpy as np
from typing import Dict, Optional
from src.training.reinforcement.mcts import MCTS
from src.utils.chess_utils import get_total_moves, convert_board_to_tensor, get_move_mapping
from src.models.Transformer import TransformerChessModel

class Bot:
    def __init__(self, path: str, use_mcts: bool, use_opening_book: bool, logger):
        self.path = path
        self.use_mcts = use_mcts
        self.use_opening_book = use_opening_book
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model
        if not os.path.exists(self.path):
            self.model = None
            self.logger.warning(f"Model path does not exist: {self.path}")
        else:
            try:
                self.model = TransformerChessModel(get_total_moves()).to(self.device)
                self.model.load_state_dict(torch.load(self.path, map_location=self.device)["model_state_dict"])
                self.model.eval()
                self.logger.info(f"Model loaded successfully from {self.path}")
            except Exception as e:
                self.logger.error(f"Failed to load model from {self.path}: {e}")
                self.model = None

        # Initialize MCTS if required
        self.mcts: Optional[MCTS] = None
        if self.use_mcts and self.model:
            try:
                # Initialize MCTS
                self.mcts = MCTS(model=self.model, device=torch.device(self.device), c_puct=1.4, n_simulations=100)
                self.logger.info("MCTS initialized successfully.")
            except Exception as e:
                self.logger.error(f"Failed to initialize MCTS: {e}")
                self.mcts = None

    def _get_board_action_probs(self, board: chess.Board) -> Dict[chess.Move, float]:
        if not self.model:
            return {}

        board_tensor = convert_board_to_tensor(board)
        board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.model(board_tensor)
            policy_logits = policy_logits[0]  # Remove batch dimension

        # Convert logits to probabilities
        policy = torch.softmax(policy_logits, dim=0).cpu().numpy()

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        action_probs = {}
        total_prob = 0.0

        for move in legal_moves:
            idx = get_move_mapping().get_index_by_move(move)
            if idx is not None and idx < len(policy):
                prob = max(policy[idx], 1e-8)
                action_probs[move] = prob
                total_prob += prob
            else:
                action_probs[move] = 1e-8
                total_prob += 1e-8

        # Normalize probabilities
        if total_prob > 0:
            for mv in action_probs:
                action_probs[mv] /= total_prob
        else:
            # Fallback to uniform if total_prob is zero
            uniform_prob = 1.0 / len(legal_moves)
            for mv in action_probs:
                action_probs[mv] = uniform_prob

        return action_probs

    def get_move_pth(self, board: chess.Board) -> chess.Move:
        if not self.model:
            self.logger.warning("Model not loaded. Returning null move.")
            return chess.Move.null()

        try:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                self.logger.warning("No legal moves available.")
                return chess.Move.null()

            action_probs = self._get_board_action_probs(board)
            if not action_probs:
                self.logger.warning("No valid moves found from the NN inference.")
                return chess.Move.null()

            # Pick move with the highest probability
            best_move = max(action_probs, key=action_probs.get)
            return best_move

        except Exception as e:
            self.logger.error(f"Error determining move with .pth model: {e}")
            return chess.Move.null()

    def get_move_mcts(self, board: chess.Board) -> chess.Move:
        if not self.mcts:
            self.logger.warning("MCTS not initialized. Returning null move.")
            return chess.Move.null()

        self.mcts.set_root_node(board.copy())

        # Get MCTS move probabilities
        move_probs = self.mcts.get_move_probs(temperature=1e-3)
        if not move_probs:
            self.logger.warning("No move probabilities available from MCTS.")
            return chess.Move.null()

        # Add Dirichlet noise on the very first move of the game
        if board.fullmove_number == 1 and board.turn == chess.WHITE:
            moves = list(move_probs.keys())
            if moves:
                # Dirichlet noise parameters: alpha=0.3, blend=0.25
                noise = np.random.dirichlet([0.3] * len(moves))
                for i, mv in enumerate(moves):
                    move_probs[mv] = 0.75 * move_probs[mv] + 0.25 * noise[i]

                # Re-normalize probabilities
                total_prob = sum(move_probs.values())
                if total_prob > 1e-8:
                    for mv in move_probs:
                        move_probs[mv] /= total_prob

        # Select the move with the highest probability
        best_move = max(move_probs, key=move_probs.get)
        return best_move

    def get_opening_book_move(self, board: chess.Board, opening_book: Dict[str, Dict[str, Dict[str, int]]]) -> chess.Move:
        if not opening_book:
            return chess.Move.null()

        fen = board.fen()
        moves_data = opening_book.get(fen, {})
        best_move: Optional[chess.Move] = None
        best_score = -1.0

        # Iterate through possible moves from the opening book
        for uci_move, stats in moves_data.items():
            if not isinstance(stats, dict):
                self.logger.error(f"Invalid stats for move {uci_move}: {stats}")
                continue

            total = stats.get("win", 0) + stats.get("draw", 0) + stats.get("loss", 0)
            if total == 0:
                continue

            # Calculate score based on wins and draws
            score = (stats.get("win", 0) + 0.5 * stats.get("draw", 0)) / total
            try:
                move_candidate = chess.Move.from_uci(uci_move)
                if move_candidate in board.legal_moves and score > best_score:
                    best_score = score
                    best_move = move_candidate
            except ValueError as e:
                self.logger.error(f"Error parsing move {uci_move}: {e}")

        # Return the best move found or a null move if none
        return best_move if best_move else chess.Move.null()

    def get_move(self, board: chess.Board, opening_book: Dict[str, Dict[str, Dict[str, int]]]) -> chess.Move:
        # Use opening book and MCTS if both are enabled
        if self.use_mcts and self.use_opening_book:
            move = self.get_opening_book_move(board, opening_book)
            if move != chess.Move.null():
                return move
            return self.get_move_mcts(board)

        # Use only MCTS
        if self.use_mcts:
            return self.get_move_mcts(board)

        # Use only opening book
        if self.use_opening_book:
            return self.get_opening_book_move(board, opening_book)

        # Fallback to model-based move
        return self.get_move_pth(board)