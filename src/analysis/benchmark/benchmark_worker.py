from PyQt5.QtCore import pyqtSignal
import os
import time
import chess
import json
import torch
import numpy as np
from src.base.base_worker import BaseWorker
from src.utils.mcts import MCTS
from src.utils.chess_utils import get_game_result, policy_value_fn, get_total_moves
from src.utils.common_utils import wait_if_paused, update_progress_time_left
from src.models.model import ChessModel

class BenchmarkWorker(BaseWorker):
    benchmark_update = pyqtSignal(dict)

    def __init__(self, bot1_path, bot2_path, num_games, time_per_move, bot1_file_type, bot2_file_type, bot1_use_mcts, bot1_use_opening_book, bot2_use_mcts, bot2_use_opening_book):
        super().__init__()
        self.bot1_path = bot1_path
        self.bot2_path = bot2_path
        self.num_games = num_games
        self.time_per_move = time_per_move
        self.bot1_file_type = bot1_file_type
        self.bot2_file_type = bot2_file_type
        self.bot1_use_mcts = bot1_use_mcts
        self.bot1_use_opening_book = bot1_use_opening_book
        self.bot2_use_mcts = bot2_use_mcts
        self.bot2_use_opening_book = bot2_use_opening_book
        self.default_mcts_simulations = 100

        path = os.path.join("data", "processed", "opening_book.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.opening_book = json.load(f)
            except json.JSONDecodeError:
                pass
        self.opening_book = {}

    def run_task(self):
        if not os.path.exists(self.bot1_path) or not os.path.exists(self.bot2_path):
            return

        start_time = time.time()
        results = []

        for game_idx in range(self.num_games):
            if self._is_stopped.is_set():
                return

            wait_if_paused(self._is_paused)
            board = chess.Board()

            self.logger.info(f"Playing game {game_idx + 1} of {self.num_games}...")
            g_result, moves_count = self._play_single_game(board)

            winner = ("Bot1" if g_result > 0 else "Bot2" if g_result < 0 else "Draw")
            results.append({'game_index': game_idx + 1, 'winner': winner, 'moves': moves_count})

            self.logger.info(f"Game {game_idx + 1} of {self.num_games} completed in {moves_count} moves. Winner: {winner}")
            update_progress_time_left(self.progress_update, self.time_left_update, start_time, game_idx + 1, self.num_games)

        final_stats = {'bot1_wins': sum(r['winner'] == 'Bot1' for r in results), 'bot2_wins': sum(r['winner'] == 'Bot2' for r in results), 'draws': sum(r['winner'] == 'Draw' for r in results), 'total_games': self.num_games}
        self.benchmark_update.emit(final_stats)

    def _play_single_game(self, board):
        moves_count = 0
        while not board.is_game_over() and not self._is_stopped.is_set():
            wait_if_paused(self._is_paused)
            move = (self._determine_move(board, self.bot1_file_type, self.bot1_use_mcts, self.bot1_use_opening_book, self.bot1_path) if board.turn == chess.WHITE else self._determine_move(board, self.bot2_file_type, self.bot2_use_mcts, self.bot2_use_opening_book, self.bot2_path))
            board.push(move)
            moves_count += 1
        return get_game_result(board), moves_count

    def _determine_move(self, board, file_type, use_mcts, use_opening_book, bot_path):
        if use_mcts and use_opening_book:
            return self._combo_move(board, bot_path)
        elif use_mcts:
            return self._get_move_mcts(board, bot_path)
        elif use_opening_book:
            return self._get_opening_book_move(board)
        return self._get_move_engine(board) if file_type == "Engine" else self._get_move_pth(board)

    def _combo_move(self, board, bot_path):
        move = self._get_opening_book_move(board)
        return move if move != chess.Move.null() else self._get_move_mcts(board, bot_path)

    def _get_move_engine(self, board):
        try:
            # Path to the engine executable (e.g., Stockfish)
            engine_path = self.bot1_path if board.turn == chess.WHITE else self.bot2_path

            # Ensure the engine executable exists
            if not os.path.exists(engine_path):
                self.logger.error(f"Engine not found at {engine_path}")
                return chess.Move.null()

            # Use python-chess's engine interface
            import chess.engine

            with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
                # Set a time limit for the engine to find a move
                limit = chess.engine.Limit(time=self.time_per_move)

                # Analyze the position and fetch the best move
                result = engine.play(board, limit)
                return result.move if result.move else chess.Move.null()

        except Exception as e:
            self.logger.error(f"Error using chess engine: {e}")
            return chess.Move.null()

    def _get_move_pth(self, board):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ChessModel(get_total_moves())
            checkpoint = torch.load(self.bot1_path if board.turn == chess.WHITE else self.bot2_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                self.logger.warning("No legal moves available.")
                return chess.Move.null()

            # Get board features using policy_value_fn
            board_features = policy_value_fn(board, model, device)

            # Validate board_features format
            if not isinstance(board_features, tuple) or len(board_features) != 2:
                raise ValueError(f"Invalid board features format: {board_features}")

            move_scores, value_score = board_features

            # Ensure move_scores is a dictionary mapping legal moves to scores
            if not isinstance(move_scores, dict) or not all(isinstance(k, chess.Move) and isinstance(v, (int, float, np.float32)) for k, v in move_scores.items()):
                raise ValueError(f"Invalid move scores in board features: {move_scores}")

            # Filter move_scores to include only legal moves
            legal_move_scores = {move: move_scores[move] for move in legal_moves if move in move_scores}

            if not legal_move_scores:
                self.logger.warning("No valid scores for legal moves.")
                return chess.Move.null()

            # Get the best move
            best_move = max(legal_move_scores, key=legal_move_scores.get)
            return best_move if best_move else chess.Move.null()

        except Exception as e:
            self.logger.error(f"Error using .pth model: {e}")
            return chess.Move.null()

    def _get_opening_book_move(self, board):
        if not self.opening_book:
            return chess.Move.null()

        # Get the best move from the opening book based on the current board position
        fen = board.fen()
        moves_data = self.opening_book.get(fen, {})
        best_move, best_score = None, -1

        # Iterate over the moves and their statistics
        for algebraic_move, stats in moves_data.items():
            if not isinstance(stats, dict):
                self.logger.error(f"Invalid stats for move {algebraic_move}: {stats}")
                continue

            total = stats.get("win", 0) + stats.get("draw", 0) + stats.get("loss", 0)
            if total == 0:
                continue

            score = (stats.get("win", 0) + 0.5 * stats.get("draw", 0)) / total
            try:
                move_candidate = chess.Move.from_uci(algebraic_move)
                if move_candidate in board.legal_moves and score > best_score:
                    best_score, best_move = score, move_candidate
            except ValueError as e:
                self.logger.error(f"Error parsing move {algebraic_move}: {e}")
                continue

        # Return the best move if found
        return best_move if best_move else chess.Move.null()

    def _get_move_mcts(self, board, bot_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize and load the model
        model = ChessModel(get_total_moves())
        checkpoint = torch.load(bot_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Initialize MCTS with the policy_value_fn
        mcts = MCTS(lambda board_state: policy_value_fn(board_state, model, device), 1.4, self.default_mcts_simulations)
        mcts.set_root_node(board.copy())

        # Perform MCTS simulations within the allowed time per move
        start_time = time.time()
        while (time.time() - start_time) < self.time_per_move:
            if board.is_game_over() or self._is_stopped.is_set():
                break
            wait_if_paused(self._is_paused)
            mcts.simulate()

        # Get the best move based on MCTS results
        move_probs = mcts.get_move_probs(temperature=1e-3)
        return max(move_probs, key=move_probs.get) if move_probs else chess.Move.null()