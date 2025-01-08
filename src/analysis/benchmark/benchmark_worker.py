from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
import time
import os
import chess
from src.utils.mcts import MCTS
from src.utils.chess_utils import get_game_result
from src.utils.common_utils import wait_if_paused

class BenchmarkWorker(BaseWorker):
    benchmark_update = pyqtSignal(dict)

    def __init__(self, model1_path: str, model2_path: str, num_games: int, time_per_move: float):
        super().__init__()
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.num_games = num_games
        self.time_per_move = time_per_move
        self.default_mcts_simulations = 200

    def run_task(self):
        if not os.path.exists(self.model1_path):
            self.logger.error(f"Model1 file not found at {self.model1_path}. Benchmark aborted.")
            return
        if not os.path.exists(self.model2_path):
            self.logger.error(f"Model2 file not found at {self.model2_path}. Benchmark aborted.")
            return
        self.logger.info("Starting benchmark worker.")
        self.logger.info(f"Model1 path: {self.model1_path}")
        self.logger.info(f"Model2 path: {self.model2_path}")
        self.logger.info(f"Running {self.num_games} games with {self.time_per_move}s per move.")
        start_time = time.time()
        results = []
        for game_idx in range(self.num_games):
            if self._is_stopped.is_set():
                self.logger.info("Benchmarking was stopped by user.")
                return
            wait_if_paused(self._is_paused)
            board = chess.Board()
            game_result, move_count = self._play_single_game(board)
            if game_result > 0:
                winner = "Model1"
            elif game_result < 0:
                winner = "Model2"
            else:
                winner = "Draw"
            results.append({
                'game_index': game_idx + 1,
                'winner': winner,
                'moves': move_count,
            })
            progress = int(((game_idx + 1) / self.num_games) * 100)
            self._update_progress(progress)
            elapsed = time.time() - start_time
            self._update_time_left(elapsed, game_idx + 1, self.num_games)
        model1_wins = sum(g['winner'] == 'Model1' for g in results)
        model2_wins = sum(g['winner'] == 'Model2' for g in results)
        draws = sum(g['winner'] == 'Draw' for g in results)
        self.logger.info("Benchmark run complete.")
        self.logger.info(f"Final results: Model1 wins: {model1_wins}, Model2 wins: {model2_wins}, Draws: {draws}")
        final_stats = {
            'model1_wins': model1_wins,
            'model2_wins': model2_wins,
            'draws': draws,
            'total_games': self.num_games,
        }
        self.benchmark_update.emit(final_stats)

    def _play_single_game(self, board: chess.Board):
        move_count = 0
        while not board.is_game_over() and not self._is_stopped.is_set():
            wait_if_paused(self._is_paused)
            if board.turn == chess.WHITE:
                move = self._get_move_model1(board)
            else:
                move = self._get_move_model2(board)
            board.push(move)
            move_count += 1
        result = get_game_result(board)
        return result, move_count

    def _get_move_model1(self, board: chess.Board):
        mcts_model1 = MCTS(
            policy_value_fn=self._policy_value_fn_model1,
            c_puct=1.4,
            n_simulations=self.default_mcts_simulations
        )
        mcts_model1.set_root_node(board.copy())
        start_time = time.time()
        while (time.time() - start_time) < self.time_per_move:
            if board.is_game_over() or self._is_stopped.is_set():
                break
            wait_if_paused(self._is_paused)
            mcts_model1.simulate()
        move_probs = mcts_model1.get_move_probs(temperature=1e-3)
        if not move_probs:
            return chess.Move.null()
        best_move = max(move_probs, key=move_probs.get)
        return best_move

    def _policy_value_fn_model1(self, board: chess.Board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, 0.0
        p = 1.0 / float(len(legal_moves))
        action_priors = {m: p for m in legal_moves}
        leaf_value = 0.0
        return action_priors, leaf_value

    def _get_move_model2(self, board: chess.Board):
        mcts_model2 = MCTS(
            policy_value_fn=self._policy_value_fn_model2,
            c_puct=1.4,
            n_simulations=self.default_mcts_simulations
        )
        mcts_model2.set_root_node(board.copy())
        start_time = time.time()
        while (time.time() - start_time) < self.time_per_move:
            if board.is_game_over() or self._is_stopped.is_set():
                break
            wait_if_paused(self._is_paused)
            mcts_model2.simulate()
        move_probs = mcts_model2.get_move_probs(temperature=1e-3)
        if not move_probs:
            return chess.Move.null()
        best_move = max(move_probs, key=move_probs.get)
        return best_move

    def _policy_value_fn_model2(self, board: chess.Board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, 0.0
        p = 1.0 / float(len(legal_moves))
        action_priors = {m: p for m in legal_moves}
        leaf_value = 0.0
        return action_priors, leaf_value

    def _update_progress(self, value: int):
        if self.progress_update:
            self.progress_update.emit(value)

    def _update_time_left(self, elapsed_time: float, steps_done: int, total_steps: int):
        if steps_done > 0 and self.time_left_update:
            estimated_total_time = (elapsed_time / steps_done) * total_steps
            time_left = max(0, estimated_total_time - elapsed_time)
            minutes, seconds = divmod(int(time_left), 60)
            hours, minutes = divmod(minutes, 60)
            if hours > 0:
                time_left_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_left_str = f"{minutes}m {seconds}s"
            else:
                time_left_str = f"{seconds}s"
            self.time_left_update.emit(time_left_str)