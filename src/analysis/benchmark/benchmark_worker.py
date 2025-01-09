from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
import time, os, time, chess, json
from src.utils.mcts import MCTS
from src.utils.chess_utils import get_game_result
from src.utils.common_utils import wait_if_paused, update_progress_time_left

class BenchmarkWorker(BaseWorker):
    benchmark_update = pyqtSignal(dict)

    def __init__(
        self,
        bot1_path,
        bot2_path,
        num_games,
        time_per_move,
        bot1_file_type,
        bot2_file_type,
        bot1_use_mcts,
        bot1_use_opening_book,
        bot2_use_mcts,
        bot2_use_opening_book
    ):
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
        self.default_mcts_simulations = 200
        self.opening_book = {}
        path = os.path.join("data", "processed", "opening_book.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.opening_book = json.load(f)
            except:
                pass

    def run_task(self):
        if not os.path.exists(self.bot1_path):
            return
        if not os.path.exists(self.bot2_path):
            return
        start_time = time.time()
        results = []
        for game_idx in range(self.num_games):
            if self._is_stopped.is_set():
                return
            wait_if_paused(self._is_paused)
            board = chess.Board()
            g_result, moves_count = self._play_single_game(board)
            if g_result > 0:
                winner = "Bot1"
            elif g_result < 0:
                winner = "Bot2"
            else:
                winner = "Draw"
            results.append({'game_index': game_idx + 1, 'winner': winner, 'moves': moves_count})
            update_progress_time_left(
                self.progress_update,
                self.time_left_update,
                start_time,
                game_idx + 1,
                self.num_games
            )
        bot1_wins = sum(r['winner'] == 'Bot1' for r in results)
        bot2_wins = sum(r['winner'] == 'Bot2' for r in results)
        draws = sum(r['winner'] == 'Draw' for r in results)
        final_stats = {
            'bot1_wins': bot1_wins,
            'bot2_wins': bot2_wins,
            'draws': draws,
            'total_games': self.num_games
        }
        self.benchmark_update.emit(final_stats)

    def _play_single_game(self, board):
        moves_count = 0
        while not board.is_game_over() and not self._is_stopped.is_set():
            wait_if_paused(self._is_paused)
            if board.turn == chess.WHITE:
                move = self._determine_move_bot1(board)
            else:
                move = self._determine_move_bot2(board)
            board.push(move)
            moves_count += 1
        return get_game_result(board), moves_count

    def _determine_move_bot1(self, board):
        if self.bot1_file_type == "Engine":
            if self.bot1_use_mcts and self.bot1_use_opening_book:
                return self._combo_move_engine(board)
            elif self.bot1_use_mcts:
                return self._get_move_mcts(board)
            elif self.bot1_use_opening_book:
                return self._get_opening_book_move(board)
            else:
                return self._get_move_engine(board)
        else:
            if self.bot1_use_mcts and self.bot1_use_opening_book:
                return self._combo_move_pth(board)
            elif self.bot1_use_mcts:
                return self._get_move_mcts(board)
            elif self.bot1_use_opening_book:
                return self._get_opening_book_move(board)
            else:
                return self._get_move_pth(board)

    def _determine_move_bot2(self, board):
        if self.bot2_file_type == "Engine":
            if self.bot2_use_mcts and self.bot2_use_opening_book:
                return self._combo_move_engine(board)
            elif self.bot2_use_mcts:
                return self._get_move_mcts(board)
            elif self.bot2_use_opening_book:
                return self._get_opening_book_move(board)
            else:
                return self._get_move_engine(board)
        else:
            if self.bot2_use_mcts and self.bot2_use_opening_book:
                return self._combo_move_pth(board)
            elif self.bot2_use_mcts:
                return self._get_move_mcts(board)
            elif self.bot2_use_opening_book:
                return self._get_opening_book_move(board)
            else:
                return self._get_move_pth(board)

    def _combo_move_engine(self, board):
        move = self._get_opening_book_move(board)
        if move != chess.Move.null():
            return move
        return self._get_move_mcts(board)

    def _combo_move_pth(self, board):
        move = self._get_opening_book_move(board)
        if move != chess.Move.null():
            return move
        return self._get_move_mcts(board)

    def _get_move_engine(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        return legal_moves[0]

    def _get_move_pth(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        return legal_moves[-1]

    def _get_opening_book_move(self, board):
        if not self.opening_book:
            return chess.Move.null()
        fen = board.fen()
        if fen not in self.opening_book:
            return chess.Move.null()
        moves_data = self.opening_book[fen]
        best_move = None
        best_score = -1
        for algebraic_move, stats in moves_data.items():
            w = stats.get("win", 0)
            d = stats.get("draw", 0)
            l = stats.get("loss", 0)
            total = w + d + l
            if total == 0:
                continue
            score = (w + 0.5 * d) / total
            move_candidate = None
            try:
                move_candidate = chess.Move.from_uci(algebraic_move)
            except:
                continue
            if move_candidate in board.legal_moves and score > best_score:
                best_score = score
                best_move = move_candidate
        if best_move:
            return best_move
        return chess.Move.null()

    def _get_move_mcts(self, board):
        mcts = MCTS(self._policy_value_fn, 1.4, self.default_mcts_simulations)
        mcts.set_root_node(board.copy())
        start_time = time.time()
        while (time.time() - start_time) < self.time_per_move:
            if board.is_game_over() or self._is_stopped.is_set():
                break
            wait_if_paused(self._is_paused)
            mcts.simulate()
        move_probs = mcts.get_move_probs(temperature=1e-3)
        if not move_probs:
            return chess.Move.null()
        return max(move_probs, key=move_probs.get)

    def _policy_value_fn(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, 0.0
        p = 1.0 / float(len(legal_moves))
        return {m: p for m in legal_moves}, 0.0