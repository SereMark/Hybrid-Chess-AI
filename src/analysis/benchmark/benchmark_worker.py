import os
import json
import time
import wandb
import chess
import chess.pgn
import numpy as np
from src.analysis.benchmark.bot import Bot

class BenchmarkWorker:
    def __init__(self, bot1_path, bot2_path, num_games, bot1_use_mcts,
                 bot1_use_opening_book, bot2_use_mcts, bot2_use_opening_book,
                 wandb_flag=False, progress_callback=None, status_callback=None,
                 switch_colors=False):
        self.num_games = num_games
        self.wandb_flag = wandb_flag
        self.progress_callback = progress_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x: None)
        self.switch_colors = switch_colors
        self.bot1 = Bot(bot1_path, bot1_use_mcts, bot1_use_opening_book)
        self.bot2 = Bot(bot2_path, bot2_use_mcts, bot2_use_opening_book)
        self.games_dir = os.path.join("data", "games", "benchmark")
        os.makedirs(self.games_dir, exist_ok=True)
        ob_path = os.path.join("data", "processed", "opening_book.json")
        if os.path.isfile(ob_path):
            try:
                with open(ob_path, "r", encoding="utf-8") as f:
                    self.opening_book = json.load(f)
            except Exception as e:
                self.status_callback(f"Failed to load opening book: {e}")
                self.opening_book = {}
        else:
            self.opening_book = {}
            self.status_callback("Opening book file not found.")

    def run(self):
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
        durations, move_counts = [], []
        bot1_wins, bot2_wins, draws = 0, 0, 0
        bot1_wins_over_time, bot2_wins_over_time, draws_over_time = [], [], []
        color_tracker = True
        for game_index in range(1, self.num_games + 1):
            self.status_callback(f"Playing game {game_index}/{self.num_games}")
            start_time = time.time()
            board = chess.Board()
            game = chess.pgn.Game()
            white_name = "Bot1" if color_tracker else "Bot2"
            black_name = "Bot2" if color_tracker else "Bot1"
            game.headers.update({
                "Event": "Bot Benchmark",
                "Site": "Local",
                "Date": time.strftime("%Y.%m.%d"),
                "Round": str(game_index),
                "White": white_name,
                "Black": black_name,
                "Result": "*"
            })
            node = game
            moves = 0
            while not board.is_game_over():
                current_bot = self.bot1 if (board.turn == chess.WHITE) == color_tracker else self.bot2
                move = current_bot.get_move(board, self.opening_book)
                if not move or move == chess.Move.null() or move not in board.legal_moves:
                    self.status_callback("Bot returned invalid/null move.")
                    break
                board.push(move)
                node = node.add_variation(move)
                moves += 1
            result = board.result()
            results[result] = results.get(result, 0) + 1
            game.headers["Result"] = result
            pgn_file = os.path.join(self.games_dir, f"game_{game_index}.pgn")
            with open(pgn_file, "w", encoding="utf-8") as pf:
                pf.write(str(game))
            end_time = time.time()
            duration = end_time - start_time
            durations.append(duration)
            move_counts.append(moves)
            if result == "1-0":
                if color_tracker:
                    bot1_wins += 1
                else:
                    bot2_wins += 1
            elif result == "0-1":
                if color_tracker:
                    bot2_wins += 1
                else:
                    bot1_wins += 1
            elif result == "1/2-1/2":
                draws += 1
            bot1_wins_over_time.append(bot1_wins)
            bot2_wins_over_time.append(bot2_wins)
            draws_over_time.append(draws)
            self.progress_callback(100 * game_index / self.num_games)
            self.status_callback(f"Game {game_index} finished result={result} in {duration:.2f}s")
            if self.switch_colors:
                color_tracker = not color_tracker
            if self.wandb_flag:
                wandb.log({
                    "game_index": game_index,
                    "game_result": result,
                    "game_duration_sec": duration,
                    "moves_made": moves,
                    "bot1_wins_so_far": bot1_wins,
                    "bot2_wins_so_far": bot2_wins,
                    "draws_so_far": draws
                })
        avg_duration = float(np.mean(durations)) if durations else 0.0
        avg_moves = float(np.mean(move_counts)) if move_counts else 0.0
        self.status_callback(f"All {self.num_games} games done. Results: Bot1={bot1_wins}, Bot2={bot2_wins}, Draws={draws}, Unfinished={results.get('*', 0)}")
        if self.wandb_flag:
            wandb.log({
                "total_games": self.num_games,
                "wins_bot1": bot1_wins,
                "wins_bot2": bot2_wins,
                "draws": draws,
                "unfinished": results.get("*", 0),
                "avg_game_duration": avg_duration,
                "avg_moves_per_game": avg_moves
            })
        return results