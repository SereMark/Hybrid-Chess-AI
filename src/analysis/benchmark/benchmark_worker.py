import os
import time
import chess
import json
import chess.pgn
from typing import Dict
from src.analysis.benchmark.bot import Bot

class BenchmarkWorker:
    def __init__(self, bot1_path: str, bot2_path: str, num_games: int, bot1_use_mcts: bool, bot1_use_opening_book: bool, bot2_use_mcts: bool, bot2_use_opening_book: bool, progress_callback=None, status_callback=None):
        self.bot1_path = bot1_path
        self.bot2_path = bot2_path
        self.num_games = num_games
        self.bot1_use_mcts = bot1_use_mcts
        self.bot1_use_opening_book = bot1_use_opening_book
        self.bot2_use_mcts = bot2_use_mcts
        self.bot2_use_opening_book = bot2_use_opening_book
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        opening_book_path = os.path.join("data", "processed", "opening_book.json")
        if os.path.exists(opening_book_path):
            try:
                with open(opening_book_path, "r", encoding="utf-8") as f:
                    self.opening_book = json.load(f)
            except Exception as e:
                self.opening_book = {}
                if self.status_callback:
                    self.status_callback(f"Failed to load opening book: {e}")
        else:
            self.opening_book = {}
            if self.status_callback:
                self.status_callback("Opening book not found. Proceeding without it.")
        self.bot1 = Bot(path=self.bot1_path, use_mcts=self.bot1_use_mcts, use_opening_book=self.bot1_use_opening_book)
        self.bot2 = Bot(path=self.bot2_path, use_mcts=self.bot2_use_mcts, use_opening_book=self.bot2_use_opening_book)
        self.games_dir = os.path.join("data", "games", "benchmark")
        os.makedirs(self.games_dir, exist_ok=True)

    def run(self) -> Dict:
        if self.status_callback:
            self.status_callback("Starting benchmarking...")
        bot1_valid = self.bot1.is_initialized()
        bot2_valid = self.bot2.is_initialized()
        if not bot1_valid:
            if self.status_callback:
                self.status_callback("Bot1 is not properly initialized.")
            return {}
        if not bot2_valid:
            if self.status_callback:
                self.status_callback("Bot2 is not properly initialized.")
            return {}
        results = []
        for game_idx in range(1, self.num_games + 1):
            if self.status_callback:
                self.status_callback(f"Playing game {game_idx}/{self.num_games}")
            board = chess.Board()
            moves_count = 0
            game = chess.pgn.Game()
            game.headers["Event"] = "Bot Benchmarking"
            game.headers["Site"] = "Local"
            game.headers["Date"] = time.strftime("%Y.%m.%d")
            game.headers["Round"] = "-"
            game.headers["White"] = "Bot1"
            game.headers["Black"] = "Bot2"
            game.headers["Result"] = "*"
            node = game
            while not board.is_game_over():
                current_bot = self.bot1 if board.turn == chess.WHITE else self.bot2
                move = current_bot.get_move(board, self.opening_book)
                if move == chess.Move.null():
                    break
                board.push(move)
                node = node.add_variation(move)
                moves_count += 1
            result_map = {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0}
            result = result_map.get(board.result(), 0.0)
            game.headers["Result"] = "1-0" if result >0 else "0-1" if result <0 else "1/2-1/2"
            game_result = result
            pgn_game = game
            pgn_filename = os.path.join(self.games_dir, f"game_{game_idx}.pgn")
            with open(pgn_filename, "w", encoding="utf-8") as pgn_file:
                pgn_file.write(str(pgn_game))
            winner = "Bot1" if game_result >0 else "Bot2" if game_result <0 else "Draw"
            results.append({"game_index": game_idx, "winner": winner, "moves": moves_count})
            if self.progress_callback:
                progress = game_idx / self.num_games
                self.progress_callback(progress)
            if self.status_callback:
                self.status_callback(f"Completed game {game_idx}/{self.num_games}")
        if self.progress_callback:
            self.progress_callback(1.0)
        metrics = {
            "total_games_played": self.num_games,
            "results": results,
            "benchmark_pgn": self.games_dir
        }
        return metrics