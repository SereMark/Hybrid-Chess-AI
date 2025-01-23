from PyQt5.QtCore import pyqtSignal
import os
import time
import chess
import chess.pgn
import json
from src.base.base_worker import BaseWorker
from src.analysis.benchmark.bot import Bot
from src.utils.common_utils import wait_if_paused, update_progress_time_left, get_game_result

class BenchmarkWorker(BaseWorker):
    benchmark_update = pyqtSignal(dict)

    def __init__(self, bot1_path: str, bot2_path: str, num_games: int, bot1_use_mcts: bool, bot1_use_opening_book: bool, bot2_use_mcts: bool, bot2_use_opening_book: bool):
        super().__init__()
        self.num_games = num_games

        # Load the opening book
        opening_book_path = os.path.join("data", "processed", "opening_book.json")
        if os.path.exists(opening_book_path):
            try:
                with open(opening_book_path, "r", encoding="utf-8") as f:
                    self.opening_book = json.load(f)
                self.logger.info("Successfully loaded the opening book.")
            except Exception as e:
                self.opening_book = {}
                self.logger.error(f"Error loading opening book from {opening_book_path}: {e}")
        else:
            self.opening_book = {}
            self.logger.warning(f"Opening book not found at {opening_book_path}. Proceeding without it.")

        # Initialize bots with their respective configurations
        self.bot1 = Bot(path=bot1_path, use_mcts=bot1_use_mcts, use_opening_book=bot1_use_opening_book, logger=self.logger)
        self.bot2 = Bot(path=bot2_path, use_mcts=bot2_use_mcts, use_opening_book=bot2_use_opening_book, logger=self.logger)

        # Ensure the benchmark games directory exists
        self.games_dir = os.path.join("data", "games", "benchmark")
        os.makedirs(self.games_dir, exist_ok=True)

    def run_task(self):
        # Validate that both bots are properly initialized
        bot1_valid = self.bot1.model or self.bot1.use_opening_book or self.bot1.use_mcts
        bot2_valid = self.bot2.model or self.bot2.use_opening_book or self.bot2.use_mcts

        if not bot1_valid:
            self.logger.error("Bot1 is not properly configured.")
            return
        if not bot2_valid:
            self.logger.error("Bot2 is not properly configured.")
            return

        start_time = time.time()
        results = []

        # Iterate through the number of games to be played
        for game_idx in range(1, self.num_games + 1):
            if self._is_stopped.is_set():
                return

            # Handle pause functionality
            wait_if_paused(self._is_paused)
            board = chess.Board()

            self.logger.info(f"Starting game {game_idx} of {self.num_games}...")

            # Play the game
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

            while not board.is_game_over() and not self._is_stopped.is_set():
                # Handle pause functionality
                wait_if_paused(self._is_paused)

                # Determine which bot's turn it is
                current_bot = self.bot1 if board.turn == chess.WHITE else self.bot2

                # Get the move from the current bot
                move = current_bot.get_move(board, self.opening_book)
                if move == chess.Move.null():
                    self.logger.warning(f"{'Bot1' if board.turn == chess.WHITE else 'Bot2'} returned a null move.")
                    break
                board.push(move)
                node = node.add_variation(move)
                moves_count += 1

            # Determine the game result
            result = get_game_result(board)
            if result > 0:
                game.headers["Result"] = "1-0"
            elif result < 0:
                game.headers["Result"] = "0-1"
            else:
                game.headers["Result"] = "1/2-1/2"

            game_result = result
            pgn_game = game

            # Save the PGN game to a file
            pgn_filename = os.path.join(self.games_dir, f"game_{game_idx}.pgn")
            try:
                with open(pgn_filename, "w", encoding="utf-8") as pgn_file:
                    pgn_file.write(str(pgn_game))
                self.logger.info(f"Saved game {game_idx} to {pgn_filename}")
            except Exception as e:
                self.logger.error(f"Failed to save PGN for game {game_idx}: {e}")

            # Determine the winner based on the game result
            if game_result > 0:
                winner = "Bot1"
            elif game_result < 0:
                winner = "Bot2"
            else:
                winner = "Draw"

            results.append({"game_index": game_idx, "winner": winner, "moves": moves_count})

            self.logger.info(f"Game {game_idx} completed in {moves_count} moves. Winner: {winner}")
            # Update progress and estimated time left
            update_progress_time_left(self.progress_update, self.time_left_update, start_time, game_idx, self.num_games)

        # Aggregate the results
        bot1_wins = sum(r["winner"] == "Bot1" for r in results)
        bot2_wins = sum(r["winner"] == "Bot2" for r in results)
        draws = sum(r["winner"] == "Draw" for r in results)
        final_stats = {
            "bot1_wins": bot1_wins,
            "bot2_wins": bot2_wins,
            "draws": draws,
            "total_games": self.num_games,
        }

        # Emit the final statistics
        self.benchmark_update.emit(final_stats)
        self.logger.info("Benchmarking completed.")