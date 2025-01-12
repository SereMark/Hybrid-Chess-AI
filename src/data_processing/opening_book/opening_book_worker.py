import json
import os
import time
from collections import defaultdict
import chess.pgn
from PyQt5.QtCore import pyqtSignal
from src.base.base_worker import BaseWorker
from src.utils.chess_utils import determine_outcome
from src.utils.common_utils import estimate_total_games, update_progress_time_left, wait_if_paused

class OpeningBookWorker(BaseWorker):
    positions_update = pyqtSignal(dict)

    def __init__(self, pgn_file_path: str, max_games: int, min_elo: int, max_opening_moves: int):
        super().__init__()
        self.pgn_file_path = pgn_file_path
        self.max_games = max_games
        self.min_elo = min_elo
        self.max_opening_moves = max_opening_moves
        self.positions = defaultdict(lambda: defaultdict(lambda: {"win": 0, "draw": 0, "loss": 0, "eco": "", "name": ""}))
        self.game_counter = 0
        self.start_time = None

    def run_task(self):
        self.start_time = time.time()
        try:
            total_estimated_games = estimate_total_games(file_paths=self.pgn_file_path, avg_game_size=5000, max_games=self.max_games, logger=self.logger)

            self.logger.info(f"Starting opening book generation from {self.pgn_file_path}.")

            with open(self.pgn_file_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
                while (self.game_counter < self.max_games and not self._is_stopped.is_set()):
                    wait_if_paused(self._is_paused)

                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    if not self._process_game(game):
                        continue

                    self.game_counter += 1

                    if self.game_counter % 1000 == 0:
                        update_progress_time_left(self.progress_update, self.time_left_update, self.start_time, self.game_counter, total_estimated_games)
                        self._emit_stats()

            # Final progress update after processing
            update_progress_time_left(self.progress_update, self.time_left_update, self.start_time, self.game_counter, total_estimated_games)

            self._emit_stats()

        except Exception as e:
            self.logger.error(f"Error during opening book generation: {str(e)}")
        finally:
            self._save_opening_book()

    def _process_game(self, game: chess.pgn.Game) -> bool:
        try:
            white_elo_str = game.headers.get("WhiteElo")
            black_elo_str = game.headers.get("BlackElo")

            if white_elo_str is None or black_elo_str is None:
                return False

            white_elo = int(white_elo_str)
            black_elo = int(black_elo_str)

            if white_elo < self.min_elo or black_elo < self.min_elo:
                return False

            result = game.headers.get("Result", "*")
            outcome = determine_outcome(result)
            if outcome is None:
                return False

            eco_code = game.headers.get("ECO", "")
            opening_name = game.headers.get("Opening", "")
            board = game.board()
            move_counter = 0

            for move in game.mainline_moves():
                if self._is_stopped.is_set():
                    break

                wait_if_paused(self._is_paused)

                if move_counter >= self.max_opening_moves:
                    break

                fen = board.fen()
                uci_move = move.uci()
                move_data = self.positions[fen][uci_move]

                # Update outcome statistics
                if outcome in {"win", "draw", "loss"}:
                    move_data[outcome] += 1

                # Update ECO code and opening name if not already set
                if not move_data["eco"]:
                    move_data["eco"] = eco_code
                if not move_data["name"]:
                    move_data["name"] = opening_name

                board.push(move)
                move_counter += 1

            return True

        except ValueError:
            # ELO values were not integers
            return False
        except Exception as e:
            self.logger.error(f"Error processing game {self.game_counter}: {str(e)}")
            return False

    def _emit_stats(self):
        if self.positions_update:
            stats = {"positions": dict(self.positions)}
            self.positions_update.emit(stats)

    def _save_opening_book(self):
        try:
            positions = {fen: {move: stats for move, stats in moves.items()} for fen, moves in self.positions.items()}
            book_file = os.path.abspath(os.path.join("data", "processed", "opening_book.json"))
            os.makedirs(os.path.dirname(book_file), exist_ok=True)
            with open(book_file, "w") as f:
                json.dump(positions, f, indent=4)
            self.logger.info(f"Opening book saved to {book_file}")
        except Exception as e:
            self.logger.error(f"Error saving opening book: {str(e)}")