import os, json, time, chess, chess.pgn
from src.analysis.benchmark.bot import Bot

class BenchmarkWorker:
    def __init__(self, bot1_path: str, bot2_path: str, num_games: int, bot1_use_mcts: bool, bot1_use_opening_book: bool, bot2_use_mcts: bool, bot2_use_opening_book: bool, progress_callback=None, status_callback=None):
        self.bot1 = Bot(path=bot1_path, use_mcts=bot1_use_mcts, use_opening_book=bot1_use_opening_book)
        self.bot2 = Bot(path=bot2_path, use_mcts=bot2_use_mcts, use_opening_book=bot2_use_opening_book)
        self.num_games, self.progress_callback, self.status_callback = num_games, progress_callback, status_callback
        self.games_dir = os.path.join("data", "games", "benchmark")
        os.makedirs(self.games_dir, exist_ok=True)
        with open(os.path.join("data", "processed", "opening_book.json"), "r", encoding="utf-8") as f:
            self.opening_book = json.load(f)

    def run(self):
        result_map = {'1-0': "1-0", '0-1': "0-1", '1/2-1/2': "1/2-1/2"}
        for game_idx in range(1, self.num_games + 1):
            self.status_callback(f"🎮 Playing game {game_idx}/{self.num_games}")
            board, game, node, moves_count = chess.Board(), chess.pgn.Game(), None, 0
            game.headers.update({
                "Event": "Bot Benchmarking",
                "Site": "Local",
                "Date": time.strftime("%Y.%m.%d"),
                "Round": "-",
                "White": "Bot1",
                "Black": "Bot2",
                "Result": "*"
            })
            node = game
            while not board.is_game_over():
                current_bot = self.bot1 if board.turn == chess.WHITE else self.bot2
                move = current_bot.get_move(board, self.opening_book)
                if move == chess.Move.null():
                    break
                board.push(move)
                node = node.add_variation(move)
                moves_count += 1
            result = result_map.get(board.result(), "1/2-1/2")
            game.headers["Result"] = result
            pgn_filename = os.path.join(self.games_dir, f"game_{game_idx}.pgn")
            with open(pgn_filename, "w", encoding="utf-8") as pgn_file:
                pgn_file.write(str(game))
            self.progress_callback(game_idx / self.num_games)
            self.status_callback(f"Completed game {game_idx}/{self.num_games}")
        return True