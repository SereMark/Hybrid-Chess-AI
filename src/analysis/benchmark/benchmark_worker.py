import os
import json
import time
import numpy as np
import chess
import chess.pgn
from typing import Optional, Callable, Dict

from src.analysis.benchmark.bot import Bot

class BenchmarkWorker:
    def __init__(
        self,
        bot1_path: str,
        bot2_path: str,
        num_games: int,
        bot1_use_mcts: bool,
        bot1_use_opening_book: bool,
        bot2_use_mcts: bool,
        bot2_use_opening_book: bool,
        wandb_flag: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ):
        self.num_games = num_games
        self.wandb_flag = wandb_flag
        self.progress_callback = progress_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x: None)

        self.bot1 = Bot(
            path=bot1_path,
            use_mcts=bot1_use_mcts,
            use_opening_book=bot1_use_opening_book
        )
        self.bot2 = Bot(
            path=bot2_path,
            use_mcts=bot2_use_mcts,
            use_opening_book=bot2_use_opening_book
        )

        self.games_dir = os.path.join("data", "games", "benchmark")
        os.makedirs(self.games_dir, exist_ok=True)

        opening_book_path = os.path.join("data", "processed", "opening_book.json")
        if os.path.isfile(opening_book_path):
            try:
                with open(opening_book_path, "r", encoding="utf-8") as f:
                    self.opening_book = json.load(f)
            except Exception as e:
                self.status_callback(f"⚠️ Failed to load opening book: {e}")
                self.opening_book = {}
        else:
            self.opening_book = {}
            self.status_callback("⚠️ Opening book file not found. Book-based moves might fail.")

    def run(self) -> Dict[str, int]:
        if self.wandb_flag:
            import wandb
            wandb.init(entity="chess_ai", project="chess_ai_app", name="benchmark", config=self.__dict__, reinit=True)

        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
        durations = []
        move_counts = []
        bot1_wins_over_time = []
        bot2_wins_over_time = []
        draws_over_time = []

        bot1_wins = 0
        bot2_wins = 0
        draws = 0

        for game_idx in range(1, self.num_games + 1):
            self.status_callback(f"🎮 Playing game {game_idx}/{self.num_games}")
            start_time = time.time()

            board = chess.Board()
            game = chess.pgn.Game()
            game.headers.update({
                "Event": "Bot Benchmarking",
                "Site": "Local",
                "Date": time.strftime("%Y.%m.%d"),
                "Round": str(game_idx),
                "White": "Bot1",
                "Black": "Bot2",
                "Result": "*"
            })

            node = game
            moves_made = 0

            while not board.is_game_over():
                current_bot = self.bot1 if board.turn == chess.WHITE else self.bot2
                next_move = current_bot.get_move(board, self.opening_book)

                if (not next_move) or (next_move == chess.Move.null()) or (next_move not in board.legal_moves):
                    self.status_callback("⚠️ Bot returned invalid or null move. Ending this game.")
                    break

                board.push(next_move)
                node = node.add_variation(next_move)
                moves_made += 1

            result = board.result()  
            if result not in results:
                results[result] = 0
            results[result] += 1
            game.headers["Result"] = result

            game_path = os.path.join(self.games_dir, f"game_{game_idx}.pgn")
            with open(game_path, "w", encoding="utf-8") as pgn_file:
                pgn_file.write(str(game))

            end_time = time.time()
            duration_sec = end_time - start_time
            durations.append(duration_sec)
            move_counts.append(moves_made)

            if result == "1-0":
                bot1_wins += 1
            elif result == "0-1":
                bot2_wins += 1
            elif result == "1/2-1/2":
                draws += 1

            bot1_wins_over_time.append(bot1_wins)
            bot2_wins_over_time.append(bot2_wins)
            draws_over_time.append(draws)

            self.progress_callback(100.0 * game_idx / self.num_games)
            self.status_callback(f"Game {game_idx} finished with result {result} in {duration_sec:.2f} seconds.")

            if self.wandb_flag:
                import wandb
                wandb.log({
                    "game_index": game_idx,
                    "game_result": result,
                    "game_duration_sec": duration_sec,
                    "moves_made": moves_made,
                    "bot1_wins_so_far": bot1_wins,
                    "bot2_wins_so_far": bot2_wins,
                    "draws_so_far": draws
                })

        total_avg_duration = float(np.mean(durations)) if durations else 0.0
        total_avg_moves = float(np.mean(move_counts)) if move_counts else 0.0

        self.status_callback(f"✅ All {self.num_games} games completed.")
        self.status_callback(
            "Results: "
            f"Bot1 wins={results.get('1-0', 0)}, "
            f"Bot2 wins={results.get('0-1', 0)}, "
            f"draws={results.get('1/2-1/2', 0)}, "
            f"unfinished={results.get('*', 0)}"
        )

        if self.wandb_flag:
            import wandb

            wandb.log({
                "total_games": self.num_games,
                "wins_bot1": results["1-0"],
                "wins_bot2": results["0-1"],
                "draws": results["1/2-1/2"],
                "unfinished": results["*"],
                "avg_game_duration": total_avg_duration,
                "avg_moves_per_game": total_avg_moves
            })

            scoreboard_table = wandb.Table(
                columns=["Result", "Count"],
                data=[
                    ["1-0", results["1-0"]],
                    ["0-1", results["0-1"]],
                    ["1/2-1/2", results["1/2-1/2"]],
                    ["*", results["*"]],
                ]
            )
            wandb.log({
                "results_bar": wandb.plot.bar(scoreboard_table, "Result", "Count", title="Game Outcomes")
            })

            if durations:
                wandb.log({
                    "game_length_distribution": wandb.plot.histogram(
                        wandb.Table(data=[[x] for x in durations], columns=["DurationSec"]),
                        "DurationSec",
                        title="Game Duration Distribution (sec)"
                    )
                })
            if move_counts:
                wandb.log({
                    "move_count_distribution": wandb.plot.histogram(
                        wandb.Table(data=[[m] for m in move_counts], columns=["Moves"]),
                        "Moves",
                        title="Game Move Count Distribution"
                    )
                })

            line_data = list(zip(
                range(1, self.num_games + 1),
                bot1_wins_over_time,
                bot2_wins_over_time,
                draws_over_time
            ))
            line_table = wandb.Table(data=line_data, columns=["Game", "Bot1WinsSoFar", "Bot2WinsSoFar", "DrawsSoFar"])
            wandb.log({
                "wins_over_time": wandb.plot.line_series(
                    xs=line_table.get_column("Game"),
                    ys=[
                        line_table.get_column("Bot1WinsSoFar"),
                        line_table.get_column("Bot2WinsSoFar"),
                        line_table.get_column("DrawsSoFar")
                    ],
                    keys=["Bot1 Wins", "Bot2 Wins", "Draws"],
                    title="Wins Over Time"
                )
            })

            wandb.run.summary.update({
                "Wins(Bot1)": results["1-0"],
                "Wins(Bot2)": results["0-1"],
                "Draws": results["1/2-1/2"],
                "Unfinished": results["*"],
                "Avg Duration (sec)": total_avg_duration,
                "Avg Moves Per Game": total_avg_moves
            })
            try:
                wandb.finish()
            except Exception as e:
                self.status_callback(f"⚠️ Error finishing wandb run: {e}")

        return results