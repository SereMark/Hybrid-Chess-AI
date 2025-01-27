import os, chess.pgn, json, time
from collections import defaultdict

class OpeningBookWorker:
    def __init__(self, pgn_file_path, max_games, min_elo, max_opening_moves, progress_callback=None, status_callback=None):
        self.pgn_file_path, self.max_games, self.min_elo, self.max_opening_moves = pgn_file_path, max_games, min_elo, max_opening_moves
        self.positions, self.game_counter, self.start_time = defaultdict(lambda: defaultdict(lambda: {"win":0,"draw":0,"loss":0,"eco":"","name":""})), 0, None
        self.progress_callback, self.status_callback = progress_callback, status_callback

    def run(self):
        self.start_time = time.time()
        self.status_callback("🔍 Starting processing of games...")
        skipped_games, last_update_time = 0, time.time()
        try:
            with open(self.pgn_file_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
                while self.game_counter < self.max_games:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        self.status_callback("🔍 Reached end of PGN file.")
                        break
                    try:
                        white_elo, black_elo = int(game.headers.get("WhiteElo",0)), int(game.headers.get("BlackElo",0))
                        if white_elo < self.min_elo or black_elo < self.min_elo:
                            skipped_games +=1
                            continue
                        outcome_map = {"1-0":"win", "0-1":"loss", "1/2-1/2":"draw"}
                        outcome = outcome_map.get(game.headers.get("Result"))
                        if not outcome:
                            skipped_games +=1
                            continue
                        eco_code, opening_name = game.headers.get("ECO",""), game.headers.get("Opening","")
                        board = game.board()
                        for move_counter, move in enumerate(game.mainline_moves(),1):
                            if move_counter > self.max_opening_moves:
                                break
                            fen, uci_move = board.fen(), move.uci()
                            move_data = self.positions[fen][uci_move]
                            move_data[outcome] +=1
                            move_data["eco"] = move_data["eco"] or eco_code
                            move_data["name"] = move_data["name"] or opening_name
                            board.push(move)
                        self.game_counter +=1
                        if self.game_counter %10 ==0 or time.time()-last_update_time >5:
                            self.progress_callback(min(int((self.game_counter / self.max_games)*100), 100))
                            self.status_callback(f"✅ Processed {self.game_counter}/{self.max_games} games. Skipped {skipped_games} games so far.")
                            last_update_time = time.time()
                    except:
                        skipped_games +=1
                        self.status_callback("❌ Exception in processing game.")
        except:
            self.status_callback("❌ Exception during processing.")
            return
        positions = {fen:dict(moves) for fen, moves in self.positions.items()}
        book_file = os.path.abspath(os.path.join("data","processed","opening_book.json"))
        os.makedirs(os.path.dirname(book_file), exist_ok=True)
        try:
            with open(book_file, "w") as f:
                json.dump(positions, f, indent=4)
            self.status_callback(f"✅ Opening book saved at {book_file}.")
        except:
            self.status_callback("❌ Failed to save opening book.")
            return
        total_time = time.time() - self.start_time
        self.status_callback(f"✅ Completed processing {self.game_counter} games with {skipped_games} skipped games in {total_time:.2f} seconds.")
        opening_stats = defaultdict(int)
        for moves in self.positions.values():
            for stats in moves.values():
                opening_stats[stats["name"]] += stats["win"] + stats["draw"] + stats["loss"]
        return True