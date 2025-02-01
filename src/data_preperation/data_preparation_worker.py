import os
import h5py
import time
import chess
import chess.pgn
import numpy as np
import chess.engine
import asyncio
import platform
import json
from collections import defaultdict
from src.utils.chess_utils import (convert_board_to_transformer_input, get_move_mapping,
                                   flip_board, flip_move, mirror_rank, mirror_move_rank)
from src.utils.common import init_wandb_run, wandb_log, finish_wandb
try:
    import wandb
except ImportError:
    wandb = None

class DataPreparationWorker:
    def __init__(self, raw_pgn, max_games, min_elo, max_elo, batch_size,
                 engine_path, engine_depth, engine_threads, engine_hash,
                 pgn_file, max_opening_moves, wandb_flag,
                 progress_callback=None, status_callback=None,
                 skip_min_moves=0, skip_max_moves=99999,
                 use_time_analysis=False, analysis_time=0.5):
        self.raw_pgn_file = raw_pgn
        self.max_games = max_games
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.batch_size = batch_size
        self.engine_path = engine_path
        self.engine_depth = engine_depth
        self.engine_threads = engine_threads
        self.engine_hash = engine_hash
        self.pgn_file = pgn_file
        self.max_opening_moves = max_opening_moves
        self.wandb_flag = wandb_flag
        self.progress_callback = progress_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x: None)
        self.skip_min_moves = skip_min_moves
        self.skip_max_moves = skip_max_moves
        self.use_time_analysis = use_time_analysis
        self.analysis_time = analysis_time
        self.positions = defaultdict(lambda: defaultdict(lambda: {"win": 0, "draw": 0, "loss": 0, "eco": "", "name": ""}))
        self.game_counter = 0
        self.start_time = None
        self.batch_inputs = []
        self.batch_policy_targets = []
        self.batch_value_targets = []
        self.move_mapping = get_move_mapping()
        self.output_dir = os.path.abspath(os.path.join("data", "processed"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.total_games_processed = 0
        self.current_dataset_size = 0
        self.elo_list = []
        self.game_lengths = []
        self.time_control_stats = defaultdict(int)
        self.augment_flip = True
        self.augment_mirror_rank = True

    def run(self):
        if self.wandb_flag and wandb is not None:
            wandb_run = init_wandb_run("data_preparation_"+time.strftime("%Y%m%d-%H%M%S"), self.__dict__)
            batch_table = wandb.Table(columns=["Batch", "Batch Size", "Mean Value", "Std Value"])
            game_table = wandb.Table(columns=["Games Processed", "Games Skipped", "Progress", "Batch Size", "Dataset Size"])
            opening_table = wandb.Table(columns=["Opening Games Processed", "Opening Games Skipped", "Opening Progress", "Unique Positions"])
        else:
            batch_table = game_table = opening_table = None
        self.start_time = time.time()
        if platform.system() == "Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
                engine.configure({"Threads": self.engine_threads, "Hash": self.engine_hash})
                self.status_callback("✅ Chess engine initialized successfully.")
                h5_path = os.path.join(self.output_dir, "dataset.h5")
                with h5py.File(h5_path, "w") as h5_file:
                    h5_file.create_dataset("inputs", (0, 64, 144), maxshape=(None, 64, 144),
                                             dtype=np.float32, compression="lzf")
                    h5_file.create_dataset("policy_targets", (0,), maxshape=(None,),
                                             dtype=np.int64, compression="lzf")
                    h5_file.create_dataset("value_targets", (0,), maxshape=(None,),
                                             dtype=np.float32, compression="lzf")
                    skipped_games = 0
                    last_update = time.time()
                    with open(self.raw_pgn_file, "r", errors="ignore") as f:
                        while self.total_games_processed < self.max_games:
                            game = chess.pgn.read_game(f)
                            if not game:
                                break
                            if "Variant" in game.headers:
                                skipped_games += 1
                                continue
                            if game.headers.get("WhiteTitle") == "BOT" or game.headers.get("BlackTitle") == "BOT":
                                skipped_games += 1
                                continue
                            we = game.headers.get("WhiteElo")
                            be = game.headers.get("BlackElo")
                            if not we or not be:
                                skipped_games += 1
                                continue
                            try:
                                we2 = int(we)
                                be2 = int(be)
                                if we2 < self.min_elo or be2 < self.min_elo or we2 > self.max_elo or be2 > self.max_elo:
                                    skipped_games += 1
                                    continue
                            except:
                                skipped_games += 1
                                continue
                            tc = game.headers.get("TimeControl", "")
                            if tc:
                                self.time_control_stats[tc] += 1
                            self.elo_list.extend([we2, be2])
                            result_map = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
                            gr = result_map.get(game.headers.get("Result"))
                            if gr is None:
                                skipped_games += 1
                                continue
                            board = game.board()
                            node = game
                            move_count = 0
                            while node.variations:
                                next_node = node.variation(0)
                                move = next_node.move
                                move_count += 1
                                if move_count < self.skip_min_moves or move_count > self.skip_max_moves:
                                    board.push(move)
                                    node = next_node
                                    continue
                                inp = convert_board_to_transformer_input(board)
                                mid = self.move_mapping.get_index_by_move(move)
                                if mid is None:
                                    board.push(move)
                                    node = next_node
                                    continue
                                limit = (chess.engine.Limit(time=self.analysis_time)
                                         if self.use_time_analysis
                                         else chess.engine.Limit(depth=self.engine_depth))
                                try:
                                    info = engine.analyse(board, limit)
                                    score = info["score"].pov(board.turn)
                                    if score.is_mate():
                                        v = 1.0 if score.mate() > 0 else -1.0
                                    else:
                                        c = score.score()
                                        v = float(np.clip(c / 1000.0, -1.0, 1.0)) if c is not None else 0.0
                                except:
                                    self.status_callback("❌ Engine error")
                                    v = 0.0
                                fv = v if board.turn else -v
                                self.batch_inputs.append(inp)
                                self.batch_policy_targets.append(mid)
                                self.batch_value_targets.append(fv)
                                if self.augment_flip:
                                    flipped_board = flip_board(board)
                                    flipped_move = flip_move(move)
                                    fm_id = self.move_mapping.get_index_by_move(flipped_move)
                                    if fm_id is not None:
                                        fi = convert_board_to_transformer_input(flipped_board)
                                        self.batch_inputs.append(fi)
                                        self.batch_policy_targets.append(fm_id)
                                        self.batch_value_targets.append(-fv)
                                if self.augment_mirror_rank:
                                    mirrored_board = mirror_rank(board)
                                    mirrored_move = mirror_move_rank(move)
                                    rm_id = self.move_mapping.get_index_by_move(mirrored_move)
                                    if rm_id is not None:
                                        ri = convert_board_to_transformer_input(mirrored_board)
                                        self.batch_inputs.append(ri)
                                        self.batch_policy_targets.append(rm_id)
                                        self.batch_value_targets.append(-fv)
                                board.push(move)
                                node = next_node
                            if move_count < self.skip_min_moves or move_count == 0:
                                skipped_games += 1
                                continue
                            self.total_games_processed += 1
                            if len(self.batch_inputs) >= self.batch_size:
                                self._write_batch_to_h5(h5_file, batch_table)
                            if (self.total_games_processed % 10 == 0) or (time.time() - last_update > 5):
                                progress = min(int((self.total_games_processed / self.max_games) * 100), 100)
                                self.progress_callback(progress)
                                self.status_callback(f"✅ Processed {self.total_games_processed}/{self.max_games} games. Skipped {skipped_games} games.")
                                if self.wandb_flag and wandb is not None:
                                    self._log_game_stats_to_wandb(game_table, skipped_games, progress)
                                last_update = time.time()
                    if self.batch_inputs:
                        self._write_batch_to_h5(h5_file, batch_table)
                    self._generate_opening_book(opening_table)
                    self._create_train_val_test_split(h5_path)
                self.status_callback(f"✅ Data Preparation completed successfully. Processed {self.total_games_processed} games with {skipped_games} skipped. Time: {time.time()-self.start_time:.2f} seconds.")
        except chess.engine.EngineError as e:
            self.status_callback(f"❌ Failed to initialize engine: {e}")
        except Exception as e:
            self.status_callback(f"❌ An unexpected error occurred: {e}")
        if self.wandb_flag and wandb is not None:
            self._final_wandb_logs()
            finish_wandb()
        return True

    def _write_batch_to_h5(self, h5_file, batch_table):
        batch_size = len(self.batch_inputs)
        end_index = self.current_dataset_size + batch_size
        try:
            i_np = np.array(self.batch_inputs, dtype=np.float32)
            p_np = np.array(self.batch_policy_targets, dtype=np.int64)
            v_np = np.array(self.batch_value_targets, dtype=np.float32)
            h5_file["inputs"].resize((end_index, 64, 144))
            h5_file["policy_targets"].resize((end_index,))
            h5_file["value_targets"].resize((end_index,))
            h5_file["inputs"][self.current_dataset_size:end_index, :, :] = i_np
            h5_file["policy_targets"][self.current_dataset_size:end_index] = p_np
            h5_file["value_targets"][self.current_dataset_size:end_index] = v_np
            m = float(np.mean(v_np))
            s = float(np.std(v_np))
            if self.wandb_flag and wandb is not None:
                wandb_log({"batch_size": batch_size, "mean_value_targets": m, "std_value_targets": s})
                batch_table.add_data(str(self.total_games_processed), batch_size, m, s)
            self.current_dataset_size += batch_size
        except (ValueError, KeyError, TypeError) as e:
            self.status_callback(f"❌ Error writing to HDF5: {e}")
        finally:
            self.batch_inputs.clear()
            self.batch_policy_targets.clear()
            self.batch_value_targets.clear()

    def _log_game_stats_to_wandb(self, game_table, skipped, progress):
        try:
            if wandb is not None:
                game_table.add_data(str(self.total_games_processed), skipped, progress, len(self.batch_inputs), self.current_dataset_size)
                wandb_log({
                    "games_processed": self.total_games_processed,
                    "games_skipped": skipped,
                    "progress": progress,
                    "current_batch_size": len(self.batch_inputs),
                    "total_dataset_size": self.current_dataset_size
                })
        except:
            pass

    def _generate_opening_book(self, opening_table):
        if not self.pgn_file or not self.max_opening_moves:
            return
        self.status_callback("🔍 Processing Opening Book...")
        skipped_opening = 0
        last_update = time.time()
        with open(self.pgn_file, "r", encoding="utf-8", errors="ignore") as pf:
            while self.game_counter < self.max_games:
                game = chess.pgn.read_game(pf)
                if not game:
                    self.status_callback("🔍 Reached end of PGN file for opening book.")
                    break
                headers = game.headers
                try:
                    we = int(headers.get("WhiteElo", 0))
                    be = int(headers.get("BlackElo", 0))
                    if we < self.min_elo or be < self.min_elo or we > self.max_elo or be > self.max_elo:
                        skipped_opening += 1
                        continue
                    result_map = {"1-0": "win", "0-1": "loss", "1/2-1/2": "draw"}
                    outcome = result_map.get(headers.get("Result"))
                    if not outcome:
                        skipped_opening += 1
                        continue
                    eco = headers.get("ECO", "")
                    opening_name = headers.get("Opening", "")
                    board = game.board()
                    for count, move in enumerate(game.mainline_moves(), 1):
                        if count > self.max_opening_moves:
                            break
                        fen = board.fen()
                        uci_move = move.uci()
                        stats = self.positions[fen][uci_move]
                        stats[outcome] += 1
                        if not stats["eco"]:
                            stats["eco"] = eco
                        if not stats["name"]:
                            stats["name"] = opening_name
                        board.push(move)
                    self.game_counter += 1
                    if (self.game_counter % 10 == 0) or (time.time() - last_update > 5):
                        progress = min(int((self.game_counter / self.max_games) * 100), 100)
                        self.progress_callback(progress)
                        self.status_callback(f"✅ Processed {self.game_counter}/{self.max_games} opening games. Skipped {skipped_opening} games.")
                        if self.wandb_flag and wandb is not None:
                            try:
                                wandb_log({
                                    "opening_games_processed": self.game_counter,
                                    "opening_games_skipped": skipped_opening,
                                    "opening_progress": progress,
                                    "unique_positions": len(self.positions)
                                })
                            except:
                                pass
                        last_update = time.time()
                except:
                    skipped_opening += 1
                    self.status_callback("❌ Invalid or error in opening game.")
        try:
            pos_dict = {k: dict(v) for k, v in self.positions.items()}
            book_file = os.path.join(self.output_dir, "opening_book.json")
            with open(book_file, "w") as f:
                json.dump(pos_dict, f, indent=4)
            self.status_callback(f"✅ Opening book saved at {book_file}.")
        except:
            self.status_callback("❌ Failed to save opening book.")

    def _create_train_val_test_split(self, h5_path):
        self.status_callback("🔍 Splitting dataset into train, validation, and test sets...")
        try:
            with h5py.File(h5_path, "r") as hf:
                n = hf["inputs"].shape[0]
                if n == 0:
                    self.status_callback("❌ No samples to split.")
                    return
                indices = np.random.permutation(n)
                train_end = int(n * 0.8)
                val_end = int(n * 0.9)
                splits = {
                    "train": indices[:train_end],
                    "val": indices[train_end:val_end],
                    "test": indices[val_end:]
                }
                for split, idx in splits.items():
                    np.save(os.path.join(self.output_dir, f"{split}_indices.npy"), idx)
                if self.wandb_flag and wandb is not None:
                    wandb_log({
                        "train_size": len(splits["train"]),
                        "val_size": len(splits["val"]),
                        "test_size": len(splits["test"])
                    })
                self.status_callback(f"✅ Dataset split into Train ({len(splits['train'])}), Validation ({len(splits['val'])}), Test ({len(splits['test'])}) samples.")
        except:
            self.status_callback("❌ Error splitting dataset.")

    def _final_wandb_logs(self):
        if self.wandb_flag and wandb is not None:
            if self.elo_list:
                et = wandb.Table(data=[[e] for e in self.elo_list], columns=["ELO"])
                wandb_log({"ELO Distribution": wandb.plot.histogram(et, "ELO", "ELO Distribution")})
            if self.game_lengths:
                lt = wandb.Table(data=[[l] for l in self.game_lengths], columns=["Length"])
                wandb_log({"Game Length Distribution": wandb.plot.histogram(lt, "Length", "Game Length Distribution")})
            if self.time_control_stats:
                tc = wandb.Table(columns=["TimeControl", "Count"])
                for k, v in self.time_control_stats.items():
                    tc.add_data(k, v)
                wandb_log({"Time Control Breakdown": wandb.plot.bar(tc, "TimeControl", "Count", "Time Control Stats")})
            import os
            from src.utils import common
            da = wandb.Artifact("chess_dataset", type="dataset")
            da.add_file(os.path.join(self.output_dir, "dataset.h5"))
            wandb.log_artifact(da)
            book_path = os.path.join(self.output_dir, "opening_book.json")
            if os.path.exists(book_path):
                ba = wandb.Artifact("opening_book", type="dataset")
                ba.add_file(book_path)
                wandb.log_artifact(ba)