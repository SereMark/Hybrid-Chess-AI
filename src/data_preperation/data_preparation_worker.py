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
from src.utils.chess_utils import (
    convert_board_to_tensor,
    get_move_mapping,
    apply_augmentations
)

class DataPreparationWorker:
    def __init__(
        self,
        raw_pgn,
        max_games,
        min_elo,
        batch_size,
        engine_path,
        engine_depth,
        engine_threads,
        engine_hash,
        pgn_file,
        max_opening_moves,
        wandb_flag,
        progress_callback=None,
        status_callback=None,
        skip_min_moves=0,
        skip_max_moves=99999,
        use_time_analysis=False,
        analysis_time=0.5
    ):
        self.raw_pgn_file = raw_pgn
        self.max_games = max_games
        self.min_elo = min_elo
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

        self.positions = defaultdict(lambda: defaultdict(lambda: {"win":0,"draw":0,"loss":0,"eco":"","name":""}))
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
        import wandb
        if self.wandb_flag:
            wandb.init(
                entity="chess_ai",
                project="chess_ai_app",
                name="data_preparation",
                config=self.__dict__,
                reinit=True
            )
            batch_table = wandb.Table(columns=["Batch", "Batch Size", "Mean Value", "Std Value"])
            game_table = wandb.Table(columns=["Games Processed", "Games Skipped", "Progress", "Batch Size", "Dataset Size"])
            opening_table = wandb.Table(columns=["Opening Games Processed", "Opening Games Skipped", "Opening Progress", "Unique Positions"])
        else:
            batch_table = None
            game_table = None
            opening_table = None

        self.start_time = time.time()

        if platform.system() == "Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        try:
            with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
                engine.configure({"Threads": self.engine_threads, "Hash": self.engine_hash})
                self.status_callback("✅ Chess engine initialized successfully.")

                h5_path = os.path.join(self.output_dir, "dataset.h5")
                with h5py.File(h5_path, "w") as h5_file:
                    for ds, shape, dtype in [
                        ("inputs", (0,25,8,8), np.float32),
                        ("policy_targets", (0,), np.int64),
                        ("value_targets", (0,), np.float32)
                    ]:
                        h5_file.create_dataset(
                            ds,
                            shape=shape,
                            maxshape=(None,) + shape[1:],
                            dtype=dtype,
                            compression="lzf"
                        )

                    skipped_games = 0
                    last_update = time.time()

                    with open(self.raw_pgn_file, "r", errors="ignore") as f:
                        while self.total_games_processed < self.max_games:
                            game = chess.pgn.read_game(f)
                            if not game:
                                break

                            headers = game.headers
                            white_elo_str = headers.get("WhiteElo")
                            black_elo_str = headers.get("BlackElo")

                            if not white_elo_str or not black_elo_str:
                                skipped_games += 1
                                continue

                            try:
                                white_elo = int(white_elo_str)
                                black_elo = int(black_elo_str)
                                if white_elo < self.min_elo or black_elo < self.min_elo:
                                    skipped_games += 1
                                    continue
                            except ValueError:
                                skipped_games += 1
                                continue

                            time_control = headers.get("TimeControl", "")
                            if time_control:
                                self.time_control_stats[time_control] += 1

                            self.elo_list.append(white_elo)
                            self.elo_list.append(black_elo)

                            result_map = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
                            game_result = result_map.get(headers.get("Result"))
                            if game_result is None:
                                skipped_games += 1
                                continue

                            board = game.board()
                            moves = list(game.mainline_moves())
                            num_moves = len(moves)
                            self.game_lengths.append(num_moves)

                            if num_moves < self.skip_min_moves or num_moves > self.skip_max_moves:
                                skipped_games += 1
                                continue

                            inputs, policy, value = [], [], []
                            for move in moves:
                                tensor = convert_board_to_tensor(board)
                                move_idx = self.move_mapping.get_index_by_move(move)
                                if move_idx is None:
                                    board.push(move)
                                    continue

                                if self.use_time_analysis:
                                    limit = chess.engine.Limit(time=self.analysis_time)
                                else:
                                    limit = chess.engine.Limit(depth=self.engine_depth)

                                try:
                                    info = engine.analyse(board, limit)
                                    score = info["score"].pov(board.turn)

                                    if score.is_mate():
                                        mate_score = score.mate()
                                        value_target = 1.0 if mate_score > 0 else -1.0
                                    else:
                                        raw_score = score.score()
                                        if raw_score is not None:
                                            value_target = np.clip(raw_score / 1000.0, -1.0, 1.0)
                                        else:
                                            value_target = 0.0
                                except chess.engine.EngineError as e:
                                    self.status_callback(f"❌ Engine analysis error: {e}")
                                    value_target = 0.0

                                inputs.append(tensor)
                                policy.append(move_idx)
                                value.append(game_result if board.turn else -game_result)

                                if self.augment_flip:
                                    flipped_board, flipped_mv = apply_augmentations(board, move, method="flip")
                                    flipped_idx = self.move_mapping.get_index_by_move(flipped_mv)
                                    if flipped_idx is not None:
                                        inputs.append(convert_board_to_tensor(flipped_board))
                                        policy.append(flipped_idx)
                                        value.append(-value_target)

                                if self.augment_mirror_rank:
                                    rank_board, rank_mv = apply_augmentations(board, move, method="mirror_rank")
                                    rank_idx = self.move_mapping.get_index_by_move(rank_mv)
                                    if rank_idx is not None:
                                        inputs.append(convert_board_to_tensor(rank_board))
                                        policy.append(rank_idx)
                                        value.append(-value_target)

                                board.push(move)

                            if not inputs:
                                skipped_games += 1
                                continue

                            self.batch_inputs.extend(inputs)
                            self.batch_policy_targets.extend(policy)
                            self.batch_value_targets.extend(value)

                            if len(self.batch_inputs) >= self.batch_size:
                                self._write_batch_to_h5(h5_file, batch_table, wandb)

                            self.total_games_processed += 1

                            if (self.total_games_processed % 10 == 0) or (time.time() - last_update > 5):
                                progress = min(int((self.total_games_processed / self.max_games) * 100), 100)
                                self.progress_callback(progress)
                                self.status_callback(
                                    f"✅ Processed {self.total_games_processed}/{self.max_games} games. "
                                    f"Skipped {skipped_games} games."
                                )
                                if self.wandb_flag:
                                    self._log_game_stats_to_wandb(
                                        wandb, game_table,
                                        skipped_games, progress
                                    )
                                last_update = time.time()

                    if self.batch_inputs:
                        self._write_batch_to_h5(h5_file, batch_table, wandb)

                    self._generate_opening_book(wandb, engine, opening_table)

                    self._create_train_val_test_split(h5_path, wandb)

                self.status_callback(
                    f"✅ Data Preparation completed successfully. "
                    f"Processed {self.total_games_processed} games with {skipped_games} skipped. "
                    f"Time: {time.time() - self.start_time:.2f} seconds."
                )

        except chess.engine.EngineError as e:
            self.status_callback(f"❌ Failed to initialize engine: {e}")
        except Exception as e:
            self.status_callback(f"❌ An unexpected error occurred: {e}")

        if self.wandb_flag:
            self._final_wandb_logs()
            try:
                wandb.finish()
            except Exception as e:
                self.status_callback(f"⚠️ Error finishing wandb run: {e}")

        return True

    def _write_batch_to_h5(self, h5_file, batch_table, wandb):
        batch_size = len(self.batch_inputs)
        end_idx = self.current_dataset_size + batch_size
        try:
            for ds_name, data_list in zip(
                ["inputs", "policy_targets", "value_targets"],
                [self.batch_inputs, self.batch_policy_targets, self.batch_value_targets]
            ):
                ds = h5_file[ds_name]
                ds.resize((end_idx,) + ds.shape[1:])
                ds[self.current_dataset_size:end_idx] = np.array(data_list, dtype=ds.dtype)

            mean_val = float(np.mean(self.batch_value_targets))
            std_val = float(np.std(self.batch_value_targets))

            if wandb:
                try:
                    batch_table.add_data(
                        str(self.total_games_processed),
                        batch_size,
                        mean_val,
                        std_val
                    )
                    wandb.log({
                        "batch_size": batch_size,
                        "mean_value_targets": mean_val,
                        "std_value_targets": std_val,
                        "Mean Value": wandb.plot.line(
                            batch_table, "Batch", "Mean Value", "Mean Value Targets Over Batches"
                        ),
                        "Value Distribution": wandb.plot.histogram(
                            wandb.Table(data=[[v] for v in self.batch_value_targets], columns=["value"]),
                            "value", "Value Targets Distribution"
                        ),
                        "Policy vs Value": wandb.plot.scatter(
                            wandb.Table(
                                data=list(zip(map(str, self.batch_policy_targets), self.batch_value_targets)),
                                columns=["Policy", "Value"]
                            ),
                            "Policy", "Value", "Policy vs Value Targets"
                        )
                    })
                except Exception as e:
                    self.status_callback(f"❌ Error during wandb logging: {e}")

            self.current_dataset_size += batch_size
        except Exception as e:
            self.status_callback(f"❌ Error writing to h5py: {e}")
        finally:
            self.batch_inputs.clear()
            self.batch_policy_targets.clear()
            self.batch_value_targets.clear()

    def _log_game_stats_to_wandb(self, wandb, game_table, skipped_games, progress):
        try:
            game_table.add_data(
                str(self.total_games_processed),
                skipped_games,
                progress,
                len(self.batch_inputs),
                self.current_dataset_size
            )
            wandb.log({
                "games_processed": self.total_games_processed,
                "games_skipped": skipped_games,
                "progress": progress,
                "current_batch_size": len(self.batch_inputs),
                "total_dataset_size": self.current_dataset_size,
                "Games Bar": wandb.plot.bar(
                    game_table,
                    "Games Processed",
                    ["Games Processed", "Games Skipped"],
                    "Processed vs Skipped Games"
                )
            })
        except Exception as e:
            self.status_callback(f"❌ Error during wandb logging: {e}")

    def _generate_opening_book(self, wandb, engine, opening_table):
        if not self.pgn_file or not self.max_opening_moves:
            return

        self.status_callback("🔍 Processing Opening Book...")
        skipped_book = 0
        last_book_update = time.time()

        try:
            import wandb
        except ImportError:
            pass

        with open(self.pgn_file, "r", encoding="utf-8", errors="ignore") as pgn_f:
            while self.game_counter < self.max_games:
                game = chess.pgn.read_game(pgn_f)
                if not game:
                    self.status_callback("🔍 Reached end of PGN file for opening book.")
                    break

                headers = game.headers
                try:
                    white_elo = int(headers.get("WhiteElo", 0))
                    black_elo = int(headers.get("BlackElo", 0))
                    if (white_elo < self.min_elo) or (black_elo < self.min_elo):
                        skipped_book += 1
                        continue

                    outcome = {"1-0": "win", "0-1": "loss", "1/2-1/2": "draw"}.get(headers.get("Result"))
                    if not outcome:
                        skipped_book += 1
                        continue

                    eco = headers.get("ECO", "")
                    name = headers.get("Opening", "")
                    board = game.board()

                    for cnt, move in enumerate(game.mainline_moves(), start=1):
                        if cnt > self.max_opening_moves:
                            break
                        fen = board.fen()
                        uci = move.uci()
                        mdata = self.positions[fen][uci]
                        mdata[outcome] += 1
                        if not mdata["eco"]:
                            mdata["eco"] = eco
                        if not mdata["name"]:
                            mdata["name"] = name
                        board.push(move)

                    self.game_counter += 1

                    if (self.game_counter % 10 == 0) or (time.time() - last_book_update > 5):
                        progress = min(int((self.game_counter / self.max_games) * 100), 100)
                        self.progress_callback(progress)
                        self.status_callback(
                            f"✅ Processed {self.game_counter}/{self.max_games} opening games. "
                            f"Skipped {skipped_book} games."
                        )
                        if wandb:
                            try:
                                opening_table.add_data(
                                    str(self.game_counter),
                                    skipped_book,
                                    progress,
                                    len(self.positions)
                                )
                                wandb.log({
                                    "opening_games_processed": self.game_counter,
                                    "opening_games_skipped": skipped_book,
                                    "opening_progress": progress,
                                    "unique_positions": len(self.positions),
                                    "Opening Bar": wandb.plot.bar(
                                        opening_table,
                                        "Opening Games Processed",
                                        ["Opening Games Processed", "Opening Games Skipped"],
                                        "Opening Games Processed vs Skipped"
                                    )
                                })
                            except Exception as e:
                                self.status_callback(f"❌ Error during wandb logging: {e}")
                        last_book_update = time.time()

                except ValueError:
                    skipped_book += 1
                    self.status_callback("❌ Invalid Elo value in opening game.")
                except Exception as e:
                    skipped_book += 1
                    self.status_callback(f"❌ Exception in processing opening game: {e}")

        try:
            positions_dict = {fen: dict(moves) for fen, moves in self.positions.items()}
            book_file = os.path.join(self.output_dir, "opening_book.json")
            with open(book_file, "w") as bf:
                json.dump(positions_dict, bf, indent=4)
            self.status_callback(f"✅ Opening book saved at {book_file}.")

            if wandb and positions_dict:
                opening_count = defaultdict(int)
                for fen_data in positions_dict.values():
                    for mv_data in fen_data.values():
                        if mv_data["name"]:
                            opening_count[mv_data["name"]] += 1
                sorted_openings = sorted(opening_count.items(), key=lambda x: x[1], reverse=True)
                top_5 = sorted_openings[:5]

                wandb.log({"top_5_openings": str(top_5)})

        except Exception as e:
            self.status_callback(f"❌ Failed to save opening book: {e}")

    def _create_train_val_test_split(self, h5_path, wandb):
        self.status_callback("🔍 Splitting dataset into train, validation, and test sets...")
        try:
            with h5py.File(h5_path, "r") as h5_f:
                num = h5_f["inputs"].shape[0]
                if num == 0:
                    self.status_callback("❌ No samples to split.")
                    return
                idx = np.random.permutation(num)
                train_end = int(num * 0.8)
                val_end = int(num * 0.9)
                splits = {
                    "train": idx[:train_end],
                    "val": idx[train_end:val_end],
                    "test": idx[val_end:]
                }
                for name, indices in splits.items():
                    np.save(os.path.join(self.output_dir, f"{name}_indices.npy"), indices)

                if wandb:
                    split_table = wandb.Table(
                        data=[
                            ["Train", len(splits["train"])],
                            ["Validation", len(splits["val"])],
                            ["Test", len(splits["test"])]
                        ],
                        columns=["Split", "Samples"]
                    )
                    wandb.log({
                        "dataset_split": wandb.plot.bar(
                            split_table, "Split", "Samples", "Dataset Split"
                        )
                    })

                self.status_callback(
                    f"✅ Dataset split into Train ({len(splits['train'])}), "
                    f"Validation ({len(splits['val'])}), Test ({len(splits['test'])}) samples."
                )
        except Exception as e:
            self.status_callback(f"❌ Error splitting dataset: {e}")

    def _final_wandb_logs(self):
        import wandb

        if self.elo_list:
            elo_table = wandb.Table(data=[[e] for e in self.elo_list], columns=["ELO"])
            wandb.log({
                "ELO Distribution": wandb.plot.histogram(elo_table, "ELO", title="ELO Distribution")
            })

        if self.game_lengths:
            length_table = wandb.Table(data=[[l] for l in self.game_lengths], columns=["Length"])
            wandb.log({
                "Game Length Distribution": wandb.plot.histogram(length_table, "Length", title="Game Length Distribution")
            })

        if self.time_control_stats:
            tc_table = wandb.Table(columns=["TimeControl", "Count"])
            for tc, count in self.time_control_stats.items():
                tc_table.add_data(tc, count)
            wandb.log({
                "Time Control Breakdown": wandb.plot.bar(tc_table, "TimeControl", "Count", title="Time Control Stats")
            })

        dataset_artifact = wandb.Artifact("chess_dataset", type="dataset")
        dataset_artifact.add_file(os.path.join(self.output_dir, "dataset.h5"))
        wandb.log_artifact(dataset_artifact)

        book_path = os.path.join(self.output_dir, "opening_book.json")
        if os.path.exists(book_path):
            book_artifact = wandb.Artifact("opening_book", type="dataset")
            book_artifact.add_file(book_path)
            wandb.log_artifact(book_artifact)