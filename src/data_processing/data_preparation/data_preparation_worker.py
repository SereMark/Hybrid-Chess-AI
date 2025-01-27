import os, h5py, time, chess, chess.pgn, numpy as np, chess.engine, asyncio, platform
from collections import defaultdict
from src.utils.chess_utils import convert_board_to_tensor, flip_board, flip_move, get_move_mapping

class DataPreparationWorker:
    def __init__(self, raw_pgn_file, max_games, min_elo, batch_size, engine_path, engine_depth, engine_threads, engine_hash, progress_callback=None, status_callback=None):
        self.raw_pgn_file, self.max_games, self.min_elo, self.batch_size = raw_pgn_file, max_games, min_elo, batch_size
        self.engine_path, self.engine_depth, self.engine_threads, self.engine_hash = engine_path, engine_depth, engine_threads, engine_hash
        self.progress_callback, self.status_callback = progress_callback, status_callback
        self.positions = defaultdict(lambda: defaultdict(lambda: {"win":0,"draw":0,"loss":0,"eco":"","name":""}))
        self.game_counter, self.start_time = 0, None
        self.batch_inputs, self.batch_policy_targets, self.batch_value_targets = [], [], []
        self.move_mapping = get_move_mapping()
        self.output_dir = os.path.abspath(os.path.join("data","processed"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.total_games_processed, self.current_dataset_size = 0, 0

    def run(self):
        self.start_time = time.time()
        try:
            if platform.system() == "Windows":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            self.engine.configure({"Threads": self.engine_threads, "Hash": self.engine_hash})
            self.status_callback("✅ Chess engine initialized successfully.")
        except Exception as e:
            self.status_callback(f"❌ Failed to initialize engine: {e}")
            self.engine = None
        h5_path = os.path.join(self.output_dir, "dataset.h5")
        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset("inputs", shape=(0,25,8,8), maxshape=(None,25,8,8), dtype=np.float32, compression="lzf")
            h5_file.create_dataset("policy_targets", shape=(0,), maxshape=(None,), dtype=np.int64, compression="lzf")
            h5_file.create_dataset("value_targets", shape=(0,), maxshape=(None,), dtype=np.float32, compression="lzf")
            skipped_games, last_update_time = 0, time.time()
            with open(self.raw_pgn_file, "r", errors="ignore") as f:
                while self.total_games_processed < self.max_games:
                    game = chess.pgn.read_game(f)
                    if not game:
                        break
                    headers = game.headers
                    white_elo_str, black_elo_str = headers.get("WhiteElo"), headers.get("BlackElo")
                    if not white_elo_str or not black_elo_str:
                        skipped_games +=1
                        continue
                    try:
                        white_elo, black_elo = int(white_elo_str), int(black_elo_str)
                        if white_elo < self.min_elo or black_elo < self.min_elo:
                            skipped_games +=1
                            continue
                    except:
                        skipped_games +=1
                        continue
                    result_map = {"1-0":1.0, "0-1":-1.0, "1/2-1/2":0.0}
                    game_result = result_map.get(headers.get("Result"))
                    if game_result is None:
                        skipped_games +=1
                        continue
                    board, moves = game.board(), list(game.mainline_moves())
                    inputs, policy_targets, value_targets = [], [], []
                    for move in moves:
                        current_tensor = convert_board_to_tensor(board)
                        move_idx = self.move_mapping.get_index_by_move(move)
                        if move_idx is None:
                            self.status_callback(f"ℹ️ Move index not found for move: {move}")
                            board.push(move)
                            continue
                        if not self.engine:
                            value_target = 0.0
                        else:
                            try:
                                info = self.engine.analyse(board, chess.engine.Limit(depth=self.engine_depth))
                                score = info["score"].pov(board.turn)
                                if score.is_mate():
                                    value_target = 1.0 if score.mate() >0 else -1.0
                                else:
                                    value_target = max(min(score.score(),1000),-1000)/1000.0
                            except:
                                self.status_callback("❌ Error evaluating position.")
                                value_target = 0.0
                        inputs.append(current_tensor)
                        policy_targets.append(move_idx)
                        value_targets.append(value_target)
                        flipped = flip_board(board)
                        flipped_move = flip_move(move)
                        flipped_idx = self.move_mapping.get_index_by_move(flipped_move)
                        if flipped_idx is not None:
                            inputs.append(convert_board_to_tensor(flipped))
                            policy_targets.append(flipped_idx)
                            value_targets.append(-value_target)
                        board.push(move)
                    if not inputs:
                        skipped_games +=1
                        continue
                    self.batch_inputs += inputs
                    self.batch_policy_targets += policy_targets
                    self.batch_value_targets += value_targets
                    if len(self.batch_inputs) >= self.batch_size:
                        batch_size = len(self.batch_inputs)
                        end_idx = self.current_dataset_size + batch_size
                        for ds, data in zip(["inputs", "policy_targets", "value_targets"], [self.batch_inputs, self.batch_policy_targets, self.batch_value_targets]):
                            h5_file[ds].resize((end_idx,) + h5_file[ds].shape[1:])
                            h5_file[ds][self.current_dataset_size:end_idx] = np.array(data, dtype=h5_file[ds].dtype)
                        self.current_dataset_size += batch_size
                        self.batch_inputs, self.batch_policy_targets, self.batch_value_targets = [], [], []
                    self.total_games_processed +=1
                    if self.total_games_processed %10 ==0 or time.time()-last_update_time >5:
                        progress = (self.total_games_processed / self.max_games) *100
                        self.progress_callback(int(progress))
                        self.status_callback(f"✅ Processed {self.total_games_processed}/{self.max_games} games. Skipped {skipped_games} games so far.")
                        last_update_time = time.time()
            if self.batch_inputs:
                batch_size = len(self.batch_inputs)
                end_idx = self.current_dataset_size + batch_size
                for ds, data in zip(["inputs", "policy_targets", "value_targets"], [self.batch_inputs, self.batch_policy_targets, self.batch_value_targets]):
                    h5_file[ds].resize((end_idx,) + h5_file[ds].shape[1:])
                    h5_file[ds][self.current_dataset_size:end_idx] = np.array(data, dtype=h5_file[ds].dtype)
                self.current_dataset_size += batch_size
            if self.engine:
                self.engine.close()
                self.status_callback("🔍 Chess engine closed.")
            self.status_callback("🔍 Splitting dataset into train, validation, and test sets...")
            train_indices_path, val_indices_path, test_indices_path = map(lambda x: os.path.join(self.output_dir, f"{x}_indices.npy"), ["train", "val", "test"])
            try:
                with h5py.File(h5_path, "r") as h5_file:
                    num_samples = h5_file["inputs"].shape[0]
                    if num_samples ==0:
                        self.status_callback("❌ No samples to split.")
                        return
                    indices = np.random.permutation(num_samples)
                    train_end, val_end = int(num_samples*0.8), int(num_samples*0.9)
                    np.save(train_indices_path, indices[:train_end])
                    np.save(val_indices_path, indices[train_end:val_end])
                    np.save(test_indices_path, indices[val_end:])
                    self.status_callback(f"✅ Dataset split into Train ({train_end}), Validation ({val_end - train_end}), Test ({num_samples - val_end}) samples.")
            except:
                self.status_callback("❌ Error splitting dataset.")
            self.status_callback(f"✅ Data Preparation completed successfully. Processed {self.total_games_processed} games with {skipped_games} skipped games in {time.time()-self.start_time:.2f} seconds.")
            return True