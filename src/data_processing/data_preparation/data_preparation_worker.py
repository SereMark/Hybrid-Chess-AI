import os, io, h5py, time, chess, chess.pgn, numpy as np, chess.engine, asyncio, platform
from collections import defaultdict
from src.utils.chess_utils import convert_board_to_tensor, flip_board, flip_move, get_move_mapping

class DataPreparationWorker:
    def __init__(self, raw_pgn_file, max_games, min_elo, batch_size, engine_path, engine_depth, engine_threads, engine_hash, progress_callback=None, status_callback=None):
        self.raw_pgn_file, self.max_games, self.min_elo, self.batch_size = raw_pgn_file, max_games, min_elo, batch_size
        self.engine_path, self.engine_depth, self.engine_threads, self.engine_hash = engine_path, engine_depth, engine_threads, engine_hash
        self.progress_callback, self.status_callback = progress_callback, status_callback
        self.positions = defaultdict(lambda: defaultdict(lambda: {"win":0,"draw":0,"loss":0,"eco":"","name":""}))
        self.game_counter, self.start_time, self.total_samples = 0, None, 0
        self.total_games_processed, self.total_moves_processed = 0, 0
        self.game_results_counter = {1.0:0, -1.0:0, 0.0:0}
        self.game_length_bins = np.arange(0,200,5)
        self.game_length_histogram = np.zeros(len(self.game_length_bins)-1, dtype=int)
        self.player_rating_bins = np.arange(1000,3000,50)
        self.player_rating_histogram = np.zeros(len(self.player_rating_bins)-1, dtype=int)
        self.batch_inputs, self.batch_policy_targets, self.batch_value_targets = [], [], []
        self.current_dataset_size = 0
        self.move_mapping = get_move_mapping()
        self.output_dir = os.path.abspath(os.path.join("data","processed"))
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        self.start_time = time.time()
        try:
            if platform.system() == "Windows":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            self.engine.configure({"Threads": self.engine_threads, "Hash": self.engine_hash})
            if self.status_callback:
                self.status_callback("✅ Chess engine initialized successfully.")
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"❌ Failed to initialize engine: {e}")
            self.engine = None
        if self.status_callback:
            self.status_callback("🔍 Starting game processing...")
        h5_path = os.path.join(self.output_dir, "dataset.h5")
        with h5py.File(h5_path, "w") as h5_file:
            self.h5_inputs = h5_file.create_dataset("inputs", shape=(0,25,8,8), maxshape=(None,25,8,8), dtype=np.float32, compression="lzf")
            self.h5_policy_targets = h5_file.create_dataset("policy_targets", shape=(0,), maxshape=(None,), dtype=np.int64, compression="lzf")
            self.h5_value_targets = h5_file.create_dataset("value_targets", shape=(0,), maxshape=(None,), dtype=np.float32, compression="lzf")
            skipped_games = 0
            with open(self.raw_pgn_file, "r", errors="ignore") as f:
                while self.total_games_processed < self.max_games:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        if self.status_callback:
                            self.status_callback("🔍 Reached end of PGN file.")
                        break
                    headers = game.headers
                    white_elo_str, black_elo_str = headers.get("WhiteElo"), headers.get("BlackElo")
                    if not white_elo_str or not black_elo_str:
                        skipped_games +=1
                        continue
                    try:
                        white_elo, black_elo = int(white_elo_str), int(black_elo_str)
                    except ValueError:
                        skipped_games +=1
                        continue
                    if white_elo < self.min_elo or black_elo < self.min_elo:
                        skipped_games +=1
                        continue
                    game_str = str(game)
                    result = self._process_game(game_str)
                    if result is None:
                        skipped_games +=1
                        continue
                    self._process_data_entry(result)
                    self.total_games_processed +=1
            if self.batch_inputs:
                if self.status_callback:
                    self.status_callback("🔄 Writing remaining batch to H5 file.")
                self._write_batch_to_h5()
        if self.engine:
            self.engine.close()
            if self.status_callback:
                self.status_callback("🔍 Chess engine closed.")
        if self.status_callback:
            self.status_callback("🔍 Splitting dataset into train, validation, and test sets...")
        self._split_dataset()
        metrics = {"total_samples":self.total_samples, "total_games_processed":self.total_games_processed, "total_moves_processed":self.total_moves_processed, "game_results_counter":self.game_results_counter, "game_length_histogram":self.game_length_histogram.tolist(), "player_rating_histogram":self.player_rating_histogram.tolist(), "dataset_path":h5_path}
        if self.status_callback:
            self.status_callback(f"✅ Data Preparation completed successfully. Processed {self.total_games_processed} games with {skipped_games} skipped games in {time.time()-self.start_time:.2f} seconds.")
        return metrics

    def evaluate_position(self, board):
        if not self.engine:
            return 0.0
        limit = chess.engine.Limit(depth=self.engine_depth)
        try:
            info = self.engine.analyse(board, limit=limit)
            score = info["score"].pov(board.turn)
            if score.is_mate():
                mate_in = score.mate()
                return 1.0 if mate_in >0 else -1.0
            cp = score.score()
            return max(min(cp,1000),-1000)/1000.0
        except:
            if self.status_callback:
                self.status_callback(f"❌ Error evaluating position.")
            return 0.0

    def _process_game(self, game_str):
        try:
            game = chess.pgn.read_game(io.StringIO(game_str))
            if not game:
                if self.status_callback:
                    self.status_callback("ℹ️ No game found in PGN string.")
                return None
            headers = game.headers
            white_elo_str, black_elo_str = headers.get("WhiteElo"), headers.get("BlackElo")
            if not white_elo_str or not black_elo_str:
                if self.status_callback:
                    self.status_callback("ℹ️ Missing ELO ratings in game headers.")
                return None
            result_map = {"1-0":1.0, "0-1":-1.0, "1/2-1/2":0.0}
            game_result = result_map.get(headers.get("Result"), None)
            if game_result is None:
                if self.status_callback:
                    self.status_callback(f"ℹ️ Unrecognized game result: {headers.get('Result')}")
                return None
            white_elo, black_elo = int(white_elo_str), int(black_elo_str)
            avg_rating = (white_elo + black_elo)/2
            board, moves = game.board(), list(game.mainline_moves())
            inputs, policy_targets, value_targets = self._extract_move_data(board, moves)
            if not inputs:
                if self.status_callback:
                    self.status_callback("ℹ️ No valid inputs extracted from game.")
                return None
            return {"inputs":inputs, "policy_targets":policy_targets, "value_targets":value_targets, "game_length":len(moves), "avg_rating":avg_rating, "game_result":game_result}
        except:
            if self.status_callback:
                self.status_callback("❌ Exception in _process_game.")
            return None

    def _extract_move_data(self, board, moves):
        inputs, policy_targets, value_targets = [], [], []
        for move in moves:
            current_tensor = convert_board_to_tensor(board)
            move_idx = self.move_mapping.get_index_by_move(move)
            if move_idx is None:
                if self.status_callback:
                    self.status_callback(f"ℹ️ Move index not found for move: {move}")
                board.push(move)
                continue
            value_target = self.evaluate_position(board)
            inputs.append(current_tensor)
            policy_targets.append(move_idx)
            value_targets.append(value_target)
            flipped_board = flip_board(board)
            flipped_move = flip_move(move)
            flipped_move_idx = self.move_mapping.get_index_by_move(flipped_move)
            if flipped_move_idx is not None:
                inputs.append(convert_board_to_tensor(flipped_board))
                policy_targets.append(flipped_move_idx)
                value_targets.append(-value_target)
            board.push(move)
        return inputs, policy_targets, value_targets

    def _process_data_entry(self, data):
        self.batch_inputs.extend(data["inputs"])
        self.batch_policy_targets.extend(data["policy_targets"])
        self.batch_value_targets.extend(data["value_targets"])
        self.total_samples += len(data["inputs"])
        self.total_moves_processed += len(data["inputs"])
        self.game_results_counter[data["game_result"]] +=1
        self._update_histograms(data["game_length"], data["avg_rating"])
        if len(self.batch_inputs) >= self.batch_size:
            if self.progress_callback:
                self.progress_callback(int((self.total_games_processed / self.max_games) *100))
            if self.status_callback:
                self.status_callback(f"✅ Processed {self.total_games_processed}/{self.max_games} games.")
            self._write_batch_to_h5()

    def _write_batch_to_h5(self):
        batch_size = len(self.batch_inputs)
        start_idx, end_idx = self.current_dataset_size, self.current_dataset_size + batch_size
        self.h5_inputs.resize((end_idx,25,8,8))
        self.h5_policy_targets.resize((end_idx,))
        self.h5_value_targets.resize((end_idx,))
        self.h5_inputs[start_idx:end_idx] = np.array(self.batch_inputs, dtype=np.float32)
        self.h5_policy_targets[start_idx:end_idx] = np.array(self.batch_policy_targets, dtype=np.int64)
        self.h5_value_targets[start_idx:end_idx] = np.array(self.batch_value_targets, dtype=np.float32)
        self.current_dataset_size += batch_size
        self.batch_inputs, self.batch_policy_targets, self.batch_value_targets = [], [], []

    def _update_histograms(self, game_length, avg_rating):
        length_idx = np.digitize(game_length, self.game_length_bins)-1
        if 0 <= length_idx < len(self.game_length_histogram):
            self.game_length_histogram[length_idx] +=1
        if avg_rating:
            rating_idx = np.digitize(avg_rating, self.player_rating_bins)-1
            if 0 <= rating_idx < len(self.player_rating_histogram):
                self.player_rating_histogram[rating_idx] +=1

    def _split_dataset(self):
        h5_path = os.path.join(self.output_dir, "dataset.h5")
        train_indices_path, val_indices_path, test_indices_path = os.path.join(self.output_dir, "train_indices.npy"), os.path.join(self.output_dir, "val_indices.npy"), os.path.join(self.output_dir, "test_indices.npy")
        try:
            with h5py.File(h5_path, "r") as h5_file:
                num_samples = h5_file["inputs"].shape[0]
                if num_samples ==0:
                    if self.status_callback:
                        self.status_callback("❌ No samples to split.")
                    return
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                train_end, val_end = int(num_samples*0.8), int(num_samples*0.9)
                np.save(train_indices_path, indices[:train_end])
                np.save(val_indices_path, indices[train_end:val_end])
                np.save(test_indices_path, indices[val_end:])
                if self.status_callback:
                    self.status_callback(f"✅ Dataset split into Train ({train_end}), Validation ({val_end - train_end}), Test ({num_samples - val_end}) samples.")
        except:
            if self.status_callback:
                self.status_callback("❌ Error splitting dataset.")