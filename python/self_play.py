from __future__ import annotations

import contextlib
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chesscore as ccore
import config as C
import numpy as np

BOARD_SIZE = 8
NSQUARES = 64
PLANES_PER_POSITION = int(getattr(ccore, "PLANES_PER_POSITION", 14))
HISTORY_LENGTH = int(getattr(ccore, "HISTORY_LENGTH", 8))
POLICY_OUTPUT = int(getattr(ccore, "POLICY_SIZE", 73 * NSQUARES))

_MATERIAL_WEIGHTS = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)

SELFPLAY_TEMP_MOVES = int(C.SELFPLAY.TEMP_MOVES)
SELFPLAY_TEMP_HIGH = float(C.SELFPLAY.TEMP_HIGH)
SELFPLAY_TEMP_LOW = float(C.SELFPLAY.TEMP_LOW)
SELFPLAY_DETERMINISTIC_TEMP_EPS = float(C.SELFPLAY.DETERMINISTIC_TEMP_EPS)
SELFPLAY_COLOR_FLIP_PROB = float(getattr(C.SELFPLAY, "COLOR_FLIP_PROB", 0.0))
SELFPLAY_REP_NOISE_COUNT = max(0, int(getattr(C.SELFPLAY, "REPETITION_NOISE_COUNT", 0)))
SELFPLAY_REP_NOISE_WINDOW = max(0, int(getattr(C.SELFPLAY, "REPETITION_NOISE_WINDOW", 0)))

_DEFAULT_START_FEN_WHITE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_DEFAULT_START_FEN_BLACK = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"


class SelfPlayEngine:
    def __init__(self, evaluator: Any) -> None:
        self.resign_consecutive = C.RESIGN.CONSECUTIVE_PLIES
        self.evaluator = evaluator
        total_planes = PLANES_PER_POSITION * HISTORY_LENGTH + 7
        self._buf = ccore.ReplayBuffer(int(C.REPLAY.BUFFER_CAPACITY), int(total_planes), BOARD_SIZE, BOARD_SIZE)
        self.buffer_lock = threading.Lock()
        self.num_workers: int = int(C.SELFPLAY.NUM_WORKERS)
        self._enc_cache: dict[int, np.ndarray] = {}
        self._enc_cache_cap: int = 100_000
        base_prob = float(np.clip(SELFPLAY_COLOR_FLIP_PROB, 0.0, 1.0))
        self._color_flip_prob_base = base_prob if base_prob > 0.0 else 0.5
        self._color_bias = self._color_flip_prob_base

    def set_num_workers(self, n: int) -> None:
        self.num_workers = int(max(1, n))

    def get_num_workers(self) -> int:
        return int(self.num_workers)

    def set_color_bias(self, prob_black: float) -> None:
        self._color_bias = float(np.clip(prob_black, 0.1, 0.9))

    def get_color_bias(self) -> float:
        return float(self._color_bias)

    @staticmethod
    def _material_balance(position: ccore.Position) -> float:
        pieces = position.pieces
        balance = 0.0
        for idx, weight in enumerate(_MATERIAL_WEIGHTS):
            if weight == 0.0:
                continue
            white_bb = int(pieces[idx][0])
            black_bb = int(pieces[idx][1])
            balance += weight * (white_bb.bit_count() - black_bb.bit_count())
        return balance

    @staticmethod
    def _position_hash_key(position: ccore.Position) -> int | None:
        key_attr = getattr(position, "hash", None)
        try:
            raw = key_attr() if callable(key_attr) else key_attr
            if raw is None:
                return None
            if isinstance(raw, (int, np.integer)):
                return int(raw)
            if isinstance(raw, str):
                return int(raw)
        except Exception:
            return None
        return None

    @staticmethod
    def encode_u8(enc: np.ndarray) -> np.ndarray:
        x = np.clip(enc, 0.0, 1.0) * C.DATA.U8_SCALE
        return np.rint(x).astype(np.uint8, copy=False)

    @staticmethod
    def encode_value_i8(v: float) -> np.int8:
        return np.int8(
            np.clip(
                np.rint(v * C.DATA.VALUE_I8_SCALE),
                -int(C.DATA.VALUE_I8_SCALE),
                int(C.DATA.VALUE_I8_SCALE),
            )
        )

    def _select_move_by_temperature(
        self,
        moves: list[Any],
        visit_counts: list[int],
        move_number: int,
        rng: np.random.Generator | None = None,
    ) -> Any:
        temperature = SELFPLAY_TEMP_HIGH if move_number < SELFPLAY_TEMP_MOVES else SELFPLAY_TEMP_LOW
        if temperature > SELFPLAY_DETERMINISTIC_TEMP_EPS:
            probs = np.maximum(np.array(visit_counts, dtype=np.float64), 0.0)
            s = probs.sum()
            if not np.isfinite(s) or s <= 0:
                idx = int(np.argmax(visit_counts))
            else:
                probs = probs ** (1.0 / temperature)
                s = probs.sum()
                if (not np.isfinite(s)) or (s <= 0):
                    idx = int(np.argmax(visit_counts))
                elif rng is None:
                    idx = int(np.random.choice(len(moves), p=probs / s))
                else:
                    idx = int(rng.choice(len(moves), p=(probs / s)))
        else:
            idx = int(np.argmax(visit_counts))
        return moves[idx]

    def _process_result(
        self,
        examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        result: int,
        first_to_move_is_white: bool,
    ) -> None:
        if result == ccore.WHITE_WIN:
            base = 1.0
        elif result == ccore.BLACK_WIN:
            base = -1.0
        else:
            base = 0.0
        with self.buffer_lock:
            for ply_index, (position_u8, idx_i32, counts_u16) in enumerate(examples):
                stm_is_white = ((ply_index % 2) == 0) == bool(first_to_move_is_white)
                target_value = base if stm_is_white else -base
                self._buf.push(position_u8, idx_i32, counts_u16, SelfPlayEngine.encode_value_i8(target_value))

    def play_single_game(
        self, seed: int | None = None
    ) -> tuple[int, int, list[tuple[np.ndarray, np.ndarray, np.ndarray]], bool, dict[str, int]]:
        position = ccore.Position()
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        try:
            prob_black = float(np.clip(self._color_bias, 0.0, 1.0))
            if SELFPLAY_COLOR_FLIP_PROB <= 0.0 and self._color_flip_prob_base > 0.0:
                prob_black = self._color_flip_prob_base
            flip_to_black = prob_black > 0.0 and float(rng.random()) < prob_black
            start_fen = _DEFAULT_START_FEN_BLACK if flip_to_black else _DEFAULT_START_FEN_WHITE
            position.from_fen(start_fen)
        except Exception:
            position = ccore.Position()

        resign_count = 0
        forced_result: int | None = None
        termination_reason = "natural"
        mcts = ccore.MCTS(
            C.MCTS.TRAIN_SIMULATIONS_BASE,
            C.MCTS.C_PUCT,
            C.MCTS.DIRICHLET_ALPHA,
            C.MCTS.DIRICHLET_WEIGHT,
        )
        mcts.set_c_puct_params(C.MCTS.C_PUCT_BASE, C.MCTS.C_PUCT_INIT)
        mcts.set_fpu_reduction(C.MCTS.FPU_REDUCTION)
        mcts.seed(int(rng.integers(2**63 - 1)))

        examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        move_count = 0
        loop_breaks = 0
        repetition_hits = 0
        repetition_counts: dict[int, int] = {}
        key_history: deque[int] | None = None
        if SELFPLAY_REP_NOISE_WINDOW > 0:
            key_history = deque(maxlen=SELFPLAY_REP_NOISE_WINDOW)

        random_opening_plies = int(rng.integers(0, max(1, C.SELFPLAY.OPENING_RANDOM_PLIES_MAX)))
        for _ in range(random_opening_plies):
            if position.result() != ccore.ONGOING:
                break
            moves = position.legal_moves()
            if not moves:
                break
            position.make_move(moves[int(rng.integers(0, len(moves)))])

        first_to_move_is_white: bool | None = None
        while position.result() == ccore.ONGOING and move_count < C.SELFPLAY.GAME_MAX_PLIES:
            pos_snapshot = ccore.Position(position)

            sims = max(
                C.MCTS.TRAIN_SIMULATIONS_MIN,
                C.MCTS.TRAIN_SIMULATIONS_BASE // (1 + move_count // C.MCTS.TRAIN_SIM_DECAY_MOVE_INTERVAL),
            )
            mcts.set_simulations(sims)

            if move_count < C.SELFPLAY.TEMP_MOVES:
                mcts.set_dirichlet_params(C.MCTS.DIRICHLET_ALPHA, C.MCTS.DIRICHLET_WEIGHT)
            else:
                mcts.set_dirichlet_params(C.MCTS.DIRICHLET_ALPHA, 0.0)

            repetition_count = 0
            pos_key = SelfPlayEngine._position_hash_key(position)
            if pos_key is not None:
                if key_history is not None and key_history.maxlen is not None:
                    if len(key_history) == key_history.maxlen:
                        old_key = key_history.popleft()
                        prev = repetition_counts.get(old_key, 0) - 1
                        if prev <= 0:
                            repetition_counts.pop(old_key, None)
                        else:
                            repetition_counts[old_key] = prev
                    key_history.append(pos_key)
                repetition_counts[pos_key] = repetition_counts.get(pos_key, 0) + 1
                repetition_count = repetition_counts[pos_key]

            moves = position.legal_moves()
            if not moves:
                break
            visit_counts = mcts.search_batched_legal(
                position, self.evaluator.infer_positions_legal, C.EVAL.BATCH_SIZE_MAX
            )
            if not visit_counts or len(visit_counts) != len(moves):
                break

            idx_list: list[int] = []
            cnt_list: list[int] = []
            for mv, vc in zip(moves, visit_counts, strict=False):
                move_index = ccore.encode_move_index(mv)
                if (move_index is not None) and (0 <= int(move_index) < POLICY_OUTPUT):
                    c = min(int(vc), C.MCTS.VISIT_COUNT_CLAMP)
                    if c > 0:
                        idx_list.append(int(move_index))
                        cnt_list.append(int(c))
            idx_arr = np.asarray(idx_list, dtype=np.int32)
            cnt_arr = np.asarray(cnt_list, dtype=np.uint16)
            key = SelfPlayEngine._position_hash_key(pos_snapshot)
            if key is None:
                key = 0
            if key in self._enc_cache:
                enc_u8 = self._enc_cache.pop(key)
                self._enc_cache[key] = enc_u8
            else:
                enc = ccore.encode_position(pos_snapshot)
                enc_u8 = SelfPlayEngine.encode_u8(enc)
                self._enc_cache[key] = enc_u8
                if len(self._enc_cache) > self._enc_cache_cap:
                    try:
                        oldest = next(iter(self._enc_cache.keys()))
                        self._enc_cache.pop(oldest, None)
                    except Exception:
                        self._enc_cache.clear()
            examples.append((enc_u8, idx_arr, cnt_arr))
            if first_to_move_is_white is None:
                first_to_move_is_white = position.turn == ccore.WHITE

            if C.RESIGN.ENABLED and self.resign_consecutive > 0:
                val_arr = self.evaluator.infer_values([pos_snapshot])
                value_estimate = float(val_arr[0])
                if value_estimate <= C.RESIGN.VALUE_THRESHOLD:
                    resign_count += 1
                    if resign_count >= self.resign_consecutive:
                        if float(rng.random()) < C.RESIGN.PLAYTHROUGH_FRACTION:
                            resign_count = 0
                        else:
                            side_to_move_is_white = position.turn == ccore.WHITE
                            forced_result = ccore.BLACK_WIN if side_to_move_is_white else ccore.WHITE_WIN
                            termination_reason = "resign"
                            break
                else:
                    resign_count = 0

            inject_noise = (
                SELFPLAY_REP_NOISE_COUNT > 0
                and repetition_count >= SELFPLAY_REP_NOISE_COUNT
                and len(moves) > 1
            )
            if inject_noise:
                visit_arr = np.asarray(visit_counts, dtype=np.float64)
                top_k = min(len(moves), 4)
                try:
                    top_indices = np.argsort(visit_arr)[-top_k:]
                except Exception:
                    top_indices = np.arange(len(moves))
                chosen_idx = int(rng.choice(top_indices))
                move = moves[chosen_idx]
                loop_breaks += 1
                repetition_hits += 1
            else:
                move = self._select_move_by_temperature(moves, visit_counts, move_count, rng=rng)
            position.make_move(move)

            with contextlib.suppress(Exception):
                mcts.advance_root(position, move)
            move_count += 1

        if forced_result is None and move_count >= C.SELFPLAY.GAME_MAX_PLIES:
            termination_reason = "exhausted"

        final_result = forced_result if forced_result is not None else position.result()
        if forced_result is None and final_result == ccore.ONGOING:
            termination_reason = "exhausted"
            final_result = ccore.DRAW

        meta = {
            "loop_breaks": int(loop_breaks),
            "repetition_hits": int(repetition_hits),
            "termination": termination_reason,
        }

        return (
            move_count,
            final_result,
            examples,
            True if first_to_move_is_white is None else bool(first_to_move_is_white),
            meta,
        )

    def get_capacity(self) -> int:
        with self.buffer_lock:
            return int(self._buf.capacity)

    def set_capacity(self, capacity: int) -> None:
        cap = int(max(1, capacity))
        with self.buffer_lock:
            self._buf.set_capacity(cap)

    def size(self) -> int:
        with self.buffer_lock:
            return int(self._buf.size)

    def _record_game(
        self,
        stats: dict[str, Any],
        game_data: tuple[int, int, list[tuple[np.ndarray, np.ndarray, np.ndarray]], bool, dict[str, int]],
    ) -> None:
        move_count, result, examples, first_to_move_is_white, meta = game_data
        self._process_result(examples, result, bool(first_to_move_is_white))
        stats["games"] += 1
        stats["moves"] += move_count
        if bool(first_to_move_is_white):
            stats["starts_white"] += 1
        else:
            stats["starts_black"] += 1
        meta = meta or {}
        stats["loop_breaks"] += int(meta.get("loop_breaks", 0))
        stats["repetition_hits"] += int(meta.get("repetition_hits", 0))
        term_reason = str(meta.get("termination", "natural"))
        if term_reason == "resign":
            stats["term_resign"] += 1
        elif term_reason == "exhausted":
            stats["term_exhausted"] += 1
        elif term_reason == "threefold":
            stats["term_threefold"] += 1
        elif term_reason == "fifty_move":
            stats["term_fifty"] += 1
        else:
            stats["term_natural"] += 1
        if result == ccore.WHITE_WIN:
            stats["white_wins"] += 1
        elif result == ccore.BLACK_WIN:
            stats["black_wins"] += 1
        else:
            if result == ccore.DRAW:
                if term_reason in {"exhausted", "fifty_move", "threefold"}:
                    stats["draws_cap"] += 1
                else:
                    stats["draws_true"] += 1
            else:
                stats["draws_cap"] += 1
            stats["draws"] += 1

    def sample_batch(
        self,
        batch_size: int,
        recent_ratio: float = C.SAMPLING.TRAIN_RECENT_SAMPLE_RATIO,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.int8]] | None:
        with self.buffer_lock:
            if batch_size > int(self._buf.size):
                return None
            states_u8, idx_list, cnt_list, values_i8 = self._buf.sample(
                int(batch_size), float(recent_ratio), float(C.SAMPLING.REPLAY_SNAPSHOT_RECENT_WINDOW_FRAC)
            )
        states = [states_u8[i] for i in range(states_u8.shape[0])]
        indices_sparse = [np.asarray(idx_list[i], dtype=np.int32) for i in range(len(idx_list))]
        counts_sparse = [np.asarray(cnt_list[i], dtype=np.uint16) for i in range(len(cnt_list))]
        values = [np.int8(values_i8[i]) for i in range(values_i8.shape[0])]
        return states, indices_sparse, counts_sparse, values

    def play_games(self, num_games: int) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "games": 0,
            "moves": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "draws_true": 0,
            "draws_cap": 0,
            "starts_white": 0,
            "starts_black": 0,
            "loop_breaks": 0,
            "repetition_hits": 0,
            "term_natural": 0,
            "term_resign": 0,
            "term_exhausted": 0,
            "term_threefold": 0,
            "term_fifty": 0,
        }
        seeds = [int(np.random.randint(0, 2**63 - 1)) for _ in range(num_games)]
        deterministic = int(C.SEED) != 0
        pending: dict[int, tuple[int, int, list[tuple[np.ndarray, np.ndarray, np.ndarray]], bool, dict[str, int]]] = {}
        with ThreadPoolExecutor(max_workers=max(1, self.num_workers)) as ex:
            futures = {ex.submit(self.play_single_game, seeds[i]): i for i in range(num_games)}
            for fut in as_completed(futures):
                idx = futures[fut]
                game_data = fut.result()
                if deterministic:
                    pending[idx] = game_data
                else:
                    self._record_game(stats, game_data)
        if deterministic:
            for idx in range(num_games):
                if idx in pending:
                    self._record_game(stats, pending[idx])
        stats["color_bias_prob_black"] = float(self._color_bias)
        return stats
