from __future__ import annotations

import contextlib
import math
import threading
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Integral
from typing import Any, cast

import chesscore as ccore
import config as C
import numpy as np
from fen_tools import flip_fen_perspective, sanitize_fen

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
SELFPLAY_REP_AVOID_TOP_K = max(0, int(getattr(C.SELFPLAY, "REPETITION_AVOID_TOP_K", 1)))
SELFPLAY_DIRICHLET_LATE = float(getattr(C.SELFPLAY, "DIRICHLET_WEIGHT_LATE", C.MCTS.DIRICHLET_WEIGHT))
SELFPLAY_DIRICHLET_HALFMOVE = int(getattr(C.SELFPLAY, "DIRICHLET_HALFMOVE_THRESHOLD", 0))

_CURRICULUM_PROB = float(getattr(C.SELFPLAY, "CURRICULUM_SAMPLE_PROB", 0.0))
_CURRICULUM_FENS: tuple[str, ...] = tuple(
    sanitize_fen(str(fen)) for fen in getattr(C.SELFPLAY, "CURRICULUM_FENS", ())
)


def _build_opening_book() -> tuple[tuple[tuple[str, float], ...], float]:
    entries: list[tuple[str, float]] = []
    total_weight = 0.0
    raw_obj = getattr(C.SELFPLAY, "OPENING_BOOK", ())
    candidates = tuple(raw_obj) if isinstance(raw_obj, (list, tuple)) else (raw_obj,)
    for raw in candidates:
        if raw in {None, ()}:
            continue
        if isinstance(raw, (tuple, list)) and len(raw) >= 2:
            fen_raw, weight_raw = raw[0], raw[1]
        else:
            fen_raw, weight_raw = raw, 1.0
        try:
            fen = sanitize_fen(str(fen_raw))
        except Exception:
            continue
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError):
            weight = 1.0
        weight = max(weight, 0.0)
        if weight == 0.0:
            continue
        entries.append((fen, weight))
        total_weight += weight
    return tuple(entries), float(total_weight)


_OPENING_BOOK, _OPENING_BOOK_WEIGHT = _build_opening_book()


def _sample_opening_fen(rng: np.random.Generator, flip_to_black: bool) -> str:
    if _OPENING_BOOK and _OPENING_BOOK_WEIGHT > 0.0:
        draw = float(rng.random()) * _OPENING_BOOK_WEIGHT
        accum = 0.0
        for fen, weight in _OPENING_BOOK:
            accum += weight
            if draw <= accum:
                chosen = fen
                break
        else:
            chosen = _OPENING_BOOK[-1][0]
        return flip_fen_perspective(chosen) if flip_to_black else chosen
    base = _DEFAULT_START_FEN_BLACK if flip_to_black else _DEFAULT_START_FEN_WHITE
    return base

_DEFAULT_START_FEN_WHITE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_DEFAULT_START_FEN_BLACK = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"


class SelfPlayEngine:
    def __init__(self, evaluator: Any) -> None:
        self.resign_consecutive = C.RESIGN.CONSECUTIVE_PLIES
        self.evaluator = evaluator
        total_planes = PLANES_PER_POSITION * HISTORY_LENGTH + 7
        self._total_planes = int(total_planes)
        self._buf = ccore.ReplayBuffer(int(C.REPLAY.BUFFER_CAPACITY), int(total_planes), BOARD_SIZE, BOARD_SIZE)
        self.buffer_lock = threading.Lock()
        self.num_workers: int = int(C.SELFPLAY.NUM_WORKERS)
        self._enc_cache: dict[int, np.ndarray] = {}
        cache_cap = int(getattr(C.SELFPLAY, "ENCODE_CACHE_CAP", 100_000))
        try:
            buf_cap = int(getattr(C.REPLAY, "BUFFER_CAPACITY", cache_cap))
        except Exception:
            buf_cap = cache_cap
        if buf_cap > 0:
            cache_cap = min(cache_cap, buf_cap)
        self._enc_cache_cap: int = max(1_000, cache_cap)
        base_prob = float(np.clip(SELFPLAY_COLOR_FLIP_PROB, 0.0, 1.0))
        self._color_flip_prob_base = base_prob if base_prob > 0.0 else 0.5
        self._color_bias = self._color_flip_prob_base
        _resign_default = float(getattr(C.RESIGN, "VALUE_THRESHOLD", -0.15))
        self.resign_threshold = float(getattr(C.RESIGN, "VALUE_THRESHOLD_INIT", _resign_default))
        self.resign_min_plies = int(max(0, getattr(C.RESIGN, "MIN_PLIES_INIT", 0)))
        self.resign_eval_margin = float(max(0.0, getattr(C.RESIGN, "EVAL_MARGIN", abs(self.resign_threshold))))
        self.resign_eval_persist = int(max(1, getattr(C.RESIGN, "EVAL_PERSIST_PLIES", self.resign_consecutive)))
        self.adjudicate_enabled = bool(getattr(C.SELFPLAY, "ADJUDICATE_ENABLED", False))
        self.adjudicate_margin = float(getattr(C.SELFPLAY, "ADJUDICATE_MATERIAL_MARGIN", 0.0))
        self.adjudicate_min_plies = int(max(0, getattr(C.SELFPLAY, "ADJUDICATE_MIN_PLIES", 0)))
        self._adjudicate_margin_start = float(
            getattr(C.SELFPLAY, "ADJUDICATE_MARGIN_START", self.adjudicate_margin)
        )
        self._adjudicate_margin_end = float(
            getattr(C.SELFPLAY, "ADJUDICATE_MARGIN_END", self.adjudicate_margin)
        )
        self._adjudicate_margin_decay_iter = max(
            1, int(getattr(C.SELFPLAY, "ADJUDICATE_MARGIN_DECAY_ITER", 1))
        )
        self._adjudicate_min_start = int(
            max(0, getattr(C.SELFPLAY, "ADJUDICATE_MIN_PLIES_START", self.adjudicate_min_plies))
        )
        self._adjudicate_min_end = int(
            max(0, getattr(C.SELFPLAY, "ADJUDICATE_MIN_PLIES_END", self.adjudicate_min_plies))
        )
        self._adjudicate_min_decay_iter = max(
            1, int(getattr(C.SELFPLAY, "ADJUDICATE_MIN_PLIES_DECAY_ITER", 1))
        )
        self._adjudicate_value_margin_start = float(
            getattr(
                C.SELFPLAY,
                "ADJUDICATE_VALUE_MARGIN_START",
                getattr(C.SELFPLAY, "ADJUDICATE_VALUE_MARGIN", 1.0),
            )
        )
        self._adjudicate_value_margin_end = float(
            getattr(
                C.SELFPLAY,
                "ADJUDICATE_VALUE_MARGIN_END",
                self._adjudicate_value_margin_start,
            )
        )
        self._adjudicate_value_decay_iter = max(
            1, int(getattr(C.SELFPLAY, "ADJUDICATE_VALUE_DECAY_ITER", 1))
        )
        self.adjudicate_value_margin = float(self._adjudicate_value_margin_start)
        self.adjudicate_value_persist = int(
            max(1, getattr(C.SELFPLAY, "ADJUDICATE_VALUE_PERSIST_PLIES", 2))
        )
        self.game_max_plies = int(max(1, getattr(C.SELFPLAY, "GAME_MAX_PLIES", 160)))
        self.endgame_sim_moves = int(max(0, getattr(C.SELFPLAY, "ENDGAME_SIM_MOVES", 0)))
        self.endgame_sim_factor = float(max(1.0, getattr(C.SELFPLAY, "ENDGAME_SIM_FACTOR", 1.0)))
        self.endgame_material_trigger = float(
            max(0.0, getattr(C.SELFPLAY, "ENDGAME_MATERIAL_TRIGGER", 0.0))
        )

        natural_weight = float(getattr(C.SAMPLING, "NATURAL_DUPLICATE_WEIGHT", 1.0))
        exhaust_weight = float(getattr(C.SAMPLING, "EXHAUST_DUPLICATE_WEIGHT", 1.0))
        adjud_weight = float(getattr(C.SAMPLING, "ADJUDICATED_DUPLICATE_WEIGHT", 1.0))
        natural_dup = max(1, round(natural_weight))
        exhaust_dup = max(1, round(exhaust_weight))
        adjud_dup = max(1, round(adjud_weight))
        self._dup_weights = {
            "natural": natural_dup,
            "resign": natural_dup,
            "threefold": exhaust_dup,
            "fifty_move": exhaust_dup,
            "exhausted": exhaust_dup,
            "adjudicated": adjud_dup,
        }

    def set_num_workers(self, n: int) -> None:
        self.num_workers = int(max(1, n))

    def get_num_workers(self) -> int:
        return int(self.num_workers)

    def set_color_bias(self, prob_black: float) -> None:
        self._color_bias = float(np.clip(prob_black, 0.3, 0.7))

    def get_color_bias(self) -> float:
        return float(self._color_bias)

    def set_resign_params(self, threshold: float | None = None, min_plies: int | None = None) -> None:
        if threshold is not None:
            self.resign_threshold = float(threshold)
            self.resign_eval_margin = float(
                max(0.0, min(abs(self.resign_threshold), getattr(C.RESIGN, "EVAL_MARGIN", abs(self.resign_threshold))))
            )
        if min_plies is not None:
            self.resign_min_plies = int(max(0, min_plies))

    def set_game_length(self, max_plies: int) -> None:
        self.game_max_plies = int(max(1, max_plies))

    def get_game_length(self) -> int:
        return int(self.game_max_plies)

    def update_adjudication(self, iteration: int) -> None:
        if not self.adjudicate_enabled:
            return
        it = max(0, int(iteration))
        margin_alpha = min(1.0, it / max(1, self._adjudicate_margin_decay_iter))
        margin = (
            self._adjudicate_margin_start
            + margin_alpha * (self._adjudicate_margin_end - self._adjudicate_margin_start)
        )
        min_alpha = min(1.0, it / max(1, self._adjudicate_min_decay_iter))
        min_plies = round(
            self._adjudicate_min_start
            + min_alpha * (self._adjudicate_min_end - self._adjudicate_min_start)
        )
        self.adjudicate_margin = float(max(0.0, margin))
        self.adjudicate_min_plies = int(max(0, min_plies))
        value_alpha = min(1.0, it / max(1, self._adjudicate_value_decay_iter))
        self.adjudicate_value_margin = float(
            self._adjudicate_value_margin_start
            + value_alpha * (self._adjudicate_value_margin_end - self._adjudicate_value_margin_start)
        )


    def get_adjudication_params(self) -> tuple[float, int]:
        if not self.adjudicate_enabled:
            return 0.0, 0
        return float(self.adjudicate_margin), int(self.adjudicate_min_plies)

    @staticmethod
    def _material_balance(position: ccore.Position) -> float:
        pieces_obj = getattr(position, "pieces", None)
        balance = 0.0
        if pieces_obj is None:
            return balance
        try:
            pieces = cast(Sequence[tuple[Any, Any]], pieces_obj)
        except Exception:
            return balance
        for idx, weight in enumerate(_MATERIAL_WEIGHTS):
            if weight == 0.0:
                continue
            try:
                white_entry = pieces[idx][0]
                black_entry = pieces[idx][1]
            except Exception:
                continue
            if isinstance(white_entry, Integral):
                white_bb = int(white_entry)
            else:
                continue
            if isinstance(black_entry, Integral):
                black_bb = int(black_entry)
            else:
                continue
            balance += weight * (white_bb.bit_count() - black_bb.bit_count())
        return balance

    @staticmethod
    def _position_hash_key(position: ccore.Position) -> int | None:
        key_attr = getattr(position, "hash", None)
        try:
            raw = key_attr() if callable(key_attr) else key_attr
        except Exception:
            return None
        if raw is None:
            return None
        if isinstance(raw, Integral):
            return int(raw)
        if isinstance(raw, str):
            return int(raw)
        return None

    @staticmethod
    def _material_total(position: ccore.Position) -> float:
        pieces_obj = getattr(position, "pieces", None)
        total = 0.0
        if pieces_obj is None:
            return total
        try:
            pieces = cast(Sequence[tuple[Any, Any]], pieces_obj)
        except Exception:
            return total
        for idx, weight in enumerate(_MATERIAL_WEIGHTS):
            if weight == 0.0:
                continue
            try:
                white_entry = pieces[idx][0]
                black_entry = pieces[idx][1]
            except Exception:
                continue
            if isinstance(white_entry, Integral):
                total += weight * int(white_entry).bit_count()
            if isinstance(black_entry, Integral):
                total += weight * int(black_entry).bit_count()
        return total

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
        value_offset: float = 0.0,
        termination: str | None = None,
    ) -> None:
        if result == ccore.WHITE_WIN:
            base = 1.0
        elif result == ccore.BLACK_WIN:
            base = -1.0
        else:
            base = 0.0
        dup_factor = int(self._dup_weights.get(str(termination or "").lower(), 1))
        dup_factor = max(1, dup_factor)
        with self.buffer_lock:
            for ply_index, (position_u8, idx_i32, counts_u16) in enumerate(examples):
                stm_is_white = ((ply_index % 2) == 0) == bool(first_to_move_is_white)
                target_value = base if stm_is_white else -base
                target_value += value_offset
                target_value = float(np.clip(target_value, -1.0, 1.0))
                value_i8 = SelfPlayEngine.encode_value_i8(target_value)
                for _ in range(dup_factor):
                    self._buf.push(position_u8, idx_i32, counts_u16, value_i8)

    def play_single_game(
        self, seed: int | None = None
    ) -> tuple[int, int, list[tuple[np.ndarray, np.ndarray, np.ndarray]], bool, dict[str, int]]:
        position = ccore.Position()
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        tempo_baseline = float(
            getattr(C.SELFPLAY, "TEMPO_BASELINE_PLIES", max(80, int(self.game_max_plies * 0.75)))
        )
        curriculum_seed = False
        flip_to_black = False
        try:
            prob_black = float(np.clip(self._color_bias, 0.0, 1.0))
            if SELFPLAY_COLOR_FLIP_PROB <= 0.0 and self._color_flip_prob_base > 0.0:
                prob_black = self._color_flip_prob_base
            flip_to_black = prob_black > 0.0 and float(rng.random()) < prob_black
            start_fen: str
            if _CURRICULUM_FENS and float(rng.random()) < _CURRICULUM_PROB:
                curriculum_seed = True
                base_fen = str(rng.choice(_CURRICULUM_FENS))
                start_fen = flip_fen_perspective(base_fen) if flip_to_black else base_fen
            else:
                start_fen = _sample_opening_fen(rng, flip_to_black)
            position.from_fen(start_fen)
        except Exception:
            position = ccore.Position()
            try:
                fallback_fen = _sample_opening_fen(rng, flip_to_black)
                position.from_fen(fallback_fen)
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
        seen_hashes: dict[int, int] = {}
        initial_key = SelfPlayEngine._position_hash_key(position)
        if initial_key is not None:
            seen_hashes[initial_key] = 1
        threefold_detected = False
        halfmove_sum = 0.0
        halfmove_max = 0
        value_trend_side = 0
        value_trend_count = 0
        eval_window: deque[float] = deque(maxlen=max(1, self.adjudicate_value_persist))
        eval_samples: list[float] = []
        sims_last = int(C.MCTS.TRAIN_SIMULATIONS_BASE)

        while position.result() == ccore.ONGOING and move_count < self.game_max_plies:
            pos_snapshot = ccore.Position(position)

            sims = max(
                C.MCTS.TRAIN_SIMULATIONS_MIN,
                C.MCTS.TRAIN_SIMULATIONS_BASE // (1 + max(0, move_count) // C.MCTS.TRAIN_SIM_DECAY_MOVE_INTERVAL),
            )
            if self.endgame_sim_moves and move_count >= self.endgame_sim_moves:
                sims = math.ceil(sims * self.endgame_sim_factor)
            if self.endgame_material_trigger > 0.0:
                material_total = SelfPlayEngine._material_total(position)
                if material_total <= self.endgame_material_trigger:
                    sims = math.ceil(sims * self.endgame_sim_factor)
            mcts.set_simulations(sims)
            sims_last = sims

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
                hits = repetition_counts.get(pos_key, 0) + 1
                repetition_counts[pos_key] = hits
                repetition_count = hits
                if hits >= 3:
                    threefold_detected = True

            moves = position.legal_moves()
            if not moves:
                break
            visit_counts_raw = mcts.search_batched_legal(
                position, self.evaluator.infer_positions_legal, C.EVAL.BATCH_SIZE_MAX
            )
            if not visit_counts_raw or len(visit_counts_raw) != len(moves):
                break

            visit_counts_arr = np.asarray(visit_counts_raw, dtype=np.float64)
            visit_counts = visit_counts_arr.tolist()

            idx_list: list[int] = []
            cnt_list: list[int] = []
            move_strings: list[str] = []
            for mv, vc in zip(moves, visit_counts, strict=False):
                move_index = ccore.encode_move_index(mv)
                if (move_index is not None) and (0 <= int(move_index) < POLICY_OUTPUT):
                    c = min(int(vc), C.MCTS.VISIT_COUNT_CLAMP)
                    if c > 0:
                        idx_list.append(int(move_index))
                        cnt_list.append(int(c))
                with contextlib.suppress(Exception):
                    move_strings.append(str(mv))
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

            val_arr = self.evaluator.infer_values([pos_snapshot])
            value_estimate = float(val_arr[0])
            if pos_snapshot.turn == ccore.BLACK:
                value_estimate = -value_estimate
            eval_samples.append(value_estimate)
            if abs(value_estimate) >= self.adjudicate_value_margin:
                trend_side = 1 if value_estimate > 0 else -1
                if trend_side == value_trend_side:
                    value_trend_count += 1
                else:
                    value_trend_side = trend_side
                    value_trend_count = 1
            else:
                value_trend_side = 0
                value_trend_count = 0
            eval_window.append(value_estimate)

            if (
                C.RESIGN.ENABLED
                and self.resign_consecutive > 0
                and move_count >= int(max(0, self.resign_min_plies))
            ):
                threshold = getattr(self, "resign_threshold", C.RESIGN.VALUE_THRESHOLD)
                if value_estimate <= threshold:
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

                if (
                    value_trend_count >= self.resign_eval_persist
                    and abs(value_estimate) >= self.resign_eval_margin
                ):
                    white_to_move = position.turn == ccore.WHITE
                    if value_estimate >= self.resign_eval_margin and not white_to_move:
                        forced_result = ccore.WHITE_WIN
                        termination_reason = "resign"
                        break
                    if value_estimate <= -self.resign_eval_margin and white_to_move:
                        forced_result = ccore.BLACK_WIN
                        termination_reason = "resign"
                        break

            halfmove_clock = 0
            with contextlib.suppress(Exception):
                halfmove_clock = int(getattr(pos_snapshot, "halfmove", 0))
            halfmove_sum += float(halfmove_clock)
            halfmove_max = max(halfmove_max, halfmove_clock)

            dirichlet_weight = C.MCTS.DIRICHLET_WEIGHT
            if move_count >= C.SELFPLAY.TEMP_MOVES:
                dirichlet_weight = max(SELFPLAY_DIRICHLET_LATE, 0.0)
            if SELFPLAY_DIRICHLET_HALFMOVE > 0 and halfmove_clock >= SELFPLAY_DIRICHLET_HALFMOVE:
                dirichlet_weight = max(dirichlet_weight, SELFPLAY_DIRICHLET_LATE)
            if repetition_count >= SELFPLAY_REP_NOISE_COUNT and SELFPLAY_DIRICHLET_LATE > 0.0:
                dirichlet_weight = max(dirichlet_weight, 0.5 * C.MCTS.DIRICHLET_WEIGHT)
            mcts.set_dirichlet_params(C.MCTS.DIRICHLET_ALPHA, dirichlet_weight)

            move = self._select_move_by_temperature(moves, visit_counts, move_count, rng=rng)
            position.make_move(move)

            next_key = SelfPlayEngine._position_hash_key(position)
            if next_key is not None:
                seen_hashes[next_key] = seen_hashes.get(next_key, 0) + 1

            with contextlib.suppress(Exception):
                mcts.advance_root(position, move)
            move_count += 1

            if threefold_detected:
                forced_result = ccore.DRAW
                termination_reason = "threefold"
                break

        adjudicated = False
        adjudicated_balance = 0.0
        adjudicate_enabled = bool(getattr(self, "adjudicate_enabled", False))
        adjudicate_margin = float(self.adjudicate_margin if adjudicate_enabled else 0.0)
        adjudicate_min_plies = int(self.adjudicate_min_plies if adjudicate_enabled else 0)
        terminal_eval = 0.0
        try:
            terminal_eval = float(self.evaluator.infer_values([position])[0])
            if position.turn == ccore.BLACK:
                terminal_eval = -terminal_eval
            eval_samples.append(terminal_eval)
            eval_window.append(terminal_eval)
            if abs(terminal_eval) >= self.adjudicate_value_margin:
                trend_side_terminal = 1 if terminal_eval > 0 else -1
                if trend_side_terminal == value_trend_side:
                    value_trend_count += 1
                else:
                    value_trend_side = trend_side_terminal
                    value_trend_count = 1
            else:
                value_trend_side = 0
                value_trend_count = 0
        except Exception:
            terminal_eval = 0.0

        if forced_result is None and move_count >= self.game_max_plies:
            termination_reason = "exhausted"

        final_result = forced_result if forced_result is not None else position.result()
        if forced_result is None and final_result == ccore.ONGOING:
            termination_reason = "exhausted"
            final_result = ccore.DRAW

        if adjudicate_enabled and forced_result is None and termination_reason == "exhausted":
            balance = SelfPlayEngine._material_balance(position)
            value_consistent = (
                value_trend_side != 0
                and value_trend_count >= self.adjudicate_value_persist
                and abs(terminal_eval) >= self.adjudicate_value_margin
            )
            if value_consistent:
                adjudicated = True
                winner_is_white = (
                    value_trend_side > 0 if position.turn == ccore.WHITE else value_trend_side < 0
                )
                final_result = ccore.WHITE_WIN if winner_is_white else ccore.BLACK_WIN
                termination_reason = "adjudicated"
                adjudicated_balance = float(np.clip(terminal_eval, -1.0, 1.0))
            elif adjudicate_margin > 0.0 and move_count >= max(adjudicate_min_plies, 1):
                if abs(balance) >= adjudicate_margin:
                    adjudicated = True
                    final_result = ccore.WHITE_WIN if balance > 0 else ccore.BLACK_WIN
                    termination_reason = "adjudicated"
                    adjudicated_balance = float(balance)

        value_shift = 0.0
        if final_result == ccore.DRAW and termination_reason in {"exhausted", "threefold", "fifty_move"}:
            scale = float(getattr(C.SELFPLAY, "EXHAUSTION_VALUE_SCALE", 0.0))
            max_shift = float(getattr(C.SELFPLAY, "EXHAUSTION_VALUE_MAX", 0.0))
            if scale > 0.0 and max_shift > 0.0:
                candidate = terminal_eval
                if abs(candidate) < 1e-6:
                    candidate = SelfPlayEngine._material_balance(position) * 0.1
                if abs(candidate) > 1e-6:
                    value_shift = float(np.clip(candidate * scale, -max_shift, max_shift))

        tempo_bonus = 0.0
        if termination_reason in {"natural", "resign"}:
            tempo_delta = tempo_baseline - float(move_count)
            if abs(tempo_delta) > 1.0:
                tempo_raw = np.clip(tempo_delta / max(tempo_baseline, 1.0), -1.0, 1.0)
                tempo_bonus = float(np.clip(tempo_raw * 0.25, -0.25, 0.25))
                if final_result == ccore.WHITE_WIN:
                    value_shift += tempo_bonus
                elif final_result == ccore.BLACK_WIN:
                    value_shift -= tempo_bonus

        if value_shift != 0.0:
            value_shift = float(np.clip(value_shift, -0.5, 0.5))

        meta = {
            "termination": termination_reason,
            "unique_positions": int(len(seen_hashes) or (1 if move_count > 0 else 0)),
            "halfmove_max": int(halfmove_max),
            "halfmove_avg": float(halfmove_sum / max(1, move_count)),
            "threefold": 1 if threefold_detected else 0,
            "curriculum_seed": 1 if curriculum_seed else 0,
            "terminal_eval": float(terminal_eval),
            "value_trend_len": int(value_trend_count),
            "value_trend_side": int(value_trend_side),
            "tempo_bonus": float(tempo_bonus),
            "sims_last": int(sims_last),
        }
        if eval_samples:
            try:
                meta["eval_std"] = float(np.std(np.asarray(eval_samples, dtype=np.float32)))
            except Exception:
                meta["eval_std"] = 0.0
        if move_count > 0:
            unique_ratio = len(seen_hashes) / float(move_count + 1)
            meta["unique_ratio"] = float(unique_ratio)
        else:
            meta["unique_ratio"] = 1.0
        if adjudicated:
            meta["adjudicated"] = 1
            meta["adjudicated_balance"] = adjudicated_balance
        if value_shift != 0.0:
            meta["value_shift"] = float(value_shift)

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

    def clear_buffer(self) -> None:
        with self.buffer_lock:
            self._buf = ccore.ReplayBuffer(
                int(C.REPLAY.BUFFER_CAPACITY),
                int(self._total_planes),
                BOARD_SIZE,
                BOARD_SIZE,
            )
        with contextlib.suppress(Exception):
            self._enc_cache.clear()

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
        meta = meta or {}
        value_shift = float(meta.get("value_shift", 0.0))
        term_reason = str(meta.get("termination", "natural"))
        self._process_result(
            examples,
            result,
            bool(first_to_move_is_white),
            value_shift,
            termination=term_reason,
        )
        stats["games"] += 1
        stats["moves"] += move_count
        if bool(first_to_move_is_white):
            stats["starts_white"] += 1
        else:
            stats["starts_black"] += 1
        stats["unique_positions_total"] += int(meta.get("unique_positions", 0))
        stats["unique_ratio_total"] += float(meta.get("unique_ratio", 0.0))
        stats["halfmove_sum"] += float(meta.get("halfmove_avg", 0.0))
        stats["halfmove_max_total"] = max(
            stats["halfmove_max_total"], int(meta.get("halfmove_max", 0))
        )
        if int(meta.get("threefold", 0)):
            stats["threefold_games"] += 1
        term_reason = str(meta.get("termination", "natural"))
        if term_reason == "resign":
            stats["term_resign"] += 1
        elif term_reason == "exhausted":
            stats["term_exhausted"] += 1
        elif term_reason == "threefold":
            stats["term_threefold"] += 1
            stats["threefold_draws"] += 1
        elif term_reason == "fifty_move":
            stats["term_fifty"] += 1
        elif term_reason == "adjudicated":
            stats["term_adjudicated"] += 1
        else:
            stats["term_natural"] += 1
        if int(meta.get("adjudicated", 0)):
            stats["adjudicated_games"] += 1
            balance = float(meta.get("adjudicated_balance", 0.0))
            stats["adjudicated_balance_total"] += balance
            stats["adjudicated_balance_abs_total"] += abs(balance)
            if result == ccore.WHITE_WIN:
                stats["adjudicated_white"] += 1
            elif result == ccore.BLACK_WIN:
                stats["adjudicated_black"] += 1
            else:
                stats["adjudicated_draw"] += 1
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
        if int(meta.get("curriculum_seed", 0)):
            stats["curriculum_games"] += 1
            if result == ccore.WHITE_WIN:
                stats["curriculum_white_wins"] += 1
            elif result == ccore.BLACK_WIN:
                stats["curriculum_black_wins"] += 1
            else:
                stats["curriculum_draws"] += 1
        terminal_eval_meta = float(meta.get("terminal_eval", 0.0))
        stats["terminal_eval_sum"] += terminal_eval_meta
        stats["terminal_eval_abs_sum"] += abs(terminal_eval_meta)
        stats["tempo_bonus_sum"] += float(meta.get("tempo_bonus", 0.0))
        stats["sims_last_sum"] += int(meta.get("sims_last", 0))
        if int(meta.get("value_trend_len", 0)) >= self.adjudicate_value_persist:
            stats["value_trend_hits"] += 1
        if result in {ccore.WHITE_WIN, ccore.BLACK_WIN}:
            losses = stats.get("_loss_eval_samples")
            if isinstance(losses, list):
                losses.append(float(np.clip(terminal_eval_meta, -1.0, 1.0)))
                max_keep = 1024
                if len(losses) > max_keep:
                    del losses[: len(losses) - max_keep]

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
            "threefold_draws": 0,
            "unique_positions_total": 0,
            "unique_ratio_total": 0.0,
            "halfmove_sum": 0.0,
            "halfmove_max_total": 0,
            "threefold_games": 0,
            "term_natural": 0,
            "term_resign": 0,
            "term_exhausted": 0,
            "term_threefold": 0,
            "term_fifty": 0,
            "term_adjudicated": 0,
            "adjudicated_games": 0,
            "adjudicated_white": 0,
            "adjudicated_black": 0,
            "adjudicated_draw": 0,
            "adjudicated_balance_total": 0.0,
            "adjudicated_balance_abs_total": 0.0,
            "curriculum_games": 0,
            "curriculum_white_wins": 0,
            "curriculum_black_wins": 0,
            "curriculum_draws": 0,
            "terminal_eval_sum": 0.0,
            "terminal_eval_abs_sum": 0.0,
            "value_trend_hits": 0,
            "tempo_bonus_sum": 0.0,
            "sims_last_sum": 0,
            "_loss_eval_samples": [],
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
        stats["adjudicate_margin"] = float(self.adjudicate_margin) if self.adjudicate_enabled else 0.0
        stats["adjudicate_min_plies"] = int(self.adjudicate_min_plies if self.adjudicate_enabled else 0)
        games_total = max(1, stats["games"])
        stats["unique_positions_avg"] = stats["unique_positions_total"] / games_total
        stats["unique_ratio_avg"] = stats["unique_ratio_total"] / games_total
        stats["halfmove_avg"] = stats["halfmove_sum"] / games_total
        stats["halfmove_max"] = int(stats["halfmove_max_total"])
        stats["threefold_pct"] = 100.0 * stats["threefold_games"] / games_total
        curr_games = max(0, int(stats.get("curriculum_games", 0)))
        if curr_games > 0:
            cw = int(stats.get("curriculum_white_wins", 0))
            cb = int(stats.get("curriculum_black_wins", 0))
            cd = int(stats.get("curriculum_draws", 0))
            stats["curriculum_win_pct"] = 100.0 * (cw + cb) / curr_games
            stats["curriculum_draw_pct"] = 100.0 * cd / curr_games
        else:
            stats["curriculum_win_pct"] = 0.0
            stats["curriculum_draw_pct"] = 0.0
        stats["terminal_eval_mean"] = stats["terminal_eval_sum"] / games_total
        stats["terminal_eval_abs_mean"] = stats["terminal_eval_abs_sum"] / games_total
        stats["tempo_bonus_avg"] = stats["tempo_bonus_sum"] / games_total
        stats["sims_avg"] = stats["sims_last_sum"] / games_total
        stats["value_trend_hit_pct"] = 100.0 * stats["value_trend_hits"] / games_total
        losses_list = stats.get("_loss_eval_samples")
        if losses_list:
            arr = np.asarray(losses_list, dtype=np.float32)
            stats["loss_eval_mean"] = float(arr.mean())
            stats["loss_eval_p05"] = float(np.percentile(arr, 5))
            stats["loss_eval_p25"] = float(np.percentile(arr, 25))
        else:
            stats["loss_eval_mean"] = 0.0
            stats["loss_eval_p05"] = 0.0
            stats["loss_eval_p25"] = 0.0
        for tmp_key in (
            "unique_positions_total",
            "unique_ratio_total",
            "halfmove_sum",
            "halfmove_max_total",
        ):
            stats.pop(tmp_key, None)
        return stats
