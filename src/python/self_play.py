from __future__ import annotations

import json
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import config as C
import encoder
import numpy as np
import chesscore as ccore
from encoder import POLICY_SIZE, encode_move_index
from replay_buffer import ReplayBuffer
from utils import flip_fen_perspective, sanitize_fen

DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_MATERIAL_WEIGHTS = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)  # P,N,B,R,Q,K


# ---------- stats

@dataclass(slots=True)
class _SelfPlayStats:
    games: int = 0
    moves: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    term_natural: int = 0
    term_resign: int = 0
    term_adjudicated: int = 0
    term_exhausted: int = 0
    visit_total: float = 0.0

    def add(self, *, result: ccore.Result, moves: int, termination: str, visits: float) -> None:
        self.games += 1
        self.moves += int(moves)
        self.visit_total += float(visits)
        if result == ccore.WHITE_WIN:
            self.white_wins += 1
        elif result == ccore.BLACK_WIN:
            self.black_wins += 1
        else:
            self.draws += 1
        term = termination.lower()
        if term == "resign":
            self.term_resign += 1
        elif term == "adjudicated":
            self.term_adjudicated += 1
        elif term == "exhausted":
            self.term_exhausted += 1
        else:
            self.term_natural += 1

    def to_dict(self) -> dict[str, float | int]:
        g = max(1, self.games)
        m = max(1, self.moves)
        return {
            "games": self.games,
            "moves": self.moves,
            "white_wins": self.white_wins,
            "black_wins": self.black_wins,
            "draws": self.draws,
            "term_natural": self.term_natural,
            "term_resign": self.term_resign,
            "term_adjudicated": self.term_adjudicated,
            "term_exhausted": self.term_exhausted,
            "avg_length": self.moves / g,
            "visit_per_move": self.visit_total / m,
        }


# ---------- helpers

def _load_opening_book(path_spec: object) -> list[tuple[str, float]]:
    """Read JSON book -> [(sanitized_fen, weight)]. Accepts file with {'entries': ...} or a list."""
    if path_spec is None:
        return []
    p = Path(path_spec).expanduser() if isinstance(path_spec, (str, Path)) else Path(str(path_spec))
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Opening book file not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    out: list[tuple[str, float]] = []
    for item in entries:
        if isinstance(item, dict):
            fen = item.get("fen")
            w = float(item.get("weight", 1.0))
        elif isinstance(item, Iterable):
            seq = list(item)
            fen = seq[0] if seq else None
            w = float(seq[1] if len(seq) > 1 else 1.0)
        else:
            fen, w = None, 1.0
        if fen:
            out.append((sanitize_fen(str(fen)), max(0.0, w)))
    return out


def _encode_state_u8(encoded: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(encoded * float(C.DATA.u8_scale)), 0, 255).astype(np.uint8, copy=False)


def _encode_value_i8(value: float) -> np.int8:
    scale = float(C.DATA.value_i8_scale)
    return np.int8(int(round(float(np.clip(value * scale, -128, 127)))))


def _material_balance(position: ccore.Position) -> float:
    pieces = getattr(position, "pieces", None)
    if pieces is None:
        return 0.0
    try:
        seq = list(pieces)
    except Exception:
        return 0.0
    bal = 0.0
    for idx, w in enumerate(_MATERIAL_WEIGHTS):
        if idx >= len(seq):
            break
        try:
            w_bb, b_bb = seq[idx]
            w_cnt = int(w_bb).bit_count()
            b_cnt = int(b_bb).bit_count()
        except Exception:
            try:
                w_cnt = bin(int(w_bb)).count("1")
                b_cnt = bin(int(b_bb)).count("1")
            except Exception:
                continue
        bal += w * (w_cnt - b_cnt)
    return float(bal)


def _policy_targets(moves: Iterable[Any], visit_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx_list: list[int] = []
    cnt_list: list[int] = []
    for mv, cnt in zip(moves, visit_counts, strict=False):
        if cnt <= 0:
            continue
        mv_idx = encode_move_index(mv)
        if mv_idx is None or not (0 <= int(mv_idx) < POLICY_SIZE):
            continue
        clamped = int(min(cnt, C.MCTS.visit_count_clamp))
        if clamped <= 0:
            continue
        idx_list.append(int(mv_idx))
        cnt_list.append(clamped)
    return np.asarray(idx_list, dtype=np.int32), np.asarray(cnt_list, dtype=np.uint16)


# ---------- engine

class SelfPlayEngine:
    """Generates self-play games and manages the replay buffer."""

    def __init__(self, evaluator: Any) -> None:
        self.log = logging.getLogger("hybridchess.selfplay")
        self.evaluator = evaluator

        self.num_workers = max(1, int(C.SELFPLAY.num_workers))
        self.game_max_plies = int(max(1, C.SELFPLAY.game_max_plies))

        planes = encoder.INPUT_PLANES
        self._buffer = ReplayBuffer(
            capacity=int(C.REPLAY.capacity),
            planes=int(planes),
            height=encoder.BOARD_SIZE,
            width=encoder.BOARD_SIZE,
        )
        self._buffer_lock = threading.Lock()

        # Resignation
        self.resign_consecutive = int(max(1, C.RESIGN.consecutive_required))
        self.resign_enabled = bool(C.RESIGN.enabled)
        self.resign_threshold = float(C.RESIGN.value_threshold)
        self.resign_min_plies = int(max(0, C.RESIGN.min_plies))
        self.resign_playthrough_fraction = float(np.clip(C.RESIGN.playthrough_fraction, 0.0, 1.0))

        # Adjudication
        self.adjudication_enabled = bool(C.SELFPLAY.adjudication_enabled)
        self.adjudication_min_plies = int(max(0, C.SELFPLAY.adjudication_min_plies))
        self.adjudication_value_margin = float(max(0.0, C.SELFPLAY.adjudication_value_margin))
        self.adjudication_persist = int(max(1, C.SELFPLAY.adjudication_persist_plies))
        self.adjudication_material_margin = float(max(0.0, getattr(C.SELFPLAY, "adjudication_material_margin", 0.0)))

        # Curriculum and opening book
        self._curriculum_prob = float(np.clip(C.SELFPLAY.curriculum.sample_probability, 0.0, 1.0))
        self._curriculum_fens: tuple[str, ...] = tuple(sanitize_fen(str(f)) for f in C.SELFPLAY.curriculum.fens)

        entries: list[tuple[str, float]] = []
        if C.SELFPLAY.opening_book_path:
            try:
                entries = _load_opening_book(C.SELFPLAY.opening_book_path)
            except Exception as exc:  # defensive
                self.log.warning("Opening book load failed: %s", exc)
        weights = np.array([w for _, w in entries], dtype=np.float64)
        self._opening_book: list[tuple[str, float]] = []
        self._opening_cumulative: np.ndarray | None = None
        if weights.size and float(weights.sum()) > 0.0:
            self._opening_book = entries
            self._opening_cumulative = np.cumsum(weights / weights.sum())

        self._rng = np.random.default_rng()
        self._eval_batch_cap = int(max(1, C.EVAL.batch_size_max))
        self.opening_random_moves = int(max(0, C.SELFPLAY.opening_random_moves))

    # ----- public API

    def enable_resign(self, enabled: bool) -> None:
        self.resign_enabled = bool(enabled)

    def set_num_workers(self, n: int) -> None:
        self.num_workers = max(1, int(n))

    def get_num_workers(self) -> int:
        return int(self.num_workers)

    def set_game_length(self, max_plies: int) -> None:
        self.game_max_plies = int(max(1, max_plies))

    def get_game_length(self) -> int:
        return int(self.game_max_plies)

    def set_resign_params(self, threshold: float, min_plies: int) -> None:
        self.resign_threshold = float(threshold)
        self.resign_min_plies = int(max(0, min_plies))

    def update_adjudication(self, iteration: int) -> None:
        """Hook retained for compatibility; schedule is static."""
        return

    def play_games(self, num_games: int) -> dict[str, float | int]:
        if num_games <= 0:
            return _SelfPlayStats().to_dict()
        stats = _SelfPlayStats()
        seeds = self._rng.integers(0, np.iinfo(np.int64).max, size=num_games)
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [pool.submit(self._play_single_game, int(s)) for s in seeds]
            for f in as_completed(futures):
                res, moves, term, visits = f.result()
                stats.add(result=res, moves=moves, termination=term, visits=visits)
        return stats.to_dict()

    def sample_batch(
        self,
        batch_size: int,
        recent_ratio: float,
        recent_window_frac: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.int8]]:
        return self._buffer.sample(
            batch_size=int(batch_size),
            recent_ratio=float(np.clip(recent_ratio, 0.0, 1.0)),
            recent_window_frac=float(np.clip(recent_window_frac, 0.0, 1.0)),
        )

    def get_capacity(self) -> int:
        return int(self._buffer.capacity)

    def set_capacity(self, capacity: int) -> None:
        with self._buffer_lock:
            self._buffer.set_capacity(int(max(1, capacity)))

    def size(self) -> int:
        return int(self._buffer.size)

    def clear_buffer(self) -> None:
        with self._buffer_lock:
            self._buffer.clear()

    # ----- internals

    def _play_single_game(self, seed: int) -> tuple[ccore.Result, int, str, float]:
        rng = np.random.default_rng(seed)
        start_fen = self.sample_start_fen(rng)
        position = ccore.Position()
        try:
            position.from_fen(start_fen)
        except Exception:
            position.from_fen(DEFAULT_START_FEN)

        mcts = self._build_mcts(rng)
        history: deque[ccore.Position] = deque(maxlen=max(0, encoder.HISTORY_LENGTH - 1))
        examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []

        moves = 0
        visit_total = 0.0
        termination = "natural"
        result: ccore.Result | None = None

        resign_streak = 0
        advantage_sign = 0
        advantage_count = 0

        while moves < self.game_max_plies:
            game_result = position.result()
            if game_result != ccore.ONGOING:
                result = game_result
                break

            legal = position.legal_moves()
            if not legal:
                result = position.result()
                break

            sims = self._simulations_for(moves)
            mcts.set_simulations(sims)
            counts = mcts.search_batched_legal(position, self.evaluator.infer_positions_legal, self._eval_batch_cap)
            vc = np.asarray(counts, dtype=np.float64)
            if vc.shape[0] != len(legal):
                termination = "exhausted"
                result = ccore.DRAW
                break
            visit_total += float(vc.sum())

            indices, counts_u16 = _policy_targets(legal, vc)
            encoded = encoder.encode_position(position, history)
            state_u8 = _encode_state_u8(encoded)
            stm_white = bool(getattr(position, "turn", ccore.WHITE) == ccore.WHITE)
            examples.append((state_u8, indices, counts_u16, stm_white))

            # Move selection
            temp = C.SELFPLAY.temperature_high if moves < C.SELFPLAY.temperature_moves else C.SELFPLAY.temperature_low
            temp = max(float(temp), float(C.SELFPLAY.deterministic_temp_eps))
            if moves < self.opening_random_moves:
                mv = legal[int(rng.integers(0, len(legal)))]
            else:
                mv = legal[self._select_move(vc, temp, rng)]

            # Value and resignation/adjudication checks
            v_raw = float(self.evaluator.infer_values([position])[0])
            player_view = v_raw if stm_white else -v_raw

            if self.resign_enabled and moves >= self.resign_min_plies:
                if player_view <= self.resign_threshold:
                    resign_streak += 1
                    if resign_streak >= self.resign_consecutive and rng.random() > self.resign_playthrough_fraction:
                        termination = "resign"
                        result = ccore.BLACK_WIN if stm_white else ccore.WHITE_WIN
                        break
                else:
                    resign_streak = 0

            if self.adjudication_enabled and moves >= self.adjudication_min_plies:
                margin = self.adjudication_value_margin
                sign = 1 if player_view > margin else (-1 if player_view < -margin else 0)
                if sign == advantage_sign and sign != 0:
                    advantage_count += 1
                elif sign != 0:
                    advantage_sign, advantage_count = sign, 1
                else:
                    advantage_sign, advantage_count = 0, 0
                if advantage_sign != 0 and advantage_count >= self.adjudication_persist:
                    termination = "adjudicated"
                    result = ccore.WHITE_WIN if advantage_sign > 0 else ccore.BLACK_WIN
                    break
                if self.adjudication_material_margin > 0.0:
                    bal = _material_balance(position)
                    if abs(bal) >= self.adjudication_material_margin:
                        termination = "adjudicated"
                        result = ccore.WHITE_WIN if bal > 0 else ccore.BLACK_WIN
                        break

            history.append(ccore.Position(position))
            position.make_move(mv)
            try:
                mcts.advance_root(position, mv)
            except Exception:
                mcts = self._build_mcts(rng)
            moves += 1

        if result is None:
            result = position.result()
        if result == ccore.ONGOING:
            termination = "exhausted"
            result = ccore.DRAW

        self._store_examples(examples, result)
        return result, moves, termination, visit_total

    def sample_start_fen(self, rng: np.random.Generator) -> str:
        fen: str | None = None
        if self._opening_book and self._opening_cumulative is not None:
            draw = float(rng.random())
            idx = int(np.searchsorted(self._opening_cumulative, draw, side="right"))
            fen = self._opening_book[min(idx, len(self._opening_book) - 1)][0]
        elif self._curriculum_fens and rng.random() < self._curriculum_prob:
            fen = str(rng.choice(self._curriculum_fens))
        if fen is None:
            fen = DEFAULT_START_FEN
        if rng.random() < 0.5:
            fen = flip_fen_perspective(fen)
        return fen

    def _build_mcts(self, rng: np.random.Generator) -> ccore.MCTS:
        m = ccore.MCTS(
            int(C.MCTS.train_simulations),
            float(C.MCTS.c_puct),
            float(C.MCTS.dirichlet_alpha),
            float(C.MCTS.dirichlet_weight),
        )
        m.set_c_puct_params(float(C.MCTS.c_puct_base), float(C.MCTS.c_puct_init))
        m.set_fpu_reduction(float(C.MCTS.fpu_reduction))
        m.seed(int(rng.integers(1, np.iinfo(np.int64).max)))
        return m

    @staticmethod
    def _simulations_for(move_count: int) -> int:
        base = int(max(1, C.MCTS.train_simulations))
        decay = int(max(1, C.MCTS.train_sim_decay_move_interval))
        sims = base // (1 + max(0, move_count) // decay)
        return max(int(C.MCTS.train_simulations_min), sims)

    @staticmethod
    def _select_move(visit_counts: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
        if visit_counts.ndim != 1 or visit_counts.size == 0:
            return 0
        if temperature <= C.SELFPLAY.deterministic_temp_eps:
            return int(np.argmax(visit_counts))
        scaled = np.maximum(visit_counts, 0.0) ** (1.0 / temperature)
        s = float(scaled.sum())
        if not np.isfinite(s) or s <= 0.0:
            return int(np.argmax(visit_counts))
        probs = np.asarray(scaled / s, dtype=np.float64)
        return int(rng.choice(len(probs), p=probs))

    def _store_examples(
        self,
        examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]],
        result: ccore.Result,
    ) -> None:
        outcome = self._normalize_result(result)
        if outcome is None:
            self.log.warning("Discarding game with unknown result %r", result)
            return
        base = 1.0 if outcome == ccore.WHITE_WIN else (-1.0 if outcome == ccore.BLACK_WIN else 0.0)
        with self._buffer_lock:
            for state_u8, idx_arr, cnt_arr, stm_white in examples:
                if idx_arr.size == 0 or cnt_arr.size == 0:
                    continue
                target = base if stm_white else -base
                self._buffer.push(state_u8, idx_arr, cnt_arr, _encode_value_i8(target))

    @staticmethod
    def _normalize_result(result: ccore.Result | int | float | None) -> ccore.Result | None:
        if isinstance(result, ccore.Result):
            return result
        try:
            return ccore.Result(int(result))
        except Exception:
            return None