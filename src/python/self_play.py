"""AlphaZero-style self-play loop feeding the replay buffer.

The workflow is:

1. Spawn a small pool of workers, each playing independent games with the
   current neural network supplied through :class:`BatchedEvaluator`.
2. For every move, run Monte-Carlo Tree Search using batched neural network
   calls, record visit-count targets, and store the encoded position.
3. Once the game ends (naturally, by resignation, adjudication, or exhaustion)
   push the collected examples into the replay buffer with the final result.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import chesscore as ccore
import config as C
import encoder
import numpy as np
from encoder import POLICY_SIZE, encode_move_index
from replay_buffer import ReplayBuffer
from utils import flip_fen_perspective, sanitize_fen

__all__ = ["SelfPlayEngine"]


DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_MATERIAL_WEIGHTS = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)


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
        games = max(1, self.games)
        moves = max(1, self.moves)
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
            "avg_length": self.moves / games,
            "visit_per_move": self.visit_total / moves,
        }


def _load_opening_book(path_spec: object) -> list[tuple[str, float]]:
    """Load a JSON opening book and return [(fen, weight), ...]."""
    if path_spec is None:
        return []
    if isinstance(path_spec, (str, Path)):
        candidate = Path(path_spec).expanduser()
    else:
        candidate = Path(str(path_spec))
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[2] / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Opening book file not found: {candidate}")
    with candidate.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        entries = payload.get("entries", [])
    else:
        entries = payload
    book: list[tuple[str, float]] = []
    for item in entries:
        if isinstance(item, dict):
            fen = item.get("fen")
            weight = float(item.get("weight", 1.0))
        elif isinstance(item, Iterable):
            sequence = list(item)
            if not sequence:
                continue
            fen = sequence[0]
            weight = float(sequence[1] if len(sequence) > 1 else 1.0)
        else:
            continue
        if not fen:
            continue
        book.append((sanitize_fen(str(fen)), max(0.0, float(weight))))
    return book


class SelfPlayEngine:
    """Generates self-play games and manages the shared replay buffer."""

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

        # Resignation parameters.
        self.resign_consecutive = int(max(1, C.RESIGN.consecutive_required))
        self.resign_enabled = bool(C.RESIGN.enabled)
        self.resign_threshold = float(C.RESIGN.value_threshold)
        self.resign_min_plies = int(max(0, C.RESIGN.min_plies))
        self.resign_playthrough_fraction = float(np.clip(C.RESIGN.playthrough_fraction, 0.0, 1.0))

        # Adjudication parameters.
        self.adjudication_enabled = bool(C.SELFPLAY.adjudication_enabled)
        self.adjudication_min_plies = int(max(0, C.SELFPLAY.adjudication_min_plies))
        self.adjudication_value_margin = float(max(0.0, C.SELFPLAY.adjudication_value_margin))
        self.adjudication_persist = int(max(1, C.SELFPLAY.adjudication_persist_plies))
        self.adjudication_material_margin = float(max(0.0, getattr(C.SELFPLAY, "adjudication_material_margin", 0.0)))

        # Optional curriculum/opening book.
        self._curriculum_prob = float(np.clip(C.SELFPLAY.curriculum.sample_probability, 0.0, 1.0))
        self._curriculum_fens: tuple[str, ...] = tuple(sanitize_fen(str(fen)) for fen in C.SELFPLAY.curriculum.fens)

        book_entries: list[tuple[str, float]] = []
        if C.SELFPLAY.opening_book_path:
            try:
                book_entries = _load_opening_book(C.SELFPLAY.opening_book_path)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.log.warning("Opening book load failed: %s", exc)
                book_entries = []
        weights = np.array([w for _, w in book_entries], dtype=np.float64)
        self._opening_book: list[tuple[str, float]] = []
        self._opening_cumulative: np.ndarray | None = None
        if weights.size and float(weights.sum()) > 0.0:
            self._opening_book = book_entries
            self._opening_cumulative = np.cumsum(weights / weights.sum())

        self._rng = np.random.default_rng()
        self._eval_batch_cap = int(max(1, C.EVAL.batch_size_max))
        self.opening_random_moves = int(max(0, C.SELFPLAY.opening_random_moves))

    # ------------------------------------------------------------------ public API
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

    def update_adjudication(self, iteration: int) -> None:  # noqa: ARG002
        """Hook retained for interface compatibility (no dynamic schedule)."""

    def play_games(self, num_games: int) -> dict[str, float | int]:
        if num_games <= 0:
            return _SelfPlayStats().to_dict()

        stats = _SelfPlayStats()
        seeds = self._rng.integers(0, np.iinfo(np.int64).max, size=num_games)
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [pool.submit(self._play_single_game, int(seed)) for seed in seeds]
            for future in as_completed(futures):
                result, moves, termination, visits = future.result()
                stats.add(result=result, moves=moves, termination=termination, visits=visits)
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

    # ------------------------------------------------------------------ internals
    def _play_single_game(self, seed: int) -> tuple[ccore.Result, int, str, float]:
        rng = np.random.default_rng(seed)
        start_fen = self.sample_start_fen(rng)
        position = ccore.Position()
        try:
            position.from_fen(start_fen)
        except Exception:
            position = ccore.Position()
            position.from_fen(DEFAULT_START_FEN)

        mcts = self._build_mcts(rng)
        history: deque[ccore.Position] = deque(maxlen=max(0, encoder.HISTORY_LENGTH - 1))
        examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []

        move_count = 0
        visit_total = 0.0
        termination = "natural"
        result: ccore.Result | None = None

        resign_streak = 0
        advantage_sign = 0
        advantage_count = 0

        while move_count < self.game_max_plies:
            game_result = position.result()
            if game_result != ccore.ONGOING:
                result = game_result
                break

            legal_moves = position.legal_moves()
            if not legal_moves:
                result = position.result()
                break

            sims = self._simulations_for(move_count)
            mcts.set_simulations(sims)
            counts = mcts.search_batched_legal(position, self.evaluator.infer_positions_legal, self._eval_batch_cap)
            visit_counts = np.asarray(counts, dtype=np.float64)
            if visit_counts.shape[0] != len(legal_moves):
                termination = "exhausted"
                result = ccore.DRAW
                break

            visit_total += float(visit_counts.sum())

            indices, counts_u16 = self._policy_targets(legal_moves, visit_counts)
            encoded = encoder.encode_position(position, history)
            state_u8 = self._encode_state(encoded)
            stm_is_white = bool(getattr(position, "turn", ccore.WHITE) == ccore.WHITE)
            examples.append((state_u8, indices, counts_u16, stm_is_white))

            temperature = (
                C.SELFPLAY.temperature_high if move_count < C.SELFPLAY.temperature_moves else C.SELFPLAY.temperature_low
            )
            temperature = max(float(temperature), float(C.SELFPLAY.deterministic_temp_eps))
            if move_count < self.opening_random_moves:
                move = legal_moves[int(rng.integers(0, len(legal_moves)))]
            else:
                move_index = self._select_move(visit_counts, temperature, rng)
                move = legal_moves[move_index]

            value_raw = float(self.evaluator.infer_values([position])[0])
            player_view = value_raw if stm_is_white else -value_raw

            if self.resign_enabled and move_count >= self.resign_min_plies:
                if player_view <= self.resign_threshold:
                    resign_streak += 1
                    if resign_streak >= self.resign_consecutive:
                        if rng.random() > self.resign_playthrough_fraction:
                            termination = "resign"
                            result = ccore.BLACK_WIN if stm_is_white else ccore.WHITE_WIN
                            break
                else:
                    resign_streak = 0

            if self.adjudication_enabled and move_count >= self.adjudication_min_plies:
                margin = self.adjudication_value_margin
                sign = 0
                if abs(player_view) >= margin:
                    if player_view > 0:
                        sign = 1
                    elif player_view < 0:
                        sign = -1
                    if sign != 0 and sign == advantage_sign:
                        advantage_count += 1
                    elif sign != 0:
                        advantage_sign = sign
                        advantage_count = 1
                    else:
                        advantage_sign = 0
                        advantage_count = 0
                else:
                    advantage_sign = 0
                    advantage_count = 0
                if advantage_sign != 0 and advantage_count >= self.adjudication_persist:
                    termination = "adjudicated"
                    result = ccore.WHITE_WIN if advantage_sign > 0 else ccore.BLACK_WIN
                    break
                if self.adjudication_material_margin > 0.0:
                    balance = self._material_balance(position)
                    if abs(balance) >= self.adjudication_material_margin:
                        termination = "adjudicated"
                        result = ccore.WHITE_WIN if balance > 0 else ccore.BLACK_WIN
                        break

            history.append(ccore.Position(position))
            position.make_move(move)
            try:
                mcts.advance_root(position, move)
            except Exception:
                mcts = self._build_mcts(rng)
            move_count += 1

        if result is None:
            result = position.result()
        if result == ccore.ONGOING:
            termination = "exhausted"
            result = ccore.DRAW

        self._store_examples(examples, result)
        return result, move_count, termination, visit_total

    def sample_start_fen(self, rng: np.random.Generator) -> str:
        fen: str | None = None
        if self._opening_book and self._opening_cumulative is not None:
            draw = float(rng.random())
            idx = int(np.searchsorted(self._opening_cumulative, draw, side="right"))
            idx = min(idx, len(self._opening_book) - 1)
            fen = self._opening_book[idx][0]
        elif self._curriculum_fens and rng.random() < self._curriculum_prob:
            fen = str(rng.choice(self._curriculum_fens))
        if fen is None:
            fen = DEFAULT_START_FEN
        if rng.random() < 0.5:
            fen = flip_fen_perspective(fen)
        return fen

    def _build_mcts(self, rng: np.random.Generator) -> ccore.MCTS:
        mcts = ccore.MCTS(
            int(C.MCTS.train_simulations),
            float(C.MCTS.c_puct),
            float(C.MCTS.dirichlet_alpha),
            float(C.MCTS.dirichlet_weight),
        )
        mcts.set_c_puct_params(float(C.MCTS.c_puct_base), float(C.MCTS.c_puct_init))
        mcts.set_fpu_reduction(float(C.MCTS.fpu_reduction))
        mcts.seed(int(rng.integers(1, np.iinfo(np.int64).max)))
        return mcts

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
        choice = rng.choice(len(probs), p=probs)
        return int(choice)

    @staticmethod
    def _encode_state(encoded: np.ndarray) -> np.ndarray:
        arr = np.clip(
            np.rint(encoded * float(C.DATA.u8_scale)),
            0,
            255,
        ).astype(np.uint8, copy=False)
        return arr

    @staticmethod
    def _encode_value(value: float) -> np.int8:
        scale = float(C.DATA.value_i8_scale)
        clipped = np.clip(value * scale, -128, 127)
        return np.int8(int(round(float(clipped))))

    @staticmethod
    def _material_balance(position: ccore.Position) -> float:
        pieces = getattr(position, "pieces", None)
        if pieces is None:
            return 0.0
        try:
            seq = list(pieces)  # type: ignore[arg-type]
        except Exception:
            return 0.0
        balance = 0.0
        for idx, weight in enumerate(_MATERIAL_WEIGHTS):
            if idx >= len(seq):
                break
            try:
                white_bb, black_bb = seq[idx]
            except Exception:
                continue
            try:
                w_count = int(white_bb).bit_count()
                b_count = int(black_bb).bit_count()
            except Exception:
                try:
                    w_count = bin(int(white_bb)).count("1")
                    b_count = bin(int(black_bb)).count("1")
                except Exception:
                    continue
            balance += weight * (w_count - b_count)
        return float(balance)

    def _policy_targets(self, moves: Iterable[Any], visit_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx_list: list[int] = []
        cnt_list: list[int] = []
        for move, count in zip(moves, visit_counts, strict=False):
            if count <= 0:
                continue
            move_idx = encode_move_index(move)
            if move_idx is None:
                continue
            if not (0 <= int(move_idx) < POLICY_SIZE):
                continue
            clamped = int(min(count, C.MCTS.visit_count_clamp))
            if clamped <= 0:
                continue
            idx_list.append(int(move_idx))
            cnt_list.append(clamped)
        idx_arr = np.asarray(idx_list, dtype=np.int32)
        cnt_arr = np.asarray(cnt_list, dtype=np.uint16)
        return idx_arr, cnt_arr

    def _store_examples(
        self,
        examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]],
        result: ccore.Result,
    ) -> None:
        outcome = self._normalize_result(result)
        if outcome is None:
            self.log.warning("Discarding game with unknown result %r", result)
            return
        if outcome == ccore.WHITE_WIN:
            base = 1.0
        elif outcome == ccore.BLACK_WIN:
            base = -1.0
        else:
            base = 0.0
        with self._buffer_lock:
            for ply_index, (state_u8, idx_arr, cnt_arr, stm_is_white) in enumerate(examples):
                if idx_arr.size == 0 or cnt_arr.size == 0:
                    continue
                target = base if stm_is_white else -base
                value_i8 = self._encode_value(target)
                self._buffer.push(state_u8, idx_arr, cnt_arr, value_i8)

    @staticmethod
    def _normalize_result(result: ccore.Result | int | float | None) -> ccore.Result | None:
        if isinstance(result, ccore.Result):
            return result
        try:
            return ccore.Result(int(result))  # type: ignore[arg-type]
        except Exception:
            return None
