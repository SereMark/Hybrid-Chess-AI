from __future__ import annotations

import contextlib
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chesscore as ccore
import numpy as np

import config as C

BOARD_SIZE = 8
NSQUARES = 64
PLANES_PER_POSITION = int(getattr(ccore, "PLANES_PER_POSITION", 14))
HISTORY_LENGTH = int(getattr(ccore, "HISTORY_LENGTH", 8))
POLICY_OUTPUT = int(getattr(ccore, "POLICY_SIZE", 73 * NSQUARES))

SELFPLAY_TEMP_MOVES = int(C.SELFPLAY.TEMP_MOVES)
SELFPLAY_TEMP_HIGH = float(C.SELFPLAY.TEMP_HIGH)
SELFPLAY_TEMP_LOW = float(C.SELFPLAY.TEMP_LOW)
SELFPLAY_DETERMINISTIC_TEMP_EPS = float(C.SELFPLAY.DETERMINISTIC_TEMP_EPS)


class SelfPlayEngine:
    def __init__(self, evaluator: Any) -> None:
        self.resign_consecutive = C.RESIGN.CONSECUTIVE_PLIES
        self.evaluator = evaluator
        self.buffer: deque[tuple[np.ndarray, np.ndarray, np.int8]] = deque(maxlen=C.REPLAY.BUFFER_CAPACITY)
        self.buffer_lock = threading.Lock()
        self.num_workers: int = int(C.SELFPLAY.NUM_WORKERS)

    def set_num_workers(self, n: int) -> None:
        self.num_workers = int(max(1, n))

    def get_num_workers(self) -> int:
        return int(self.num_workers)

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
        examples: list[tuple[np.ndarray, np.ndarray]],
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
            for ply_index, (position_u8, counts_u16) in enumerate(examples):
                stm_is_white = ((ply_index % 2) == 0) == bool(first_to_move_is_white)
                target_value = base if stm_is_white else -base
                self.buffer.append(
                    (
                        position_u8,
                        counts_u16,
                        SelfPlayEngine.encode_value_i8(target_value),
                    )
                )

    def play_single_game(self, seed: int | None = None) -> tuple[int, int, list[tuple[np.ndarray, np.ndarray]], bool]:
        position = ccore.Position()
        resign_count = 0
        forced_result: int | None = None
        mcts = ccore.MCTS(
            C.MCTS.TRAIN_SIMULATIONS_BASE,
            C.MCTS.C_PUCT,
            C.MCTS.DIRICHLET_ALPHA,
            C.MCTS.DIRICHLET_WEIGHT,
        )
        mcts.set_c_puct_params(C.MCTS.C_PUCT_BASE, C.MCTS.C_PUCT_INIT)
        mcts.set_fpu_reduction(C.MCTS.FPU_REDUCTION)
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        mcts.seed(int(rng.integers(2**63 - 1)))

        examples: list[tuple[np.ndarray, np.ndarray]] = []
        position_history: list[Any] = []
        move_count = 0

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

            moves = position.legal_moves()
            if not moves:
                break
            visit_counts = mcts.search_batched(position, self.evaluator.infer_positions, C.EVAL.BATCH_SIZE_MAX)
            if not visit_counts or len(visit_counts) != len(moves):
                break

            policy_counts = np.zeros(POLICY_OUTPUT, dtype=np.uint16)
            for mv, vc in zip(moves, visit_counts, strict=False):
                move_index = ccore.encode_move_index(mv)
                if (move_index is not None) and (0 <= int(move_index) < POLICY_OUTPUT):
                    c = min(int(vc), C.MCTS.VISIT_COUNT_CLAMP)
                    policy_counts[int(move_index)] = c
            enc = ccore.encode_position(pos_snapshot)
            examples.append((SelfPlayEngine.encode_u8(enc), policy_counts))
            if first_to_move_is_white is None:
                first_to_move_is_white = position.turn == ccore.WHITE

            if C.RESIGN.ENABLED:
                _, val_arr = self.evaluator.infer_positions([pos_snapshot])
                value_estimate = float(val_arr[0])
                if value_estimate <= C.RESIGN.VALUE_THRESHOLD:
                    resign_count += 1
                    if resign_count >= self.resign_consecutive:
                        if float(rng.random()) < C.RESIGN.PLAYTHROUGH_FRACTION:
                            resign_count = 0
                        else:
                            side_to_move_is_white = position.turn == ccore.WHITE
                            forced_result = ccore.BLACK_WIN if side_to_move_is_white else ccore.WHITE_WIN
                            break
                else:
                    resign_count = 0

            move = self._select_move_by_temperature(moves, visit_counts, move_count, rng=rng)
            position.make_move(move)

            with contextlib.suppress(Exception):
                mcts.advance_root(position, move)
            position_history.append(pos_snapshot)
            if len(position_history) > HISTORY_LENGTH:
                position_history.pop(0)
            move_count += 1

        final_result = forced_result if forced_result is not None else position.result()

        return (
            move_count,
            final_result,
            examples,
            True if first_to_move_is_white is None else bool(first_to_move_is_white),
        )

    def snapshot(self) -> list[tuple[np.ndarray, np.ndarray, np.int8]]:
        with self.buffer_lock:
            return list(self.buffer)

    def get_capacity(self) -> int:
        with self.buffer_lock:
            return int(self.buffer.maxlen or 0)

    def set_capacity(self, capacity: int) -> None:
        cap = int(max(1, capacity))
        with self.buffer_lock:
            items = list(self.buffer)
            if len(items) > cap:
                items = items[-cap:]
            self.buffer = deque(items, maxlen=cap)

    def sample_from_snapshot(
        self,
        snapshot: list[tuple[np.ndarray, np.ndarray, np.int8]],
        batch_size: int,
        recent_ratio: float = C.SAMPLING.REPLAY_SNAPSHOT_RECENT_RATIO_DEFAULT,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
        sample_count = len(snapshot)
        if batch_size > sample_count:
            return None
        recent_window_count = max(1, int(sample_count * C.SAMPLING.REPLAY_SNAPSHOT_RECENT_WINDOW_FRAC))
        n_recent = round(batch_size * recent_ratio)
        n_old = batch_size - n_recent
        recent_indices = np.random.randint(max(0, sample_count - recent_window_count), sample_count, size=n_recent)
        old_indices = np.random.randint(0, max(1, sample_count - recent_window_count), size=n_old)
        indices = np.concatenate([recent_indices, old_indices])
        states_u8_list, counts_u16_list, values_i8_list = zip(*[snapshot[int(i)] for i in indices], strict=False)
        states = list(states_u8_list)
        counts = list(counts_u16_list)
        values = list(values_i8_list)
        return states, counts, values

    def play_games(self, num_games: int) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "games": 0,
            "moves": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
        }
        seeds = [int(np.random.randint(0, 2**63 - 1)) for _ in range(num_games)]
        if int(C.SEED) != 0:
            results: dict[int, tuple[int, int, list[tuple[np.ndarray, np.ndarray]], bool]] = {}
            with ThreadPoolExecutor(max_workers=max(1, self.num_workers)) as ex:
                futures = {ex.submit(self.play_single_game, seeds[i]): i for i in range(num_games)}
                for fut in as_completed(futures):
                    i = futures[fut]
                    results[i] = fut.result()
            for i in range(num_games):
                moves, result, examples, first_to_move_is_white = results[i]
                self._process_result(examples, result, bool(first_to_move_is_white))
                stats["games"] += 1
                stats["moves"] += moves
                if result == ccore.WHITE_WIN:
                    stats["white_wins"] += 1
                elif result == ccore.BLACK_WIN:
                    stats["black_wins"] += 1
                else:
                    stats["draws"] += 1
        else:
            with ThreadPoolExecutor(max_workers=max(1, self.num_workers)) as ex:
                futures = {ex.submit(self.play_single_game, seeds[i]): i for i in range(num_games)}
                for fut in as_completed(futures):
                    moves, result, examples, first_to_move_is_white = fut.result()
                    self._process_result(examples, result, bool(first_to_move_is_white))
                    stats["games"] += 1
                    stats["moves"] += moves
                    if result == ccore.WHITE_WIN:
                        stats["white_wins"] += 1
                    elif result == ccore.BLACK_WIN:
                        stats["black_wins"] += 1
                    else:
                        stats["draws"] += 1
        return stats
