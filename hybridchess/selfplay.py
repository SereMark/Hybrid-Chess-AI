from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import chesscore as ccore
import numpy as np

from .config import (
    BOARD_SIZE,
    DIR_MAP_MIRROR,
    DIR_MAP_ROT180,
    DIR_MAP_VFLIP_CS,
    DIR_MAX_DIST,
    EVAL_BATCH_SIZE_MAX,
    GAME_MAX_PLIES,
    HISTORY_LENGTH,
    KMAP_MIRROR,
    KMAP_ROT180,
    KMAP_VFLIP_CS,
    KNIGHT_PLANES_BASE,
    MCTS_C_PUCT,
    MCTS_C_PUCT_BASE,
    MCTS_C_PUCT_INIT,
    MCTS_DIRICHLET_ALPHA,
    MCTS_DIRICHLET_WEIGHT,
    MCTS_FPU_REDUCTION,
    MCTS_TRAIN_SIM_DECAY_MOVE_INTERVAL,
    MCTS_TRAIN_SIMULATIONS_BASE,
    MCTS_TRAIN_SIMULATIONS_MIN,
    MCTS_VISIT_COUNT_CLAMP,
    NSQUARES,
    NUM_DIRECTIONS,
    NUM_KNIGHT_DIRS,
    PLANES_PER_POSITION,
    PMAP_PROMOS,
    POLICY_OUTPUT,
    PROMO_CHOICES,
    PROMO_STRIDE,
    REPLAY_BUFFER_CAPACITY,
    REPLAY_SNAPSHOT_RECENT_RATIO_DEFAULT,
    REPLAY_SNAPSHOT_RECENT_WINDOW_FRAC,
    RESIGN_CONSECUTIVE_PLIES,
    RESIGN_PLAYTHROUGH_FRACTION,
    RESIGN_VALUE_THRESHOLD,
    SELFPLAY_DETERMINISTIC_TEMP_EPS,
    SELFPLAY_NUM_WORKERS,
    SELFPLAY_OPENING_RANDOM_PLIES_MAX,
    SELFPLAY_TEMP_HIGH,
    SELFPLAY_TEMP_LOW,
    SELFPLAY_TEMP_MOVES,
    U8_SCALE,
    VALUE_I8_SCALE,
)

if TYPE_CHECKING:
    from .model import BatchedEvaluator


class Augment:
    """State/policy augmentations and their plane/index remappings."""

    _policy_map_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _policy_index_permutation(transform: str) -> np.ndarray:
        if transform in Augment._policy_map_cache:
            return Augment._policy_map_cache[transform]
        assert POLICY_OUTPUT % NSQUARES == 0, "POLICY_OUTPUT must be divisible by NSQUARES"
        planes = POLICY_OUTPUT // NSQUARES
        required_planes = max(
            KNIGHT_PLANES_BASE + NUM_KNIGHT_DIRS,
            NSQUARES + PROMO_STRIDE * PROMO_CHOICES,
        )
        base = np.arange(POLICY_OUTPUT, dtype=np.int32).reshape(planes, BOARD_SIZE, BOARD_SIZE)
        out = base
        if transform == "mirror":
            arr = base[:, :, ::-1]
            out = arr.copy()
            if planes >= required_planes:
                dir_map = DIR_MAP_MIRROR
                for d in range(NUM_DIRECTIONS):
                    for dist in range(DIR_MAX_DIST):
                        out[dir_map[d] * DIR_MAX_DIST + dist] = arr[d * DIR_MAX_DIST + dist]
                knight_map = KMAP_MIRROR
                for k in range(NUM_KNIGHT_DIRS):
                    out[KNIGHT_PLANES_BASE + knight_map[k]] = arr[KNIGHT_PLANES_BASE + k]
                promo_map = PMAP_PROMOS
                for promo in range(PROMO_CHOICES):
                    b = NSQUARES + promo * PROMO_STRIDE
                    out[b + promo_map[0]] = arr[b + 0]
                    out[b + promo_map[1]] = arr[b + 1]
                    out[b + promo_map[2]] = arr[b + 2]
            else:
                Augment._policy_map_cache[transform] = np.arange(POLICY_OUTPUT, dtype=np.int32)
                return Augment._policy_map_cache[transform]
        elif transform == "rot180":
            arr = base[:, ::-1, ::-1]
            out = arr.copy()
            if planes >= required_planes:
                dir_map = DIR_MAP_ROT180
                for d in range(NUM_DIRECTIONS):
                    for dist in range(DIR_MAX_DIST):
                        out[dir_map[d] * DIR_MAX_DIST + dist] = arr[d * DIR_MAX_DIST + dist]
                knight_map = KMAP_ROT180
                for k in range(NUM_KNIGHT_DIRS):
                    out[KNIGHT_PLANES_BASE + knight_map[k]] = arr[KNIGHT_PLANES_BASE + k]
                promo_map = PMAP_PROMOS
                for promo in range(PROMO_CHOICES):
                    b = NSQUARES + promo * PROMO_STRIDE
                    out[b + promo_map[0]] = arr[b + 0]
                    out[b + promo_map[1]] = arr[b + 1]
                    out[b + promo_map[2]] = arr[b + 2]
            else:
                Augment._policy_map_cache[transform] = np.arange(POLICY_OUTPUT, dtype=np.int32)
                return Augment._policy_map_cache[transform]

        elif transform == "vflip_cs":
            arr = base[:, ::-1, :]
            out = arr.copy()
            if planes >= required_planes:
                dir_map = DIR_MAP_VFLIP_CS
                for d in range(NUM_DIRECTIONS):
                    for dist in range(DIR_MAX_DIST):
                        out[dir_map[d] * DIR_MAX_DIST + dist] = arr[d * DIR_MAX_DIST + dist]
                knight_map = KMAP_VFLIP_CS
                for k in range(NUM_KNIGHT_DIRS):
                    out[KNIGHT_PLANES_BASE + knight_map[k]] = arr[KNIGHT_PLANES_BASE + k]
            else:
                Augment._policy_map_cache[transform] = np.arange(POLICY_OUTPUT, dtype=np.int32)
                return Augment._policy_map_cache[transform]
        Augment._policy_map_cache[transform] = out.reshape(-1)
        return Augment._policy_map_cache[transform]

    @staticmethod
    def _feature_plane_indices() -> dict[str, int]:
        turn_plane = HISTORY_LENGTH * PLANES_PER_POSITION
        fullmove_plane = turn_plane + 1
        castling_base = turn_plane + 2
        return {
            "planes_per_pos": PLANES_PER_POSITION,
            "hist_len": HISTORY_LENGTH,
            "turn_plane": turn_plane,
            "fullmove_plane": fullmove_plane,
            "castling_base": castling_base,
        }

    @staticmethod
    def _vflip_cs_plane_permutation(num_planes: int) -> np.ndarray:
        meta = Augment._feature_plane_indices()
        perm = np.arange(num_planes, dtype=np.int32)
        for t in range(meta["hist_len"]):
            base = t * meta["planes_per_pos"]
            for piece in range(6):
                plane_a = base + piece * 2 + 0
                plane_b = base + piece * 2 + 1
                perm[plane_a], perm[plane_b] = perm[plane_b], perm[plane_a]
        castling_plane_base = meta["castling_base"]
        if castling_plane_base + 3 < num_planes:
            perm[castling_plane_base + 0], perm[castling_plane_base + 2] = (
                perm[castling_plane_base + 2],
                perm[castling_plane_base + 0],
            )
            perm[castling_plane_base + 1], perm[castling_plane_base + 3] = (
                perm[castling_plane_base + 3],
                perm[castling_plane_base + 1],
            )
        return perm

    @staticmethod
    def apply(states: list[np.ndarray], policies: list[np.ndarray], transform: str) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
        """Apply augmentation and remap policy indices; returns (states, policies, stm_swapped)."""
        if not states:
            return states, policies, False
        state_batch = np.stack(states, axis=0)
        policy_batch = np.stack(policies, axis=0)
        stm_swapped = False
        if transform == "mirror":
            state_batch = state_batch[..., ::-1].copy()
            policy_batch = policy_batch[:, Augment._policy_index_permutation("mirror")]
        elif transform == "rot180":
            state_batch = state_batch[..., ::-1, ::-1].copy()
            policy_batch = policy_batch[:, Augment._policy_index_permutation("rot180")]
        elif transform == "vflip_cs":
            # Vertical flip + castling-side swap; side-to-move flips.
            state_batch = state_batch[..., ::-1, :].copy()
            perm = Augment._vflip_cs_plane_permutation(state_batch.shape[1])
            state_batch = state_batch[:, perm]
            idx = Augment._feature_plane_indices()
            tp = idx["turn_plane"]
            if tp < state_batch.shape[1]:
                state_batch[:, tp] = 1.0 - state_batch[:, tp]
            policy_batch = policy_batch[:, Augment._policy_index_permutation("vflip_cs")]
            stm_swapped = True
        else:
            return states, policies, False
        out_states = [state_batch[i].copy() for i in range(state_batch.shape[0])]
        out_pols = [policy_batch[i].copy() for i in range(policy_batch.shape[0])]
        return out_states, out_pols, stm_swapped


class SelfPlayEngine:
    """Generates self-play games, builds replay buffer, and samples batches."""

    def __init__(self, evaluator: BatchedEvaluator) -> None:
        self.resign_consecutive = RESIGN_CONSECUTIVE_PLIES
        self.evaluator = evaluator
        self.buffer: deque[tuple[np.ndarray, np.ndarray, np.int8]] = deque(maxlen=REPLAY_BUFFER_CAPACITY)
        self.buffer_lock = threading.Lock()

    @staticmethod
    def encode_u8(enc: np.ndarray) -> np.ndarray:
        """Quantize [0,1] float planes to uint8."""
        x = np.clip(enc, 0.0, 1.0) * U8_SCALE
        return np.rint(x).astype(np.uint8, copy=False)

    @staticmethod
    def encode_value_i8(v: float) -> np.int8:
        """Scale value target to int8."""
        return np.int8(np.clip(np.rint(v * VALUE_I8_SCALE), -int(VALUE_I8_SCALE), int(VALUE_I8_SCALE)))

    def _select_move_by_temperature(self, moves: list[Any], visit_counts: list[int], move_number: int, rng: np.random.Generator | None = None) -> Any:
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
                else:
                    if rng is None:
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
                self.buffer.append((position_u8, counts_u16, SelfPlayEngine.encode_value_i8(target_value)))

    def play_single_game(self, seed: int | None = None) -> tuple[int, int, list[tuple[np.ndarray, np.ndarray]], bool]:
        position = ccore.Position()
        resign_count = 0
        forced_result: int | None = None
        mcts = ccore.MCTS(MCTS_TRAIN_SIMULATIONS_BASE, MCTS_C_PUCT, MCTS_DIRICHLET_ALPHA, MCTS_DIRICHLET_WEIGHT)
        mcts.set_c_puct_params(MCTS_C_PUCT_BASE, MCTS_C_PUCT_INIT)
        mcts.set_fpu_reduction(MCTS_FPU_REDUCTION)
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        import math as _math
        _ = _math.e  # no-op
        mcts.seed(int(rng.integers(2**63 - 1)))

        examples: list[tuple[np.ndarray, np.ndarray]] = []
        position_history: list[Any] = []
        move_count = 0

        random_opening_plies = int(rng.integers(0, max(1, SELFPLAY_OPENING_RANDOM_PLIES_MAX)))
        for _ in range(random_opening_plies):
            if position.result() != ccore.ONGOING:
                break
            moves = position.legal_moves()
            if not moves:
                break
            position.make_move(moves[int(rng.integers(0, len(moves)))])

        first_to_move_is_white: bool | None = None
        while position.result() == ccore.ONGOING and move_count < GAME_MAX_PLIES:
            pos_snapshot = ccore.Position(position)

            sims = max(
                MCTS_TRAIN_SIMULATIONS_MIN,
                MCTS_TRAIN_SIMULATIONS_BASE // (1 + move_count // MCTS_TRAIN_SIM_DECAY_MOVE_INTERVAL),
            )
            mcts.set_simulations(sims)

            if move_count < SELFPLAY_TEMP_MOVES:
                mcts.set_dirichlet_params(MCTS_DIRICHLET_ALPHA, MCTS_DIRICHLET_WEIGHT)
            else:
                mcts.set_dirichlet_params(MCTS_DIRICHLET_ALPHA, 0.0)

            moves = position.legal_moves()
            if not moves:
                break
            visit_counts = mcts.search_batched(position, self.evaluator.infer_positions, EVAL_BATCH_SIZE_MAX)
            if not visit_counts or len(visit_counts) != len(moves):
                break

            policy_counts = np.zeros(POLICY_OUTPUT, dtype=np.uint16)
            for mv, vc in zip(moves, visit_counts, strict=False):
                move_index = ccore.encode_move_index(mv)
                if (move_index is not None) and (0 <= int(move_index) < POLICY_OUTPUT):
                    c = min(int(vc), MCTS_VISIT_COUNT_CLAMP)
                    policy_counts[int(move_index)] = np.uint16(c)

            if position_history:
                history_window = position_history[-HISTORY_LENGTH:] + [pos_snapshot]
                encoded = ccore.encode_batch([history_window])[0]
            else:
                encoded = ccore.encode_position(pos_snapshot)
            encoded_u8 = SelfPlayEngine.encode_u8(encoded)
            policy_counts_u16 = policy_counts
            examples.append((encoded_u8, policy_counts_u16))

            if first_to_move_is_white is None:
                first_to_move_is_white = pos_snapshot.turn == ccore.WHITE

            if self.resign_consecutive > 0:
                _, val_arr = self.evaluator.infer_positions([pos_snapshot])
                value_estimate = float(val_arr[0])
                if value_estimate <= RESIGN_VALUE_THRESHOLD:
                    resign_count += 1
                    if resign_count >= self.resign_consecutive:
                        if float(rng.random()) < RESIGN_PLAYTHROUGH_FRACTION:
                            resign_count = 0
                        else:
                            side_to_move_is_white = position.turn == ccore.WHITE
                            forced_result = ccore.BLACK_WIN if side_to_move_is_white else ccore.WHITE_WIN
                            break
                else:
                    resign_count = 0

            move = self._select_move_by_temperature(moves, visit_counts, move_count, rng=rng)
            position.make_move(move)
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

    def sample_from_snapshot(
        self,
        snapshot: list[tuple[np.ndarray, np.ndarray, np.int8]],
        batch_size: int,
        recent_ratio: float = REPLAY_SNAPSHOT_RECENT_RATIO_DEFAULT,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]] | None:
        sample_count = len(snapshot)
        if batch_size > sample_count:
            return None
        recent_window_count = max(1, int(sample_count * REPLAY_SNAPSHOT_RECENT_WINDOW_FRAC))
        n_recent = int(round(batch_size * recent_ratio))
        n_old = batch_size - n_recent
        recent_indices = np.random.randint(max(0, sample_count - recent_window_count), sample_count, size=n_recent)
        old_indices = np.random.randint(0, max(1, sample_count - recent_window_count), size=n_old)
        indices = np.concatenate([recent_indices, old_indices])
        states_u8_list, counts_u16_list, values_i8_list = zip(*[snapshot[int(i)] for i in indices], strict=False)
        states = [s.astype(np.float32) / U8_SCALE for s in states_u8_list]
        counts = [p.astype(np.float32) for p in counts_u16_list]
        policies: list[np.ndarray] = []
        for c in counts:
            s = float(c.sum())
            if s > 0.0 and np.isfinite(s):
                policies.append(c / s)
            else:
                policies.append(np.full_like(c, 1.0 / max(1, c.size), dtype=np.float32))
        values = [float(v) / VALUE_I8_SCALE for v in values_i8_list]
        return states, policies, values

    def play_games(self, num_games: int) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "games": 0,
            "moves": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
        }
        seeds = [int(np.random.randint(0, 2**63 - 1)) for _ in range(num_games)]
        results: dict[int, tuple[int, int, list[tuple[np.ndarray, np.ndarray]], bool]] = {}
        with ThreadPoolExecutor(max_workers=max(1, SELFPLAY_NUM_WORKERS)) as ex:
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
        return stats
