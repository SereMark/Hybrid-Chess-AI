from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import chesscore as ccore
import numpy as np

from .config import (
    BOARD_SIZE,
    BUFFER_SIZE,
    C_PUCT,
    C_PUCT_BASE,
    C_PUCT_INIT,
    DIR_MAP_MIRROR,
    DIR_MAP_ROT180,
    DIR_MAP_VFLIP_CS,
    DIR_MAX_DIST,
    DIRICHLET_ALPHA,
    DIRICHLET_WEIGHT,
    EVAL_MAX_BATCH,
    FPU_REDUCTION,
    HISTORY_LENGTH,
    KMAP_MIRROR,
    KMAP_ROT180,
    KMAP_VFLIP_CS,
    KNIGHT_PLANES_BASE,
    MAX_GAME_MOVES,
    MCTS_MIN_SIMS,
    NSQUARES,
    NUM_DIRECTIONS,
    NUM_KNIGHT_DIRS,
    OPENING_RANDOM_PLIES_MAX,
    PLANES_PER_POSITION,
    PMAP_PROMOS,
    POLICY_OUTPUT,
    PROMO_CHOICES,
    PROMO_STRIDE,
    RESIGN_CONSECUTIVE,
    RESIGN_PLAYTHROUGH_FRAC,
    RESIGN_THRESHOLD,
    SELFPLAY_WORKERS,
    SIMULATIONS_DECAY_INTERVAL,
    SIMULATIONS_TRAIN,
    SNAPSHOT_RECENT_RATIO_DEFAULT,
    SNAPSHOT_RECENT_WINDOW_FRAC,
    TEMP_DETERMINISTIC_THRESHOLD,
    TEMP_HIGH,
    TEMP_LOW,
    TEMP_MOVES,
    U8_SCALE,
    VALUE_I8_SCALE,
    VISIT_COUNT_CLAMP,
)

if TYPE_CHECKING:
    from .model import BatchedEvaluator


class Augment:
    _policy_map_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _policy_index_map(which: str) -> np.ndarray:
        if which in Augment._policy_map_cache:
            return Augment._policy_map_cache[which]
        assert POLICY_OUTPUT % NSQUARES == 0, "POLICY_OUTPUT must be divisible by NSQUARES"
        planes = POLICY_OUTPUT // NSQUARES
        req = max(
            KNIGHT_PLANES_BASE + NUM_KNIGHT_DIRS,
            NSQUARES + PROMO_STRIDE * PROMO_CHOICES,
        )
        base = np.arange(POLICY_OUTPUT, dtype=np.int32).reshape(planes, BOARD_SIZE, BOARD_SIZE)
        if which == "mirror":
            arr = base[:, :, ::-1]
            out = arr.copy()
            if planes >= req:
                dir_map = DIR_MAP_MIRROR
                for d in range(NUM_DIRECTIONS):
                    for r in range(DIR_MAX_DIST):
                        out[dir_map[d] * DIR_MAX_DIST + r] = arr[d * DIR_MAX_DIST + r]
                kmap = KMAP_MIRROR
                for i in range(NUM_KNIGHT_DIRS):
                    out[KNIGHT_PLANES_BASE + kmap[i]] = arr[KNIGHT_PLANES_BASE + i]
                pmap = PMAP_PROMOS
                for p in range(PROMO_CHOICES):
                    b = NSQUARES + p * PROMO_STRIDE
                    out[b + pmap[0]] = arr[b + 0]
                    out[b + pmap[1]] = arr[b + 1]
                    out[b + pmap[2]] = arr[b + 2]
            else:
                Augment._policy_map_cache[which] = np.arange(POLICY_OUTPUT, dtype=np.int32)
                return Augment._policy_map_cache[which]
        elif which == "rot180":
            arr = base[:, ::-1, ::-1]
            out = arr.copy()
            if planes >= req:
                dir_map = DIR_MAP_ROT180
                for d in range(NUM_DIRECTIONS):
                    for r in range(DIR_MAX_DIST):
                        out[dir_map[d] * DIR_MAX_DIST + r] = arr[d * DIR_MAX_DIST + r]
                kmap = KMAP_ROT180
                for i in range(NUM_KNIGHT_DIRS):
                    out[KNIGHT_PLANES_BASE + kmap[i]] = arr[KNIGHT_PLANES_BASE + i]
                pmap = PMAP_PROMOS
                for p in range(PROMO_CHOICES):
                    b = NSQUARES + p * PROMO_STRIDE
                    out[b + pmap[0]] = arr[b + 0]
                    out[b + pmap[1]] = arr[b + 1]
                    out[b + pmap[2]] = arr[b + 2]
            else:
                Augment._policy_map_cache[which] = np.arange(POLICY_OUTPUT, dtype=np.int32)
                return Augment._policy_map_cache[which]

        elif which == "vflip_cs":
            arr = base[:, ::-1, :]
            out = arr.copy()
            if planes >= req:
                dir_map = DIR_MAP_VFLIP_CS
                for d in range(NUM_DIRECTIONS):
                    for r in range(DIR_MAX_DIST):
                        out[dir_map[d] * DIR_MAX_DIST + r] = arr[d * DIR_MAX_DIST + r]
                kmap = KMAP_VFLIP_CS
                for i in range(NUM_KNIGHT_DIRS):
                    out[KNIGHT_PLANES_BASE + kmap[i]] = arr[KNIGHT_PLANES_BASE + i]
            else:
                Augment._policy_map_cache[which] = np.arange(POLICY_OUTPUT, dtype=np.int32)
                return Augment._policy_map_cache[which]
        Augment._policy_map_cache[which] = out.reshape(-1)
        return Augment._policy_map_cache[which]

    @staticmethod
    def _plane_indices() -> dict[str, int]:
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
    def _vflip_cs_plane_perm(num_planes: int) -> np.ndarray:
        idx = Augment._plane_indices()
        perm = np.arange(num_planes, dtype=np.int32)
        for t in range(idx["hist_len"]):
            base = t * idx["planes_per_pos"]
            for piece in range(6):
                a = base + piece * 2 + 0
                b = base + piece * 2 + 1
                perm[a], perm[b] = perm[b], perm[a]
        cs = idx["castling_base"]
        if cs + 3 < num_planes:
            perm[cs + 0], perm[cs + 2] = perm[cs + 2], perm[cs + 0]
            perm[cs + 1], perm[cs + 3] = perm[cs + 3], perm[cs + 1]
        return perm

    @staticmethod
    def apply(states: list[np.ndarray], policies: list[np.ndarray], which: str) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
        if not states:
            return states, policies, False
        st = np.stack(states, axis=0)
        pol = np.stack(policies, axis=0)
        swapped = False
        if which == "mirror":
            st = st[..., ::-1].copy()
            pol = pol[:, Augment._policy_index_map("mirror")]
        elif which == "rot180":
            st = st[..., ::-1, ::-1].copy()
            pol = pol[:, Augment._policy_index_map("rot180")]
        elif which == "vflip_cs":
            st = st[..., ::-1, :].copy()
            perm = Augment._vflip_cs_plane_perm(st.shape[1])
            st = st[:, perm]
            idx = Augment._plane_indices()
            tp = idx["turn_plane"]
            if tp < st.shape[1]:
                st[:, tp] = 1.0 - st[:, tp]
            pol = pol[:, Augment._policy_index_map("vflip_cs")]
            swapped = True
        else:
            return states, policies, False
        out_states = [st[i].copy() for i in range(st.shape[0])]
        out_pols = [pol[i].copy() for i in range(pol.shape[0])]
        return out_states, out_pols, swapped


class SelfPlayEngine:
    def __init__(self, evaluator: BatchedEvaluator) -> None:
        self.resign_consecutive = RESIGN_CONSECUTIVE
        self.evaluator = evaluator
        self.buffer: deque[tuple[np.ndarray, np.ndarray, np.int8]] = deque(maxlen=BUFFER_SIZE)
        self.buffer_lock = threading.Lock()

    @staticmethod
    def _to_u8_plane(enc: np.ndarray) -> np.ndarray:
        x = np.clip(enc, 0.0, 1.0) * U8_SCALE
        return np.rint(x).astype(np.uint8, copy=False)

    @staticmethod
    def _value_to_i8(v: float) -> np.int8:
        return np.int8(np.clip(np.rint(v * VALUE_I8_SCALE), -int(VALUE_I8_SCALE), int(VALUE_I8_SCALE)))

    def _temp_select(self, moves: list[Any], visits: list[int], move_number: int) -> Any:
        temperature = TEMP_HIGH if move_number < TEMP_MOVES else TEMP_LOW
        if temperature > TEMP_DETERMINISTIC_THRESHOLD:
            probs = np.maximum(np.array(visits, dtype=np.float64), 0.0)
            s = probs.sum()
            if not np.isfinite(s) or s <= 0:
                idx = int(np.argmax(visits))
            else:
                probs = probs ** (1.0 / temperature)
                s = probs.sum()
                if not np.isfinite(s) or s <= 0:
                    idx = int(np.argmax(visits))
                else:
                    idx = int(np.random.choice(len(moves), p=probs / s))
        else:
            idx = int(np.argmax(visits))
        return moves[idx]

    def _process_result(
        self,
        data: list[tuple[np.ndarray, np.ndarray]],
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
            for i, (position_u8, counts_u16) in enumerate(data):
                stm_is_white = ((i % 2) == 0) == bool(first_to_move_is_white)
                v = base if stm_is_white else -base
                self.buffer.append((position_u8, counts_u16, SelfPlayEngine._value_to_i8(v)))

    def play_single_game(self) -> tuple[int, int]:
        position = ccore.Position()
        resign_count = 0
        forced_result: int | None = None
        mcts = ccore.MCTS(SIMULATIONS_TRAIN, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_WEIGHT)
        mcts.set_c_puct_params(C_PUCT_BASE, C_PUCT_INIT)
        mcts.set_fpu_reduction(FPU_REDUCTION)

        data: list[tuple[np.ndarray, np.ndarray]] = []
        history: list[Any] = []
        move_count = 0

        open_plies = int(np.random.randint(0, OPENING_RANDOM_PLIES_MAX))
        for _ in range(open_plies):
            if position.result() != ccore.ONGOING:
                break
            moves = position.legal_moves()
            if not moves:
                break
            position.make_move(moves[int(np.random.randint(0, len(moves)))])

        first_stm_white: bool | None = None
        while position.result() == ccore.ONGOING and move_count < MAX_GAME_MOVES:
            pos_copy = ccore.Position(position)

            sims = max(
                MCTS_MIN_SIMS,
                SIMULATIONS_TRAIN // (1 + move_count // SIMULATIONS_DECAY_INTERVAL),
            )
            mcts.set_simulations(sims)

            if move_count < TEMP_MOVES:
                mcts.set_dirichlet_params(DIRICHLET_ALPHA, DIRICHLET_WEIGHT)
            else:
                mcts.set_dirichlet_params(DIRICHLET_ALPHA, 0.0)

            moves = position.legal_moves()
            if not moves:
                break
            visits = mcts.search_batched(position, self.evaluator.infer_positions, EVAL_MAX_BATCH)
            if not visits or len(visits) != len(moves):
                break

            counts = np.zeros(POLICY_OUTPUT, dtype=np.uint16)
            for mv, vc in zip(moves, visits, strict=False):
                idx = ccore.encode_move_index(mv)
                if (idx is not None) and (0 <= int(idx) < POLICY_OUTPUT):
                    c = min(int(vc), VISIT_COUNT_CLAMP)
                    counts[int(idx)] = np.uint16(c)

            if history:
                histories = history[-HISTORY_LENGTH:] + [pos_copy]
                encoded = ccore.encode_batch([histories])[0]
            else:
                encoded = ccore.encode_position(pos_copy)
            encoded_u8 = SelfPlayEngine._to_u8_plane(encoded)
            counts_u16 = counts
            data.append((encoded_u8, counts_u16))

            if first_stm_white is None:
                first_stm_white = pos_copy.turn == ccore.WHITE

            if self.resign_consecutive > 0:
                _, val_arr = self.evaluator.infer_positions([pos_copy])
                v = float(val_arr[0])
                if v <= RESIGN_THRESHOLD:
                    resign_count += 1
                    if resign_count >= self.resign_consecutive:
                        if np.random.rand() < RESIGN_PLAYTHROUGH_FRAC:
                            resign_count = 0
                        else:
                            stm_white = position.turn == ccore.WHITE
                            forced_result = ccore.BLACK_WIN if stm_white else ccore.WHITE_WIN
                            break
                else:
                    resign_count = 0

            move = self._temp_select(moves, visits, move_count)
            position.make_move(move)
            history.append(pos_copy)
            if len(history) > HISTORY_LENGTH:
                history.pop(0)
            move_count += 1

        final_result = forced_result if forced_result is not None else position.result()

        self._process_result(
            data,
            final_result,
            True if first_stm_white is None else bool(first_stm_white),
        )

        return move_count, final_result

    def snapshot(self) -> list[tuple[np.ndarray, np.ndarray, np.int8]]:
        with self.buffer_lock:
            return list(self.buffer)

    def sample_from_snapshot(
        self,
        snapshot: list[tuple[np.ndarray, np.ndarray, np.int8]],
        batch_size: int,
        recent_ratio: float = SNAPSHOT_RECENT_RATIO_DEFAULT,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]] | None:
        N = len(snapshot)
        if batch_size > N:
            return None
        recent_N = max(1, int(N * SNAPSHOT_RECENT_WINDOW_FRAC))
        n_recent = int(round(batch_size * recent_ratio))
        n_old = batch_size - n_recent
        recent_idx = np.random.randint(max(0, N - recent_N), N, size=n_recent)
        old_idx = np.random.randint(0, max(1, N - recent_N), size=n_old)
        sel_idx = np.concatenate([recent_idx, old_idx])
        states_u8, counts_u16, values_i8 = zip(*[snapshot[int(i)] for i in sel_idx], strict=False)
        states = [s.astype(np.float32) / U8_SCALE for s in states_u8]
        counts = [p.astype(np.float32) for p in counts_u16]
        policies: list[np.ndarray] = []
        for c in counts:
            s = float(c.sum())
            if s > 0.0 and np.isfinite(s):
                policies.append(c / s)
            else:
                policies.append(np.full_like(c, 1.0 / max(1, c.size), dtype=np.float32))
        values = [float(v) / VALUE_I8_SCALE for v in values_i8]
        return states, policies, values

    def play_games(self, num_games: int) -> dict[str, Any]:
        res: dict[str, Any] = {
            "games": 0,
            "moves": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
        }
        with ThreadPoolExecutor(max_workers=max(1, SELFPLAY_WORKERS)) as ex:
            futures = [ex.submit(self.play_single_game) for _ in range(num_games)]
            for fut in as_completed(futures):
                moves, result = fut.result()
                res["games"] += 1
                res["moves"] += moves
                if result == ccore.WHITE_WIN:
                    res["white_wins"] += 1
                elif result == ccore.BLACK_WIN:
                    res["black_wins"] += 1
                else:
                    res["draws"] += 1
        return res
