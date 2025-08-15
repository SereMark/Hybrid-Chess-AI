from __future__ import annotations

import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import chesscore as ccore
import numpy as np

from .model import EVAL_MAX_BATCH, HISTORY_LENGTH, PLANES_PER_POSITION, POLICY_OUTPUT

BUFFER_SIZE = 300_000
SELFPLAY_WORKERS = 10
MAX_GAME_MOVES = 512
RESIGN_THRESHOLD = -0.9
RESIGN_CONSECUTIVE = 0
TEMP_MOVES = 40
TEMP_HIGH = 1.0
TEMP_LOW = 0.01
TEMP_DETERMINISTIC_THRESHOLD = 0.01
SIMULATIONS_TRAIN = 384
MCTS_MIN_SIMS = 64
SIMULATIONS_DECAY_INTERVAL = 60
C_PUCT = 1.25
C_PUCT_BASE = 19652.0
C_PUCT_INIT = 1.25
DIRICHLET_ALPHA = 0.3
DIRICHLET_WEIGHT = 0.25
if TYPE_CHECKING:
    from .model import BatchedEvaluator


class Augment:
    _policy_map_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _policy_index_map(which: str) -> np.ndarray:
        if which in Augment._policy_map_cache:
            return Augment._policy_map_cache[which]
        planes = POLICY_OUTPUT // 64
        base = np.arange(POLICY_OUTPUT, dtype=np.int32).reshape(planes, 8, 8).copy()
        if which == "mirror":
            arr = base[:, :, ::-1]
            out = np.empty_like(arr)
            dir_map = [2, 1, 0, 4, 3, 7, 6, 5]
            for d in range(8):
                for r in range(7):
                    out[dir_map[d] * 7 + r] = arr[d * 7 + r]
            kmap = [1, 0, 3, 2, 5, 4, 7, 6]
            for i in range(8):
                out[56 + kmap[i]] = arr[56 + i]
            pmap = [0, 2, 1]
            for p in range(3):
                b = 64 + p * 3
                out[b + pmap[0]] = arr[b + 0]
                out[b + pmap[1]] = arr[b + 1]
                out[b + pmap[2]] = arr[b + 2]
        elif which == "rot180":
            arr = base[:, ::-1, ::-1]
            out = np.empty_like(arr)
            dir_map = [7, 6, 5, 4, 3, 2, 1, 0]
            for d in range(8):
                for r in range(7):
                    out[dir_map[d] * 7 + r] = arr[d * 7 + r]
            kmap = [7, 6, 5, 4, 3, 2, 1, 0]
            for i in range(8):
                out[56 + kmap[i]] = arr[56 + i]
            for p in range(3):
                b = 64 + p * 3
                out[b + 0] = arr[b + 0]
                out[b + 1] = arr[b + 1]
                out[b + 2] = arr[b + 2]
        elif which == "vflip_cs":
            arr = base[:, ::-1, :]
            out = np.empty_like(arr)
            dir_map = [5, 6, 7, 4, 3, 2, 1, 0]
            for d in range(8):
                for r in range(7):
                    out[dir_map[d] * 7 + r] = arr[d * 7 + r]
            kmap = [6, 7, 4, 5, 2, 3, 0, 1]
            for i in range(8):
                out[56 + kmap[i]] = arr[56 + i]
            for p in range(3):
                b = 64 + p * 3
                out[b + 0] = arr[b + 0]
                out[b + 1] = arr[b + 1]
                out[b + 2] = arr[b + 2]
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
        perm[cs + 0], perm[cs + 2] = perm[cs + 2], perm[cs + 0]
        perm[cs + 1], perm[cs + 3] = perm[cs + 3], perm[cs + 1]
        return perm

    @staticmethod
    def apply(
        states: list[np.ndarray], policies: list[np.ndarray], which: str
    ) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
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
        self.evaluator = evaluator
        self.buffer: deque[tuple[np.ndarray, np.ndarray, np.int8]] = deque(
            maxlen=BUFFER_SIZE
        )
        self.buffer_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self._metrics = {
            "games_total": 0,
            "moves_total": 0,
            "resigns_total": 0,
            "forced_results_total": 0,
            "mcts_sims_total": 0,
            "mcts_calls_total": 0,
            "mcts_batch_max": 0,
            "temp_moves_high_total": 0,
            "temp_moves_low_total": 0,
        }

    @staticmethod
    def _to_u8_plane(enc: np.ndarray) -> np.ndarray:
        x = np.clip(enc, 0.0, 1.0) * 255.0
        return np.rint(x).astype(np.uint8, copy=False)

    @staticmethod
    def _value_to_i8(v: float) -> np.int8:
        return np.int8(np.clip(np.rint(v * 127.0), -127, 127))

    def _temp_select(
        self, moves: list[Any], visits: list[int], move_number: int
    ) -> Any:
        temperature = TEMP_HIGH if move_number < TEMP_MOVES else TEMP_LOW
        if temperature > TEMP_DETERMINISTIC_THRESHOLD:
            probs = np.maximum(np.array(visits, dtype=np.float64), 0)
            s = probs.sum()
            if not np.isfinite(s) or s <= 0:
                idx = int(np.argmax(visits))
            else:
                probs = probs ** (1.0 / temperature)
                s = probs.sum()
                idx = (
                    int(np.argmax(visits))
                    if (not np.isfinite(s) or s <= 0)
                    else int(np.random.choice(len(moves), p=probs / s))
                )
        else:
            idx = int(np.argmax(visits))
        return moves[idx]

    def _process_result(
        self, data: list[tuple[np.ndarray, np.ndarray]], result: int
    ) -> None:
        if result == ccore.WHITE_WIN:
            values = [1.0 if i % 2 == 0 else -1.0 for i in range(len(data))]
        elif result == ccore.BLACK_WIN:
            values = [-1.0 if i % 2 == 0 else 1.0 for i in range(len(data))]
        else:
            values = [0.0] * len(data)
        with self.buffer_lock:
            for (position_u8, counts_u8), value in zip(data, values, strict=False):
                self.buffer.append(
                    (position_u8, counts_u8, SelfPlayEngine._value_to_i8(value))
                )

    def play_single_game(self) -> tuple[int, int]:
        position = ccore.Position()
        resign_count = 0
        forced_result: int | None = None
        mcts = ccore.MCTS(SIMULATIONS_TRAIN, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_WEIGHT)
        mcts.set_c_puct_params(C_PUCT_BASE, C_PUCT_INIT)
        data: list[tuple[np.ndarray, np.ndarray]] = []
        history: list[Any] = []
        move_count = 0
        local_mcts_sims_total = 0
        local_mcts_calls_total = 0
        local_mcts_batch_max = 0
        local_temp_high = 0
        local_temp_low = 0
        local_resigns = 0
        local_forced_results = 0
        open_plies = int(np.random.randint(0, 7))
        for _ in range(open_plies):
            if position.result() != ccore.ONGOING:
                break
            moves = position.legal_moves()
            if not moves:
                break
            position.make_move(moves[int(np.random.randint(0, len(moves)))])
        while position.result() == ccore.ONGOING and move_count < MAX_GAME_MOVES:
            pos_copy = ccore.Position(position)
            sims = max(
                MCTS_MIN_SIMS,
                SIMULATIONS_TRAIN // (1 + move_count // SIMULATIONS_DECAY_INTERVAL),
            )
            mcts.set_simulations(sims)
            local_mcts_sims_total += int(sims)
            local_mcts_calls_total += 1
            visits = mcts.search_batched(
                position, self.evaluator.infer_positions, EVAL_MAX_BATCH
            )
            if not visits:
                break
            moves = position.legal_moves()
            if moves:
                local_mcts_batch_max = max(local_mcts_batch_max, len(moves))
            counts = np.zeros(POLICY_OUTPUT, dtype=np.uint8)
            for mv, vc in zip(moves, visits, strict=False):
                idx = ccore.encode_move_index(mv)
                if (idx is not None) and (0 <= int(idx) < POLICY_OUTPUT):
                    c = min(int(vc), 255)
                    counts[int(idx)] = np.uint8(c)
            if history:
                histories = history[-HISTORY_LENGTH:] + [pos_copy]
                encoded = ccore.encode_batch([histories])[0]
            else:
                encoded = ccore.encode_position(pos_copy)
            encoded_u8 = SelfPlayEngine._to_u8_plane(encoded)
            counts_u8 = counts
            data.append((encoded_u8, counts_u8))
            if RESIGN_CONSECUTIVE > 0:
                _, val_arr = self.evaluator.infer_positions([pos_copy])
                v = float(val_arr[0])
                if v <= RESIGN_THRESHOLD:
                    resign_count += 1
                    if resign_count >= RESIGN_CONSECUTIVE:
                        stm_white = position.turn == ccore.WHITE
                        forced_result = (
                            ccore.BLACK_WIN if stm_white else ccore.WHITE_WIN
                        )
                        local_resigns += 1
                        break
                else:
                    resign_count = 0
            move = self._temp_select(moves, visits, move_count)
            if move_count < TEMP_MOVES:
                local_temp_high += 1
            else:
                local_temp_low += 1
            position.make_move(move)
            history.append(pos_copy)
            if len(history) > HISTORY_LENGTH:
                history.pop(0)
            move_count += 1
        final_result = forced_result if forced_result is not None else position.result()
        if forced_result is not None:
            local_forced_results += 1
        self._process_result(data, final_result)
        with self.metrics_lock:
            self._metrics["games_total"] += 1
            self._metrics["moves_total"] += int(move_count)
            self._metrics["mcts_sims_total"] += local_mcts_sims_total
            self._metrics["mcts_calls_total"] += local_mcts_calls_total
            self._metrics["mcts_batch_max"] = max(
                int(self._metrics["mcts_batch_max"]), local_mcts_batch_max
            )
            self._metrics["temp_moves_high_total"] += local_temp_high
            self._metrics["temp_moves_low_total"] += local_temp_low
            self._metrics["resigns_total"] += local_resigns
            self._metrics["forced_results_total"] += local_forced_results
        return move_count, final_result

    def snapshot(self) -> list[tuple[np.ndarray, np.ndarray, np.int8]]:
        with self.buffer_lock:
            return list(self.buffer)

    def sample_from_snapshot(
        self,
        snapshot: list[tuple[np.ndarray, np.ndarray, np.int8]],
        batch_size: int,
        recent_ratio: float = 0.6,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]] | None:
        N = len(snapshot)
        if batch_size > N:
            return None
        recent_N = max(1, int(N * 0.2))
        n_recent = int(round(batch_size * recent_ratio))
        n_old = batch_size - n_recent
        recent_idx = np.random.randint(max(0, N - recent_N), N, size=n_recent)
        old_idx = np.random.randint(0, max(1, N - recent_N), size=n_old)
        sel_idx = np.concatenate([recent_idx, old_idx])
        states_u8, counts_u8, values_i8 = zip(
            *[snapshot[int(i)] for i in sel_idx], strict=False
        )
        states = [s.astype(np.float32) / 255.0 for s in states_u8]
        counts = [p.astype(np.float32) for p in counts_u8]
        policies: list[np.ndarray] = []
        for c in counts:
            s = float(c.sum())
            if s > 0.0 and np.isfinite(s):
                policies.append(c / s)
            else:
                policies.append(np.full_like(c, 1.0 / max(1, c.size), dtype=np.float32))
        values = [float(v) / 127.0 for v in values_i8]
        return states, policies, values

    def play_games(self, num_games: int) -> dict[str, int | float]:
        res: dict[str, int | float] = {
            "games": 0,
            "moves": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
        }
        with self.metrics_lock:
            sp_start = dict(self._metrics)
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
        with self.metrics_lock:
            sp_end = dict(self._metrics)
        res["sp_metrics"] = sp_end
        sp_delta: dict[str, int | float] = {}
        for k, v in sp_end.items():
            if k in sp_start and isinstance(v, int | float):
                sp_delta[k] = float(v) - float(sp_start.get(k, 0.0))
        res["sp_metrics_iter"] = sp_delta
        return res
