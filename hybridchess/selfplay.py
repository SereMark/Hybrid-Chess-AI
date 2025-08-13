from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import chesscore as ccore
import numpy as np

from .model import EVAL_MAX_BATCH, HISTORY_LENGTH, PLANES_PER_POSITION, POLICY_OUTPUT

BUFFER_SIZE = 40000
SELFPLAY_WORKERS = 12
MAX_GAME_MOVES = 512
RESIGN_THRESHOLD = -0.9
RESIGN_CONSECUTIVE = 0
TEMP_MOVES = 30
TEMP_HIGH = 1.0
TEMP_LOW = 0.01
TEMP_DETERMINISTIC_THRESHOLD = 0.01
SIMULATIONS_TRAIN = 160
MCTS_MIN_SIMS = 32
SIMULATIONS_DECAY_INTERVAL = 30
C_PUCT = 1.2
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
        fifty_plane = turn_plane + 1
        castling_base = turn_plane + 2
        enpassant_base = castling_base + 4
        return {
            "planes_per_pos": PLANES_PER_POSITION,
            "hist_len": HISTORY_LENGTH,
            "turn_plane": turn_plane,
            "fifty_plane": fifty_plane,
            "castling_base": castling_base,
            "enpassant_base": enpassant_base,
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
    def __init__(self, evaluator: "BatchedEvaluator") -> None:
        self.evaluator = evaluator
        self.buffer: deque[tuple[Any, np.ndarray, float]] = deque(maxlen=BUFFER_SIZE)
        self.buffer_lock = threading.Lock()

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

    def _process_result(self, data: list[tuple[Any, np.ndarray]], result: int) -> None:
        if result == ccore.WHITE_WIN:
            values = [1.0 if i % 2 == 0 else -1.0 for i in range(len(data))]
        elif result == ccore.BLACK_WIN:
            values = [-1.0 if i % 2 == 0 else 1.0 for i in range(len(data))]
        else:
            values = [0.0] * len(data)
        with self.buffer_lock:
            for (position, policy), value in zip(data, values, strict=False):
                self.buffer.append((position, policy, value))

    def play_single_game(self) -> tuple[int, int]:
        position = ccore.Position()
        resign_count = 0
        forced_result: int | None = None
        mcts = ccore.MCTS(SIMULATIONS_TRAIN, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_WEIGHT)
        mcts.set_c_puct_params(C_PUCT_BASE, C_PUCT_INIT)
        data: list[tuple[Any, np.ndarray]] = []
        history: list[Any] = []
        move_count = 0
        while position.result() == ccore.ONGOING and move_count < MAX_GAME_MOVES:
            pos_copy = ccore.Position(position)
            sims = max(
                MCTS_MIN_SIMS,
                SIMULATIONS_TRAIN // (1 + move_count // SIMULATIONS_DECAY_INTERVAL),
            )
            mcts.set_simulations(sims)
            visits = mcts.search_batched(
                position, self.evaluator.infer_positions, EVAL_MAX_BATCH
            )
            if not visits:
                break
            moves = position.legal_moves()
            target = np.zeros(POLICY_OUTPUT, dtype=np.float32)
            for mv, vc in zip(moves, visits, strict=False):
                idx = ccore.encode_move_index(mv)
                if (idx is not None) and (0 <= int(idx) < POLICY_OUTPUT):
                    target[int(idx)] = vc
            ps = target.sum()
            if ps > 0:
                target /= ps
            if history:
                histories = history[-HISTORY_LENGTH:] + [pos_copy]
                encoded = ccore.encode_batch([histories])[0]
            else:
                encoded = ccore.encode_position(pos_copy)
            encoded = encoded.astype(np.float16, copy=False)
            target = target.astype(np.float16, copy=False)
            data.append((encoded, target))
            if RESIGN_CONSECUTIVE > 0:
                _, val_arr = self.evaluator.infer_positions([pos_copy])
                v = float(val_arr[0])
                if v <= RESIGN_THRESHOLD:
                    resign_count += 1
                    if resign_count >= RESIGN_CONSECUTIVE:
                        try:
                            stm_white = bool(
                                getattr(
                                    position,
                                    "white_to_move",
                                    (move_count % 2) == 0,
                                )
                            )
                        except Exception:
                            stm_white = (move_count % 2) == 0
                        forced_result = (
                            ccore.BLACK_WIN if stm_white else ccore.WHITE_WIN
                        )
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
        self._process_result(data, final_result)
        return move_count, final_result

    def snapshot(self) -> list[tuple[Any, np.ndarray, float]]:
        with self.buffer_lock:
            return list(self.buffer)

    def sample_from_snapshot(
        self, snapshot: list[tuple[Any, np.ndarray, float]], batch_size: int
    ):
        if len(snapshot) < batch_size:
            return None
        idx = np.random.randint(0, len(snapshot), size=batch_size)
        batch = [snapshot[int(i)] for i in idx]
        s, p, v = zip(*batch, strict=False)
        return list(s), list(p), list(v)

    def play_games(self, num_games: int) -> dict[str, int | float]:
        res: dict[str, int | float] = {
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
