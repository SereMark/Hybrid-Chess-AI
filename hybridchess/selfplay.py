from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import chesscore as ccore
import numpy as np

from .config import CONFIG

if TYPE_CHECKING:
    from .model import BatchedEvaluator


class Augment:
    @staticmethod
    def mirror_policy(pi: np.ndarray) -> np.ndarray:
        policy_planes = CONFIG.policy_output // 64
        arr = pi.reshape((policy_planes, 8, 8))
        arr = arr[:, :, ::-1].copy()
        out = np.empty_like(arr)
        dir_map = [2, 1, 0, 4, 3, 7, 6, 5]
        for d in range(8):
            for r in range(7):
                old_idx = d * 7 + r
                new_idx = dir_map[d] * 7 + r
                out[new_idx] = arr[old_idx]
        kmap = [1, 0, 3, 2, 5, 4, 7, 6]
        for i in range(8):
            out[56 + kmap[i]] = arr[56 + i]
        pmap = [0, 2, 1]
        for p in range(3):
            base = 64 + p * 3
            out[base + pmap[0]] = arr[base + 0]
            out[base + pmap[1]] = arr[base + 1]
            out[base + pmap[2]] = arr[base + 2]
        return out.reshape(-1)

    @staticmethod
    def rotate180_policy(pi: np.ndarray) -> np.ndarray:
        policy_planes = CONFIG.policy_output // 64
        arr = pi.reshape((policy_planes, 8, 8))
        arr = arr[:, ::-1, ::-1].copy()
        out = np.empty_like(arr)
        dir_map = [7, 6, 5, 4, 3, 2, 1, 0]
        for d in range(8):
            for r in range(7):
                old_idx = d * 7 + r
                new_idx = dir_map[d] * 7 + r
                out[new_idx] = arr[old_idx]
        kmap = [7, 6, 5, 4, 3, 2, 1, 0]
        for i in range(8):
            out[56 + kmap[i]] = arr[56 + i]
        for p in range(3):
            base = 64 + p * 3
            out[base + 0] = arr[base + 0]
            out[base + 1] = arr[base + 1]
            out[base + 2] = arr[base + 2]
        return out.reshape(-1)

    @staticmethod
    def vflip_colorswap_policy(pi: np.ndarray) -> np.ndarray:
        policy_planes = CONFIG.policy_output // 64
        arr = pi.reshape((policy_planes, 8, 8))
        arr = arr[:, ::-1, :].copy()
        out = np.empty_like(arr)
        dir_map = [5, 6, 7, 4, 3, 2, 1, 0]
        for d in range(8):
            for r in range(7):
                old_idx = d * 7 + r
                new_idx = dir_map[d] * 7 + r
                out[new_idx] = arr[old_idx]
        kmap = [6, 7, 4, 5, 2, 3, 0, 1]
        for i in range(8):
            out[56 + kmap[i]] = arr[56 + i]
        for p in range(3):
            base = 64 + p * 3
            out[base + 0] = arr[base + 0]
            out[base + 1] = arr[base + 1]
            out[base + 2] = arr[base + 2]
        return out.reshape(-1)

    @staticmethod
    def apply(
        states: list[np.ndarray], policies: list[np.ndarray], which: str
    ) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
        if which == "mirror":
            states = [s[..., ::-1].copy() for s in states]
            policies = [Augment.mirror_policy(p) for p in policies]
            return states, policies, False
        elif which == "rot180":
            states = [s[..., ::-1, ::-1].copy() for s in states]
            policies = [Augment.rotate180_policy(p) for p in policies]
            return states, policies, False
        elif which == "vflip_cs":
            out_states: list[np.ndarray] = []
            for s in states:
                x = s[..., ::-1, :].copy()
                planes_per_pos = CONFIG.planes_per_position
                hist_len = CONFIG.history_length
                for t in range(hist_len):
                    base = t * planes_per_pos
                    for piece in range(6):
                        a = base + piece * 2 + 0
                        b = base + piece * 2 + 1
                        xa = x[a].copy()
                        x[a] = x[b]
                        x[b] = xa
                turn_plane = hist_len * planes_per_pos
                x[turn_plane] = 1.0 - x[turn_plane]
                cs_base = turn_plane + 2
                xa = x[cs_base + 0].copy()
                x[cs_base + 0] = x[cs_base + 2]
                x[cs_base + 2] = xa
                xb = x[cs_base + 1].copy()
                x[cs_base + 1] = x[cs_base + 3]
                x[cs_base + 3] = xb
                out_states.append(x)
            policies = [Augment.vflip_colorswap_policy(p) for p in policies]
            return out_states, policies, True
        return states, policies, False


class SelfPlayEngine:
    def __init__(self, evaluator: "BatchedEvaluator") -> None:
        self.evaluator = evaluator
        self.buffer: deque[tuple[Any, np.ndarray, float]] = deque(
            maxlen=CONFIG.buffer_size
        )
        self.buffer_lock = threading.Lock()

    def _temp_select(
        self, moves: list[Any], visits: list[int], move_number: int
    ) -> Any:
        if move_number < CONFIG.temp_moves:
            temperature = CONFIG.temp_high
        else:
            temperature = CONFIG.temp_low
        if temperature > CONFIG.temp_deterministic_threshold:
            probs = np.array(visits, dtype=np.float64)
            probs = np.maximum(probs, 0)
            s = probs.sum()
            if not np.isfinite(s) or s <= 0:
                move_idx = int(np.argmax(visits))
            else:
                probs = probs ** (1.0 / temperature)
                s = probs.sum()
                if not np.isfinite(s) or s <= 0:
                    move_idx = int(np.argmax(visits))
                else:
                    probs /= s
                    move_idx = int(np.random.choice(len(moves), p=probs))
        else:
            move_idx = int(np.argmax(visits))
        return moves[move_idx]

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
        self._resign_count = 0
        mcts = ccore.MCTS(
            CONFIG.simulations_train,
            CONFIG.c_puct,
            CONFIG.dirichlet_alpha,
            CONFIG.dirichlet_weight,
        )
        mcts.set_c_puct_params(CONFIG.c_puct_base, CONFIG.c_puct_init)
        data: list[tuple[Any, np.ndarray]] = []
        history: list[Any] = []
        move_count = 0
        while position.result() == ccore.ONGOING and move_count < CONFIG.max_game_moves:
            pos_copy = ccore.Position(position)
            sims = max(
                CONFIG.mcts_min_sims,
                CONFIG.simulations_train
                // (1 + move_count // CONFIG.simulations_decay_interval),
            )
            mcts.set_simulations(sims)
            visits = mcts.search_batched(
                position, self.evaluator.infer_positions, CONFIG.eval_max_batch
            )
            if not visits:
                break

            moves = position.legal_moves()
            target = np.zeros(CONFIG.policy_output, dtype=np.float32)
            for move, visit_count in zip(moves, visits, strict=False):
                move_index = ccore.encode_move_index(move)
                if (move_index is not None) and (
                    0 <= int(move_index) < CONFIG.policy_output
                ):
                    target[move_index] = visit_count
            policy_sum = target.sum()
            if policy_sum > 0:
                target /= policy_sum

            if history:
                histories = history[-CONFIG.history_length :] + [pos_copy]
                encoded = ccore.encode_batch([histories])[0]
            else:
                encoded = ccore.encode_position(pos_copy)
            encoded = encoded.astype(np.float16, copy=False)
            target = target.astype(np.float16, copy=False)
            data.append((encoded, target))
            if CONFIG.resign_consecutive > 0:
                _, val_arr = self.evaluator.infer_positions([pos_copy])
                value_est = float(val_arr[0])
                if value_est <= CONFIG.resign_threshold:
                    consecutive = self._resign_count + 1
                    self._resign_count = consecutive
                    if consecutive >= CONFIG.resign_consecutive:
                        break
                else:
                    self._resign_count = 0

            move = self._temp_select(moves, visits, move_count)
            position.make_move(move)
            history.append(pos_copy)
            if len(history) > CONFIG.history_length:
                history.pop(0)
            move_count += 1

        self._process_result(data, position.result())
        return move_count, position.result()

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
        results: dict[str, int | float] = {
            "games": 0,
            "moves": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
        }
        workers = max(1, CONFIG.selfplay_workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self.play_single_game) for _ in range(num_games)]
            for fut in as_completed(futures):
                moves, result = fut.result()
                results["games"] += 1
                results["moves"] += moves
                if result == ccore.WHITE_WIN:
                    results["white_wins"] += 1
                elif result == ccore.BLACK_WIN:
                    results["black_wins"] += 1
                else:
                    results["draws"] += 1
        return results
