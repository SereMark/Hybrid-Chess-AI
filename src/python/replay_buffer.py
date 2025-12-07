from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
from numpy.typing import DTypeLike

__all__ = ["ReplayBuffer"]


@dataclass(slots=True)
class _Entry:
    indices: np.ndarray
    counts: np.ndarray
    value: np.int8


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        planes: int,
        height: int,
        width: int,
        *,
        seed: int | np.random.SeedSequence | None = None,
    ) -> None:
        capacity = int(max(1, capacity))
        self._capacity = capacity
        self._state_shape = (int(planes), int(height), int(width))
        self._states = np.zeros((capacity,) + self._state_shape, dtype=np.uint8)
        self._entries: list[_Entry | None] = [None] * capacity
        self._size = 0
        self._head = 0
        self._rng = np.random.default_rng(seed)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    def seed(self, seed: int | np.random.SeedSequence | None) -> None:
        self._rng = np.random.default_rng(seed)

    def clear(self) -> None:
        self._size = 0
        self._head = 0
        self._entries = [None] * self._capacity

    def set_capacity(self, capacity: int) -> None:
        capacity = int(max(1, capacity))
        if capacity == self._capacity:
            return

        positions = self._ordered_positions()
        keep = min(len(positions), capacity)
        positions = positions[-keep:]

        new_states = np.zeros((capacity,) + self._state_shape, dtype=np.uint8)
        new_entries: list[_Entry | None] = [None] * capacity
        for dst, src in enumerate(positions):
            new_states[dst] = self._states[src]
            entry = self._entries[src]
            if entry is not None:
                new_entries[dst] = _Entry(entry.indices.copy(), entry.counts.copy(), np.int8(entry.value))

        self._capacity = capacity
        self._states = new_states
        self._entries = new_entries
        self._size = keep
        self._head = keep % capacity

    def push(
        self,
        state: np.ndarray,
        indices: Sequence[int] | np.ndarray,
        counts: Sequence[int] | np.ndarray,
        value: int | np.integer,
    ) -> None:
        position = self._head
        self._states[position] = self._validate_state(state)

        idx_arr = self._validate_sparse(indices, dtype=np.int32)
        cnt_arr = self._validate_sparse(counts, dtype=np.uint16)
        if idx_arr.shape != cnt_arr.shape:
            limit = min(idx_arr.size, cnt_arr.size)
            idx_arr = idx_arr[:limit]
            cnt_arr = cnt_arr[:limit]

        self._entries[position] = _Entry(idx_arr.copy(), cnt_arr.copy(), np.int8(value))
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def sample(
        self,
        batch_size: int,
        recent_ratio: float,
        recent_window_frac: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.int8]]:
        if self._size == 0:
            return ([], [], [], [])

        batch_size = min(int(batch_size), self._size)
        if batch_size <= 0:
            return ([], [], [], [])

        ordered = self._ordered_positions()
        window = min(max(1, int(round(self._size * float(recent_window_frac)))), self._size)

        recent_candidates = ordered[-window:]
        stale_candidates = ordered[:-window] or recent_candidates

        recent_samples = max(0, min(int(round(batch_size * float(recent_ratio))), batch_size))
        stale_samples = batch_size - recent_samples

        picks: list[int] = []
        if recent_samples > 0:
            picks.extend(self._rng.choice(recent_candidates, size=recent_samples, replace=True).tolist())
        if stale_samples > 0:
            picks.extend(self._rng.choice(stale_candidates, size=stale_samples, replace=True).tolist())
        self._rng.shuffle(picks)

        if not picks:
            return ([], [], [], [])

        picks_arr = np.asarray(picks, dtype=np.intp)
        states_arr = self._states[picks_arr].copy()

        states_out: list[np.ndarray] = [states_arr[i].copy() for i in range(states_arr.shape[0])]
        indices_out: list[np.ndarray] = []
        counts_out: list[np.ndarray] = []
        values_out: list[np.int8] = []

        for pos in picks:
            entry = self._entries[pos]
            if entry is None:
                continue
            indices_out.append(entry.indices.copy())
            counts_out.append(entry.counts.copy())
            values_out.append(np.int8(entry.value))

        return states_out, indices_out, counts_out, values_out

    def _ordered_positions(self) -> list[int]:
        if self._size == 0:
            return []
        start = (self._head - self._size) % self._capacity
        return [(start + offset) % self._capacity for offset in range(self._size)]

    def _validate_state(self, state: np.ndarray) -> np.ndarray:
        array = np.asarray(state, dtype=np.uint8)
        if array.shape != self._state_shape:
            raise ValueError(f"a state tömb alakja {array.shape} nem egyezik a várt {self._state_shape} alakzattal")
        return array

    @staticmethod
    def _validate_sparse(values: Iterable[int], *, dtype: DTypeLike) -> np.ndarray:
        array = np.asarray(values, dtype=dtype)
        if array.ndim != 1:
            raise ValueError("a ritka (sparse) tömböknek egydimenziósaknak kell lenniük")
        return array

    def state_dict(self) -> dict[str, Any]:
        entries: list[dict[str, Any] | None] = []
        for entry in self._entries:
            if entry is None:
                entries.append(None)
            else:
                entries.append(
                    {
                        "indices": entry.indices.copy(),
                        "counts": entry.counts.copy(),
                        "value": int(entry.value),
                    }
                )
        return {
            "capacity": self._capacity,
            "state_shape": self._state_shape,
            "states": self._states.copy(),
            "entries": entries,
            "size": self._size,
            "head": self._head,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, data: Mapping[str, Any]) -> None:
        capacity = int(data["capacity"])
        state_shape_raw = tuple(int(v) for v in data["state_shape"])
        if len(state_shape_raw) != 3:
            raise ValueError(f"expected 3D state shape, got {state_shape_raw!r}")
        state_shape = cast(tuple[int, int, int], state_shape_raw)
        if state_shape != self._state_shape or capacity != self._capacity:
            self._capacity = capacity
            self._state_shape = state_shape
            self._states = np.zeros((capacity,) + state_shape, dtype=np.uint8)
            self._entries = [None] * capacity

        states_arr = np.asarray(data["states"], dtype=np.uint8)
        expected_shape = (capacity,) + state_shape
        if states_arr.shape != expected_shape:
            raise ValueError(f"a state tömb alakja {states_arr.shape} nem egyezik a várt {expected_shape} alakzattal")
        self._states[:] = states_arr

        entries_serialized = list(data["entries"])
        if len(entries_serialized) != capacity:
            raise ValueError("az entries lista hossza nem egyezik a kapacitással")

        self._entries = []
        for item in entries_serialized:
            if item is None:
                self._entries.append(None)
                continue
            indices = np.asarray(item["indices"], dtype=np.int32)
            counts = np.asarray(item["counts"], dtype=np.uint16)
            value = np.int8(int(item["value"]))
            self._entries.append(_Entry(indices, counts, value))

        size_raw = int(data["size"])
        head_raw = int(data["head"])
        self._size = max(0, min(size_raw, self._capacity))
        if self._size == 0:
            self._head = 0
        else:
            self._head = max(0, min(head_raw, self._capacity - 1))

        rng_state = data.get("rng_state")
        if rng_state is not None:
            self._rng = np.random.default_rng()
            self._rng.bit_generator.state = rng_state
