"""Experience replay buffer for self-play training samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import DTypeLike

__all__ = ["ReplayBuffer"]


@dataclass(slots=True)
class _Entry:
    indices: np.ndarray
    counts: np.ndarray
    value: np.int8


class ReplayBuffer:
    """Ring buffer storing encoded board states and sparse policy targets."""

    def __init__(self, capacity: int, planes: int, height: int, width: int) -> None:
        capacity = int(max(1, capacity))
        self._capacity = capacity
        self._state_shape = (int(planes), int(height), int(width))

        self._states = np.zeros((capacity,) + self._state_shape, dtype=np.uint8)
        self._entries: list[_Entry | None] = [None] * capacity
        self._values = np.zeros((capacity,), dtype=np.int8)

        self._size = 0
        self._head = 0
        self._rng = np.random.default_rng(seed=0xC0FFEE)

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

    def push(
        self,
        state: np.ndarray,
        indices: Sequence[int] | np.ndarray,
        counts: Sequence[int] | np.ndarray,
        value: int | np.integer,
    ) -> None:
        pos = self._head
        self._states[pos] = self._validate_state(state)
        idx_arr = self._validate_sparse(indices, dtype=np.int32)
        cnt_arr = self._validate_sparse(counts, dtype=np.uint16)
        if idx_arr.shape != cnt_arr.shape:
            slc = slice(0, min(idx_arr.size, cnt_arr.size))
            idx_arr = idx_arr[slc]
            cnt_arr = cnt_arr[slc]
        value_i8 = np.int8(value)
        self._entries[pos] = _Entry(idx_arr.copy(), cnt_arr.copy(), value_i8)
        self._values[pos] = value_i8

        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def set_capacity(self, capacity: int) -> None:
        capacity = int(max(1, capacity))
        if capacity == self._capacity:
            return

        positions = self._ordered_positions()
        keep = min(len(positions), capacity)
        positions = positions[-keep:]

        new_states = np.zeros((capacity,) + self._state_shape, dtype=np.uint8)
        new_entries: list[_Entry | None] = [None] * capacity
        new_values = np.zeros((capacity,), dtype=np.int8)

        for idx, pos in enumerate(positions):
            new_states[idx] = self._states[pos]
            entry = self._entries[pos]
            if entry is None:
                continue
            new_entries[idx] = _Entry(
                entry.indices.copy(), entry.counts.copy(), np.int8(entry.value)
            )
            new_values[idx] = np.int8(entry.value)

        self._capacity = capacity
        self._states = new_states
        self._entries = new_entries
        self._values = new_values
        self._size = keep
        self._head = keep % capacity

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
        recent_window = max(1, int(round(self._size * float(recent_window_frac))))
        recent_window = min(recent_window, self._size)

        recent_candidates = ordered[-recent_window:]
        old_candidates = ordered[:-recent_window] or recent_candidates

        recent_samples = int(round(batch_size * float(recent_ratio)))
        recent_samples = max(0, min(recent_samples, batch_size))
        old_samples = batch_size - recent_samples

        picks: list[int] = []
        if recent_samples > 0 and recent_candidates:
            picks.extend(
                self._rng.choice(
                    recent_candidates, size=recent_samples, replace=True
                ).tolist()
            )
        if old_samples > 0:
            picks.extend(
                self._rng.choice(old_candidates, size=old_samples, replace=True).tolist()
            )

        self._rng.shuffle(picks)

        states: list[np.ndarray] = []
        idx_list: list[np.ndarray] = []
        cnt_list: list[np.ndarray] = []
        values: list[np.int8] = []

        for pos in picks:
            entry = self._entries[pos]
            if entry is None:
                continue
            states.append(self._states[pos].copy())
            idx_list.append(entry.indices.copy())
            cnt_list.append(entry.counts.copy())
            values.append(np.int8(entry.value))

        return states, idx_list, cnt_list, values

    def _ordered_positions(self) -> list[int]:
        if self._size == 0:
            return []
        start = (self._head - self._size) % self._capacity
        return [(start + i) % self._capacity for i in range(self._size)]

    def _validate_state(self, state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=np.uint8)
        if arr.shape != self._state_shape:
            raise ValueError(
                f"state shape {arr.shape} does not match {self._state_shape}"
            )
        return arr

    @staticmethod
    def _validate_sparse(values: Iterable[int], *, dtype: DTypeLike) -> np.ndarray:
        arr = np.asarray(values, dtype=dtype)
        if arr.ndim != 1:
            raise ValueError("sparse arrays must be 1D")
        return arr
