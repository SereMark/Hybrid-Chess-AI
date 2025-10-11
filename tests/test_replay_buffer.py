from __future__ import annotations

import numpy as np
from replay_buffer import ReplayBuffer


def test_replay_buffer_push_and_sample() -> None:
    buffer = ReplayBuffer(capacity=4, planes=2, height=2, width=2)
    state = np.ones((2, 2, 2), dtype=np.uint8)
    indices = np.array([1, 2], dtype=np.int32)
    counts = np.array([10, 5], dtype=np.uint16)
    for _ in range(3):
        buffer.push(state, indices, counts, 1)
    states, idx_list, cnt_list, values = buffer.sample(batch_size=2, recent_ratio=1.0, recent_window_frac=1.0)
    assert len(states) == len(idx_list) == len(cnt_list) == len(values)
    assert values[0] == 1


def test_replay_buffer_resize_and_clear() -> None:
    buffer = ReplayBuffer(capacity=2, planes=1, height=1, width=1)
    state = np.zeros((1, 1, 1), dtype=np.uint8)
    buffer.push(state, [0], [1], 1)
    buffer.push(state, [0], [1], 2)
    buffer.set_capacity(4)
    assert buffer.capacity == 4
    assert buffer.size == 2
    buffer.clear()
    assert buffer.size == 0
    assert buffer.sample(batch_size=1, recent_ratio=1.0, recent_window_frac=1.0) == ([], [], [], [])
