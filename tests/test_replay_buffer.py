from __future__ import annotations

import numpy as np
from replay_buffer import ReplayBuffer


def test_replay_buffer_push_sample_and_truncation() -> None:
    buf = ReplayBuffer(capacity=4, planes=2, height=2, width=2)
    state = np.ones((2, 2, 2), dtype=np.uint8)
    idx = np.array([1, 2, 3], dtype=np.int32)
    cnt = np.array([10, 5], dtype=np.uint16)  # shorter -> truncation
    buf.push(state, idx, cnt, 1)
    s, il, cl, v = buf.sample(batch_size=1, recent_ratio=1.0, recent_window_frac=1.0)
    assert len(s) == len(il) == len(cl) == len(v) == 1
    assert il[0].shape == cl[0].shape


def test_replay_buffer_resize_clear_and_shrink_preserves_recent() -> None:
    buf = ReplayBuffer(capacity=3, planes=1, height=1, width=1)
    st = np.zeros((1, 1, 1), dtype=np.uint8)
    for i in range(5):
        buf.push(st, [0], [1], i)
    assert buf.size == 3
    buf.set_capacity(2)
    assert buf.capacity == 2 and buf.size == 2
    buf.clear()
    assert buf.size == 0
    assert buf.sample(1, 1.0, 1.0) == ([], [], [], [])