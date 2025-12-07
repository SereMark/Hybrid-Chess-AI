from __future__ import annotations

import numpy as np
from replay_buffer import ReplayBuffer


def test_replay_buffer_push_sample_and_truncation() -> None:
    buf = ReplayBuffer(capacity=4, planes=2, height=2, width=2)
    state = np.ones((2, 2, 2), dtype=np.uint8)
    idx = np.array([1, 2, 3], dtype=np.int32)
    cnt = np.array([10, 5], dtype=np.uint16)
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


def test_replay_buffer_recent_sampling_bias() -> None:
    buf = ReplayBuffer(capacity=4, planes=1, height=1, width=1, seed=123)
    for i in range(4):
        state = np.full((1, 1, 1), i, dtype=np.uint8)
        buf.push(state, [i], [1], i)

    states_recent, *_ = buf.sample(batch_size=4, recent_ratio=1.0, recent_window_frac=0.5)
    assert len(states_recent) > 0
    recent_set = {int(s[0, 0, 0]) for s in states_recent}
    assert recent_set.issubset({2, 3})

    states_stale, *_ = buf.sample(batch_size=4, recent_ratio=0.0, recent_window_frac=0.5)
    assert len(states_stale) > 0
    stale_set = {int(s[0, 0, 0]) for s in states_stale}
    assert stale_set.issubset({0, 1})


def test_replay_buffer_state_dict_roundtrip_preserves_rng() -> None:
    buf = ReplayBuffer(capacity=3, planes=1, height=1, width=1, seed=999)
    for i in range(3):
        state = np.full((1, 1, 1), i, dtype=np.uint8)
        buf.push(state, [i], [1], i)

    _ = buf.sample(batch_size=2, recent_ratio=0.5, recent_window_frac=1.0)
    snapshot = buf.state_dict()

    restored = ReplayBuffer(capacity=1, planes=1, height=1, width=1)
    restored.load_state_dict(snapshot)

    states_a, idx_a, cnt_a, val_a = buf.sample(3, 0.5, 1.0)
    states_b, idx_b, cnt_b, val_b = restored.sample(3, 0.5, 1.0)

    assert len(states_a) == len(states_b) == 3
    for sa, sb in zip(states_a, states_b, strict=False):
        assert np.array_equal(sa, sb)
    for ia, ib in zip(idx_a, idx_b, strict=False):
        assert np.array_equal(ia, ib)
    for ca, cb in zip(cnt_a, cnt_b, strict=False):
        assert np.array_equal(ca, cb)
    assert np.array_equal(np.asarray(val_a, dtype=np.int8), np.asarray(val_b, dtype=np.int8))
