from __future__ import annotations

import encoder
import numpy as np
import pytest
from augmentation import POLICY_OUTPUT, Augment


def test_policy_index_permutation_shapes() -> None:
    perm = Augment._policy_index_permutation("mirror")
    assert perm.ndim == 1 and perm.size == POLICY_OUTPUT


def test_apply_shapes_and_roundtrip() -> None:
    planes = encoder.INPUT_PLANES
    rng = np.random.default_rng(0)
    state = rng.random((planes, 8, 8), dtype=np.float32)
    policy = rng.random(POLICY_OUTPUT, dtype=np.float32)
    for t in ("mirror", "rot180", "vflip_cs"):
        s1, p1, _ = Augment.apply([state], [policy], t)
        s2, p2, _ = Augment.apply(s1, p1, t)
        assert s1[0].shape == state.shape and p1[0].shape == policy.shape
        np.testing.assert_allclose(state, s2[0])
        np.testing.assert_allclose(policy, p2[0])


@pytest.mark.usefixtures("ensure_chesscore")
def test_vflip_turn_plane_toggle() -> None:
    enc = encoder.encode_position(encoder.ccore.Position()).astype(np.float32)
    pol = np.linspace(0.0, 1.0, POLICY_OUTPUT, dtype=np.float32)
    s, p, swapped = Augment.apply([enc], [pol], "vflip_cs")
    assert swapped is True
    tp = Augment._feature_plane_indices()["turn_plane"]
    assert np.isclose(s[0][tp].mean(), 0.0)
    s2, p2, swapped2 = Augment.apply(s, p, "vflip_cs")
    assert swapped2 is True
    np.testing.assert_allclose(enc, s2[0])
    np.testing.assert_allclose(pol, p2[0])


def test_policy_permutation_is_bijective() -> None:
    base = np.arange(POLICY_OUTPUT, dtype=np.int32)
    for t in ("mirror", "rot180", "vflip_cs"):
        perm = Augment._policy_index_permutation(t)
        assert np.array_equal(np.sort(perm), base)
