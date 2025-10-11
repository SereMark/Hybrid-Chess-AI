from __future__ import annotations

import chesscore as ccore  # type: ignore
import encoder
import numpy as np
import pytest
from augmentation import POLICY_OUTPUT, Augment


def test_policy_index_permutation_shapes() -> None:
    perm = Augment._policy_index_permutation("mirror")
    assert perm.ndim == 1
    assert perm.size == POLICY_OUTPUT


def test_apply_returns_expected_shapes() -> None:
    states = [np.ones((POLICY_OUTPUT // 8, 8, 8), dtype=np.float32)]
    policies = [np.ones((POLICY_OUTPUT,), dtype=np.float32)]
    out_states, out_policies, _ = Augment.apply(states, policies, "mirror")
    assert out_states[0].shape == states[0].shape
    assert out_policies[0].shape == policies[0].shape


def test_augmentation_roundtrip_state_and_policy() -> None:
    rng = np.random.default_rng(0)
    planes = POLICY_OUTPUT // 64
    state = rng.random((planes, 8, 8), dtype=np.float32)
    policy = rng.random(POLICY_OUTPUT, dtype=np.float32)
    mirror_state, mirror_policy, _ = Augment.apply([state], [policy], "mirror")
    restored_state, restored_policy, _ = Augment.apply(mirror_state, mirror_policy, "mirror")
    np.testing.assert_allclose(state, restored_state[0])
    np.testing.assert_allclose(policy, restored_policy[0])


@pytest.mark.usefixtures("ensure_chesscore")
def test_vflip_turn_swap_and_roundtrip() -> None:
    position = ccore.Position()
    encoded = encoder.encode_position(position).astype(np.float32)
    policy = np.linspace(0.0, 1.0, POLICY_OUTPUT, dtype=np.float32)

    vflip_state, vflip_policy, swapped = Augment.apply([encoded], [policy], "vflip_cs")
    assert swapped is True
    turn_plane = Augment._feature_plane_indices()["turn_plane"]
    assert np.isclose(vflip_state[0][turn_plane].mean(), 0.0)

    restored_state, restored_policy, swapped_back = Augment.apply(vflip_state, vflip_policy, "vflip_cs")
    assert swapped_back is True
    np.testing.assert_allclose(encoded, restored_state[0])
    np.testing.assert_allclose(policy, restored_policy[0])


def test_policy_permutation_is_bijective() -> None:
    base = np.arange(POLICY_OUTPUT, dtype=np.int32)
    for transform in ("mirror", "rot180", "vflip_cs"):
        perm = Augment._policy_index_permutation(transform)
        assert np.array_equal(np.sort(perm), base)
