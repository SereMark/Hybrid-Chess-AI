from __future__ import annotations

from typing import ClassVar

import numpy as np
import chesscore as ccore
import config as C
import encoder

BOARD_SIZE = 8
NSQUARES = 64
PLANES_PER_POSITION = encoder.PLANES_PER_POSITION
HISTORY_LENGTH = encoder.HISTORY_LENGTH
POLICY_OUTPUT = int(getattr(ccore, "POLICY_SIZE", 73 * NSQUARES))

# Directional plane layouts
NUM_DIR = 8
DIR_MAX = 7
NUM_KNIGHT = 8
KNIGHT_BASE = 56
PROMO_CHOICES = 3
PROMO_STRIDE = 3

# Plane remaps for symmetries
DIR_MIRROR = [2, 1, 0, 4, 3, 7, 6, 5]
K_MIRROR = [1, 0, 3, 2, 5, 4, 7, 6]
DIR_ROT180 = [7, 6, 5, 4, 3, 2, 1, 0]
K_ROT180 = [7, 6, 5, 4, 3, 2, 1, 0]
DIR_VFLIP_CS = [5, 6, 7, 3, 4, 0, 1, 2]
K_VFLIP_CS = [6, 7, 4, 5, 2, 3, 0, 1]
PROMO_MAP = [0, 2, 1]


class Augment:
    """Symmetry-based augmentation for states and policy indices."""

    _policy_map_cache: ClassVar[dict[str, np.ndarray]] = {}

    @staticmethod
    def _policy_index_permutation(transform: str) -> np.ndarray:
        """Return a flat permutation of POLICY_OUTPUT indices for transform."""
        if transform in Augment._policy_map_cache:
            return Augment._policy_map_cache[transform]

        assert POLICY_OUTPUT % NSQUARES == 0
        planes = POLICY_OUTPUT // NSQUARES
        required = max(KNIGHT_BASE + NUM_KNIGHT, NSQUARES + PROMO_STRIDE * PROMO_CHOICES)

        base = np.arange(POLICY_OUTPUT, dtype=np.int32).reshape(planes, BOARD_SIZE, BOARD_SIZE)
        arr = base
        out = arr.copy()

        def _dir_knight_promo(dir_map: list[int], k_map: list[int]) -> None:
            for d in range(NUM_DIR):
                for dist in range(DIR_MAX):
                    out[dir_map[d] * DIR_MAX + dist] = arr[d * DIR_MAX + dist]
            for k in range(NUM_KNIGHT):
                out[KNIGHT_BASE + k_map[k]] = arr[KNIGHT_BASE + k]
            for p in range(PROMO_CHOICES):
                b = NSQUARES + p * PROMO_STRIDE
                out[b + PROMO_MAP[0]] = arr[b + 0]
                out[b + PROMO_MAP[1]] = arr[b + 1]
                out[b + PROMO_MAP[2]] = arr[b + 2]

        if transform == "mirror":
            arr = base[:, :, ::-1]
            out = arr.copy()
            if planes >= required:
                _dir_knight_promo(DIR_MIRROR, K_MIRROR)
            else:
                Augment._policy_map_cache[transform] = np.arange(POLICY_OUTPUT, dtype=np.int32)
                return Augment._policy_map_cache[transform]
        elif transform == "rot180":
            arr = base[:, ::-1, ::-1]
            out = arr.copy()
            if planes >= required:
                _dir_knight_promo(DIR_ROT180, K_ROT180)
            else:
                Augment._policy_map_cache[transform] = np.arange(POLICY_OUTPUT, dtype=np.int32)
                return Augment._policy_map_cache[transform]
        elif transform == "vflip_cs":
            arr = base[:, ::-1, :]
            out = arr.copy()
            if planes >= required:
                for d in range(NUM_DIR):
                    for dist in range(DIR_MAX):
                        out[DIR_VFLIP_CS[d] * DIR_MAX + dist] = arr[d * DIR_MAX + dist]
                for k in range(NUM_KNIGHT):
                    out[KNIGHT_BASE + K_VFLIP_CS[k]] = arr[KNIGHT_BASE + k]
            else:
                Augment._policy_map_cache[transform] = np.arange(POLICY_OUTPUT, dtype=np.int32)
                return Augment._policy_map_cache[transform]
        else:
            Augment._policy_map_cache[transform] = np.arange(POLICY_OUTPUT, dtype=np.int32)
            return Augment._policy_map_cache[transform]

        perm = out.reshape(-1)
        Augment._policy_map_cache[transform] = perm
        return perm

    @staticmethod
    def _feature_plane_indices() -> dict[str, int]:
        turn_plane = HISTORY_LENGTH * PLANES_PER_POSITION
        return {
            "planes_per_pos": PLANES_PER_POSITION,
            "hist_len": HISTORY_LENGTH,
            "turn_plane": turn_plane,
            "fullmove_plane": turn_plane + 1,
            "castling_base": turn_plane + 2,
        }

    @staticmethod
    def _vflip_cs_plane_permutation(num_planes: int) -> np.ndarray:
        meta = Augment._feature_plane_indices()
        perm = np.arange(num_planes, dtype=np.int32)

        for t in range(meta["hist_len"]):
            base = t * meta["planes_per_pos"]
            for piece in range(6):
                a = base + piece * 2 + 0
                b = base + piece * 2 + 1
                perm[a], perm[b] = perm[b], perm[a]

        cb = meta["castling_base"]
        if cb + 3 < num_planes:
            perm[cb + 0], perm[cb + 2] = perm[cb + 2], perm[cb + 0]
            perm[cb + 1], perm[cb + 3] = perm[cb + 3], perm[cb + 1]
        return perm

    @staticmethod
    def apply(
        states: list[np.ndarray], policies: list[np.ndarray], transform: str
    ) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
        """Apply transform to batches. Returns (states, policies, stm_swapped)."""
        if not states:
            return states, policies, False

        s_batch = np.stack(states, axis=0)
        p_batch = np.stack(policies, axis=0)
        stm_swapped = False

        if transform == "mirror":
            s_batch = s_batch[..., ::-1].copy()
            p_batch = p_batch[:, Augment._policy_index_permutation("mirror")]
        elif transform == "rot180":
            s_batch = s_batch[..., ::-1, ::-1].copy()
            p_batch = p_batch[:, Augment._policy_index_permutation("rot180")]
        elif transform == "vflip_cs":
            s_batch = s_batch[..., ::-1, :].copy()
            perm = Augment._vflip_cs_plane_permutation(s_batch.shape[1])
            s_batch = s_batch[:, perm]
            tp = Augment._feature_plane_indices()["turn_plane"]
            if tp < s_batch.shape[1]:
                one = np.array(1.0 if np.issubdtype(s_batch.dtype, np.floating) else C.DATA.u8_scale, dtype=s_batch.dtype)
                s_batch[:, tp] = one - s_batch[:, tp]
            p_batch = p_batch[:, Augment._policy_index_permutation("vflip_cs")]
            stm_swapped = True
        else:
            return states, policies, False

        out_s = [s_batch[i].copy() for i in range(s_batch.shape[0])]
        out_p = [p_batch[i].copy() for i in range(p_batch.shape[0])]
        return out_s, out_p, stm_swapped