"""Symmetry-based data augmentation for state and policy tensors."""

from __future__ import annotations

from typing import ClassVar

import chesscore as ccore
import config as C
import encoder
import numpy as np

BOARD_SIZE = encoder.BOARD_SIZE
NSQUARES = BOARD_SIZE * BOARD_SIZE
PLANES_PER_POSITION = encoder.PLANES_PER_POSITION
HISTORY_LENGTH = encoder.HISTORY_LENGTH
POLICY_OUTPUT = int(getattr(ccore, "POLICY_SIZE", encoder.POLICY_SIZE))

NUM_DIR = 8
DIR_MAX = 7
NUM_KNIGHT = 8
KNIGHT_BASE = 56
PROMO_CHOICES = 3
PROMO_STRIDE = 3

DIR_MIRROR = [2, 1, 0, 4, 3, 7, 6, 5]
K_MIRROR = [1, 0, 3, 2, 5, 4, 7, 6]
DIR_ROT180 = [7, 6, 5, 4, 3, 2, 1, 0]
K_ROT180 = [7, 6, 5, 4, 3, 2, 1, 0]
DIR_VFLIP_CS = [5, 6, 7, 3, 4, 0, 1, 2]
K_VFLIP_CS = [6, 7, 4, 5, 2, 3, 0, 1]
PROMO_MAP = [0, 2, 1]

__all__ = ["Augment"]


class Augment:
    """Symmetry-based augmentation for encoded states and sparse policies."""

    _policy_map_cache: ClassVar[dict[str, np.ndarray]] = {}

    @staticmethod
    def apply(
        states: list[np.ndarray],
        policies: list[np.ndarray],
        transform: str,
    ) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
        """Return transformed (states, policies, stm_swapped) for the given symmetry."""
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
            turn_plane = Augment._feature_plane_indices()["turn_plane"]
            if 0 <= turn_plane < s_batch.shape[1]:
                one = np.array(
                    1.0 if np.issubdtype(s_batch.dtype, np.floating) else C.DATA.u8_scale,
                    dtype=s_batch.dtype,
                )
                s_batch[:, turn_plane] = one - s_batch[:, turn_plane]
            p_batch = p_batch[:, Augment._policy_index_permutation("vflip_cs")]
            stm_swapped = True
        else:
            return states, policies, False

        out_states = [s_batch[idx].copy() for idx in range(s_batch.shape[0])]
        out_policies = [p_batch[idx].copy() for idx in range(p_batch.shape[0])]
        return out_states, out_policies, stm_swapped

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _policy_index_permutation(transform: str) -> np.ndarray:
        """Return the cached permutation for the requested policy symmetry."""
        cached = Augment._policy_map_cache.get(transform)
        if cached is not None:
            return cached

        planes = POLICY_OUTPUT // NSQUARES
        required = max(KNIGHT_BASE + NUM_KNIGHT, NSQUARES + PROMO_STRIDE * PROMO_CHOICES)

        base = np.arange(POLICY_OUTPUT, dtype=np.int32).reshape(planes, BOARD_SIZE, BOARD_SIZE)
        arr = base
        out = arr.copy()

        def _dir_knight_promo(dir_map: list[int], k_map: list[int]) -> None:
            for direction in range(NUM_DIR):
                for dist in range(DIR_MAX):
                    out[dir_map[direction] * DIR_MAX + dist] = arr[direction * DIR_MAX + dist]
            for idx in range(NUM_KNIGHT):
                out[KNIGHT_BASE + k_map[idx]] = arr[KNIGHT_BASE + idx]
            for choice in range(PROMO_CHOICES):
                base_idx = NSQUARES + choice * PROMO_STRIDE
                out[base_idx + PROMO_MAP[0]] = arr[base_idx + 0]
                out[base_idx + PROMO_MAP[1]] = arr[base_idx + 1]
                out[base_idx + PROMO_MAP[2]] = arr[base_idx + 2]

        if transform == "mirror":
            arr = base[:, :, ::-1]
            out = arr.copy()
            if planes >= required:
                _dir_knight_promo(DIR_MIRROR, K_MIRROR)
            else:
                perm = np.arange(POLICY_OUTPUT, dtype=np.int32)
                Augment._policy_map_cache[transform] = perm
                return perm
        elif transform == "rot180":
            arr = base[:, ::-1, ::-1]
            out = arr.copy()
            if planes >= required:
                _dir_knight_promo(DIR_ROT180, K_ROT180)
            else:
                perm = np.arange(POLICY_OUTPUT, dtype=np.int32)
                Augment._policy_map_cache[transform] = perm
                return perm
        elif transform == "vflip_cs":
            arr = base[:, ::-1, :]
            out = arr.copy()
            if planes >= required:
                for direction in range(NUM_DIR):
                    for dist in range(DIR_MAX):
                        out[DIR_VFLIP_CS[direction] * DIR_MAX + dist] = arr[direction * DIR_MAX + dist]
                for idx in range(NUM_KNIGHT):
                    out[KNIGHT_BASE + K_VFLIP_CS[idx]] = arr[KNIGHT_BASE + idx]
            else:
                perm = np.arange(POLICY_OUTPUT, dtype=np.int32)
                Augment._policy_map_cache[transform] = perm
                return perm
        else:
            perm = np.arange(POLICY_OUTPUT, dtype=np.int32)
            Augment._policy_map_cache[transform] = perm
            return perm

        perm = out.reshape(-1).astype(np.int64, copy=True)
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

        for history_idx in range(meta["hist_len"]):
            base = history_idx * meta["planes_per_pos"]
            for piece in range(6):
                a = base + piece * 2
                b = a + 1
                perm[a], perm[b] = perm[b], perm[a]

        castling_base = meta["castling_base"]
        if castling_base + 3 < num_planes:
            perm[castling_base + 0], perm[castling_base + 2] = perm[castling_base + 2], perm[castling_base + 0]
            perm[castling_base + 1], perm[castling_base + 3] = perm[castling_base + 3], perm[castling_base + 1]
        return perm
