"""Data augmentation utilities for encoded chess positions."""

from __future__ import annotations

from typing import ClassVar

import chesscore as ccore
import numpy as np

import config as C
import encoder

BOARD_SIZE = 8
NSQUARES = 64
PLANES_PER_POSITION = encoder.PLANES_PER_POSITION
HISTORY_LENGTH = encoder.HISTORY_LENGTH
POLICY_OUTPUT = int(getattr(ccore, "POLICY_SIZE", 73 * NSQUARES))

__all__ = ["Augment"]


NUM_DIRECTIONS = 8
DIR_MAX_DIST = 7
NUM_KNIGHT_DIRS = 8
KNIGHT_PLANES_BASE = 56
PROMO_CHOICES = 3
PROMO_STRIDE = 3
DIR_MAP_MIRROR = [2, 1, 0, 4, 3, 7, 6, 5]
KMAP_MIRROR = [1, 0, 3, 2, 5, 4, 7, 6]
DIR_MAP_ROT180 = [7, 6, 5, 4, 3, 2, 1, 0]
KMAP_ROT180 = [7, 6, 5, 4, 3, 2, 1, 0]
DIR_MAP_VFLIP_CS = [5, 6, 7, 3, 4, 0, 1, 2]
KMAP_VFLIP_CS = [6, 7, 4, 5, 2, 3, 0, 1]
PMAP_PROMOS = [0, 2, 1]


class Augment:
    """Applies symmetry-based data augmentation to batched positions."""

    _policy_map_cache: ClassVar[dict[str, np.ndarray]] = {}

    @staticmethod
    def _policy_index_permutation(transform: str) -> np.ndarray:
        if transform in Augment._policy_map_cache:
            return Augment._policy_map_cache[transform]
        assert (
            POLICY_OUTPUT % NSQUARES == 0
        ), "POLICY_OUTPUT must be divisible by NSQUARES"
        planes = POLICY_OUTPUT // NSQUARES
        required_planes = max(
            KNIGHT_PLANES_BASE + NUM_KNIGHT_DIRS,
            NSQUARES + PROMO_STRIDE * PROMO_CHOICES,
        )
        base = np.arange(POLICY_OUTPUT, dtype=np.int32).reshape(
            planes, BOARD_SIZE, BOARD_SIZE
        )
        out = base
        if transform == "mirror":
            arr = base[:, :, ::-1]
            out = arr.copy()
            if planes >= required_planes:
                dir_map = DIR_MAP_MIRROR
                for d in range(NUM_DIRECTIONS):
                    for dist in range(DIR_MAX_DIST):
                        out[dir_map[d] * DIR_MAX_DIST + dist] = arr[
                            d * DIR_MAX_DIST + dist
                        ]
                knight_map = KMAP_MIRROR
                for k in range(NUM_KNIGHT_DIRS):
                    out[KNIGHT_PLANES_BASE + knight_map[k]] = arr[
                        KNIGHT_PLANES_BASE + k
                    ]
                promo_map = PMAP_PROMOS
                for promo in range(PROMO_CHOICES):
                    b = NSQUARES + promo * PROMO_STRIDE
                    out[b + promo_map[0]] = arr[b + 0]
                    out[b + promo_map[1]] = arr[b + 1]
                    out[b + promo_map[2]] = arr[b + 2]
            else:
                Augment._policy_map_cache[transform] = np.arange(
                    POLICY_OUTPUT, dtype=np.int32
                )
                return Augment._policy_map_cache[transform]
        elif transform == "rot180":
            arr = base[:, ::-1, ::-1]
            out = arr.copy()
            if planes >= required_planes:
                dir_map = DIR_MAP_ROT180
                for d in range(NUM_DIRECTIONS):
                    for dist in range(DIR_MAX_DIST):
                        out[dir_map[d] * DIR_MAX_DIST + dist] = arr[
                            d * DIR_MAX_DIST + dist
                        ]
                knight_map = KMAP_ROT180
                for k in range(NUM_KNIGHT_DIRS):
                    out[KNIGHT_PLANES_BASE + knight_map[k]] = arr[
                        KNIGHT_PLANES_BASE + k
                    ]
                promo_map = PMAP_PROMOS
                for promo in range(PROMO_CHOICES):
                    b = NSQUARES + promo * PROMO_STRIDE
                    out[b + promo_map[0]] = arr[b + 0]
                    out[b + promo_map[1]] = arr[b + 1]
                    out[b + promo_map[2]] = arr[b + 2]
            else:
                Augment._policy_map_cache[transform] = np.arange(
                    POLICY_OUTPUT, dtype=np.int32
                )
                return Augment._policy_map_cache[transform]

        elif transform == "vflip_cs":
            arr = base[:, ::-1, :]
            out = arr.copy()
            if planes >= required_planes:
                dir_map = DIR_MAP_VFLIP_CS
                for d in range(NUM_DIRECTIONS):
                    for dist in range(DIR_MAX_DIST):
                        out[dir_map[d] * DIR_MAX_DIST + dist] = arr[
                            d * DIR_MAX_DIST + dist
                        ]
                knight_map = KMAP_VFLIP_CS
                for k in range(NUM_KNIGHT_DIRS):
                    out[KNIGHT_PLANES_BASE + knight_map[k]] = arr[
                        KNIGHT_PLANES_BASE + k
                    ]
            else:
                Augment._policy_map_cache[transform] = np.arange(
                    POLICY_OUTPUT, dtype=np.int32
                )
                return Augment._policy_map_cache[transform]
        Augment._policy_map_cache[transform] = out.reshape(-1)
        return Augment._policy_map_cache[transform]

    @staticmethod
    def _feature_plane_indices() -> dict[str, int]:
        turn_plane = HISTORY_LENGTH * PLANES_PER_POSITION
        fullmove_plane = turn_plane + 1
        castling_base = turn_plane + 2
        return {
            "planes_per_pos": PLANES_PER_POSITION,
            "hist_len": HISTORY_LENGTH,
            "turn_plane": turn_plane,
            "fullmove_plane": fullmove_plane,
            "castling_base": castling_base,
        }

    @staticmethod
    def _vflip_cs_plane_permutation(num_planes: int) -> np.ndarray:
        meta = Augment._feature_plane_indices()
        perm = np.arange(num_planes, dtype=np.int32)
        for t in range(meta["hist_len"]):
            base = t * meta["planes_per_pos"]
            for piece in range(6):
                plane_a = base + piece * 2 + 0
                plane_b = base + piece * 2 + 1
                perm[plane_a], perm[plane_b] = perm[plane_b], perm[plane_a]
        castling_plane_base = meta["castling_base"]
        if castling_plane_base + 3 < num_planes:
            perm[castling_plane_base + 0], perm[castling_plane_base + 2] = (
                perm[castling_plane_base + 2],
                perm[castling_plane_base + 0],
            )
            perm[castling_plane_base + 1], perm[castling_plane_base + 3] = (
                perm[castling_plane_base + 3],
                perm[castling_plane_base + 1],
            )
        return perm

    @staticmethod
    def apply(
        states: list[np.ndarray], policies: list[np.ndarray], transform: str
    ) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
        if not states:
            return states, policies, False
        state_batch = np.stack(states, axis=0)
        policy_batch = np.stack(policies, axis=0)
        stm_swapped = False
        if transform == "mirror":
            state_batch = state_batch[..., ::-1].copy()
            policy_batch = policy_batch[:, Augment._policy_index_permutation("mirror")]
        elif transform == "rot180":
            state_batch = state_batch[..., ::-1, ::-1].copy()
            policy_batch = policy_batch[:, Augment._policy_index_permutation("rot180")]
        elif transform == "vflip_cs":
            state_batch = state_batch[..., ::-1, :].copy()
            perm = Augment._vflip_cs_plane_permutation(state_batch.shape[1])
            state_batch = state_batch[:, perm]
            idx = Augment._feature_plane_indices()
            tp = idx["turn_plane"]
            if tp < state_batch.shape[1]:
                if np.issubdtype(state_batch.dtype, np.floating):
                    one_val = np.array(1.0, dtype=state_batch.dtype)
                else:
                    one_val = np.array(C.DATA.U8_SCALE, dtype=state_batch.dtype)
                state_batch[:, tp] = one_val - state_batch[:, tp]
            policy_batch = policy_batch[
                :, Augment._policy_index_permutation("vflip_cs")
            ]
            stm_swapped = True
        else:
            return states, policies, False
        out_states = [state_batch[i].copy() for i in range(state_batch.shape[0])]
        out_pols = [policy_batch[i].copy() for i in range(policy_batch.shape[0])]
        return out_states, out_pols, stm_swapped
