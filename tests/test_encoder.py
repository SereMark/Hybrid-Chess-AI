from __future__ import annotations

import numpy as np
import encoder


def test_encode_position_and_batch(ensure_chesscore) -> None:
    pos = ensure_chesscore.Position()
    enc = encoder.encode_position(pos)
    assert enc.shape == (encoder.INPUT_PLANES, encoder.BOARD_SIZE, encoder.BOARD_SIZE)
    assert enc.dtype == np.float32 and np.isfinite(enc).all()

    batch = encoder.encode_batch([pos, pos])
    assert batch.shape[0] == 2
    assert batch.shape[1:] == enc.shape


def test_encode_move_index_valid(ensure_chesscore) -> None:
    pos = ensure_chesscore.Position()
    moves = list(pos.legal_moves())
    idx = [encoder.encode_move_index(m) for m in moves]
    assert idx and all(0 <= i < encoder.POLICY_SIZE for i in idx)