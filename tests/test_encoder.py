from __future__ import annotations

import encoder
import numpy as np


def test_encode_position_roundtrip(ensure_chesscore) -> None:  # noqa: ARG001
    position = ensure_chesscore.Position()
    encoded = encoder.encode_position(position)
    assert encoded.shape == (encoder.INPUT_PLANES, encoder.BOARD_SIZE, encoder.BOARD_SIZE)
    assert encoded.dtype == np.float32
    assert np.isfinite(encoded).all()


def test_encode_move_index_valid(ensure_chesscore) -> None:  # noqa: ARG001
    board = ensure_chesscore.Position()
    moves = list(board.legal_moves())
    indices = [encoder.encode_move_index(move) for move in moves]
    assert indices
    assert all(0 <= idx < encoder.POLICY_SIZE for idx in indices)
