from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("chesscore", reason="hiányzik a chesscore kiterjesztés")

import chesscore as ccore
import encoder


def test_encoder_piece_placement(ensure_chesscore):
    pos = ccore.Position()
    pos.from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")

    enc = encoder.encode_position(pos)

    assert enc[0, 1, 4] == 1.0, "A fehér gyalog (e2) hiányzik a 0. síkról"
    assert enc[0].sum() == 1.0, "A 0. síkon pontosan egy gyalognak kell lennie"

    assert enc[11, 7, 4] == 1.0, "A fekete király (e8) hiányzik a 11. síkról"


def test_encoder_auxiliary_planes(ensure_chesscore):
    pos = ccore.Position()
    pos.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w k - 0 1")

    enc = encoder.encode_position(pos)

    aux_base = encoder.INPUT_PLANES - 7

    assert enc[aux_base, 0, 0] == 1.0, "A lépésen lévő oldal síkja hibás (fehérre)"

    assert enc[aux_base + 2, 0, 0] == 0.0
    assert enc[aux_base + 3, 0, 0] == 0.0
    assert enc[aux_base + 4, 0, 0] == 1.0
    assert enc[aux_base + 5, 0, 0] == 0.0


def test_encoder_history_planes(ensure_chesscore):
    pos = ccore.Position()
    history = []

    pos.from_fen("rnbqkbnr/pppppppp/8/8/8/8/4P3/RNBQKBNR w KQkq - 0 1")
    h1 = ccore.Position(pos)
    history.append(h1)

    pos.make_move(list(pos.legal_moves())[0])

    enc = encoder.encode_position(pos, history)

    current_planes = enc[0:14]
    history_planes = enc[14:28]

    assert current_planes.sum() > 0
    assert history_planes.sum() > 0
    assert not np.array_equal(current_planes, history_planes)


def test_encode_move_index_correctness(ensure_chesscore):
    pos = ccore.Position()
    pos.from_fen("8/8/8/3R4/8/8/8/8 w - - 0 1")

    moves = pos.legal_moves()
    for m in moves:
        idx = encoder.encode_move_index(m)
        assert 0 <= idx < encoder.POLICY_SIZE

    pass
