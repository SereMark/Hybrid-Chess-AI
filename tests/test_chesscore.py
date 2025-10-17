from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("chesscore", reason="chesscore extension missing")

import chesscore as ccore


def _move_by_uci(position: "ccore.Position", uci: str) -> "ccore.Move":
    for m in position.legal_moves():
        if ccore.uci_of_move(m) == uci:
            return m
    raise ValueError(uci)


@pytest.mark.usefixtures("ensure_chesscore")
def test_initial_position_legal_and_encodable() -> None:
    pos = ccore.Position()
    moves = pos.legal_moves()
    assert len(moves) == 20 and pos.result() == ccore.ONGOING
    for m in moves:
        enc = ccore.encode_move_index(m)
        assert 0 <= enc < ccore.POLICY_SIZE


@pytest.mark.usefixtures("ensure_chesscore")
def test_make_moves_and_detect_repetition() -> None:
    pos = ccore.Position()
    seq = ["e2e4", "e7e5", "g1f3", "b8c6", "f3g1", "c6b8", "g1f3", "b8c6", "f3g1", "c6b8"]
    for u in seq:
        result = pos.make_move(_move_by_uci(pos, u))
        assert result in {ccore.ONGOING, ccore.DRAW}
    assert pos.count_repetitions() >= 1


@pytest.mark.usefixtures("ensure_chesscore")
def test_mcts_search_returns_visit_counts() -> None:
    pos = ccore.Position()
    mcts = ccore.MCTS(inc := 64, 1.0, 0.3, 0.25)
    mcts.seed(42)

    def _uniform(positions, moves_lists):
        pol = [np.full(len(m), 1.0 / max(1, len(m)), dtype=np.float32) for m in moves_lists]
        val = np.zeros(len(positions), dtype=np.float32)
        return pol, val

    visits = mcts.search_batched_legal(pos, _uniform, 32)
    assert len(visits) == len(pos.legal_moves())
    assert sum(visits) > 0 and max(visits) <= inc
