from __future__ import annotations

import numpy as np
import pytest

try:
    import chesscore as ccore  # type: ignore
except ImportError:  # pragma: no cover - handled by fixture skip
    ccore = None


def _move_by_uci(position: "ccore.Position", uci: str) -> "ccore.Move":
    for move in position.legal_moves():
        if ccore.uci_of_move(move) == uci:
            return move
    raise ValueError(f"Move {uci} not legal in position {position.to_fen()}")


@pytest.mark.usefixtures("ensure_chesscore")
def test_initial_position_has_expected_legal_moves() -> None:
    position = ccore.Position()
    moves = position.legal_moves()
    assert len(moves) == 20
    assert position.result() == ccore.ONGOING
    for move in moves:
        uci = ccore.uci_of_move(move)
        assert len(uci) in {4, 5}
        encoded = ccore.encode_move_index(move)
        assert 0 <= encoded < ccore.POLICY_SIZE


@pytest.mark.usefixtures("ensure_chesscore")
def test_position_make_moves_and_repetition() -> None:
    position = ccore.Position()
    sequence = ["e2e4", "e7e5", "g1f3", "b8c6", "f3g1", "c6b8", "g1f3", "b8c6", "f3g1", "c6b8"]
    for uci in sequence:
        move = _move_by_uci(position, uci)
        result = position.make_move(move)
        assert result in {ccore.ONGOING, ccore.DRAW}
    assert position.count_repetitions() >= 2


@pytest.mark.usefixtures("ensure_chesscore")
def test_encode_move_indices_batch_shape() -> None:
    position = ccore.Position()
    moves = position.legal_moves()
    encoded_list = ccore.encode_move_indices_batch([moves, moves])
    assert len(encoded_list) == 2
    for arr in encoded_list:
        np_arr = np.asarray(arr, dtype=np.int32)
        assert np_arr.ndim == 1
        assert np_arr.size == len(moves)
        assert np.all(np_arr >= -1)


def _uniform_evaluator(positions, moves_lists):
    policies = []
    for moves in moves_lists:
        count = len(moves)
        if count == 0:
            policies.append(np.zeros(0, dtype=np.float32))
            continue
        policies.append(np.full(count, 1.0 / count, dtype=np.float32))
    values = np.zeros(len(positions), dtype=np.float32)
    return policies, values


@pytest.mark.usefixtures("ensure_chesscore")
def test_mcts_search_returns_visit_counts() -> None:
    position = ccore.Position()
    engine = ccore.MCTS(inc_sims := 64, 1.0, 0.3, 0.25)
    engine.seed(42)
    visits = engine.search_batched_legal(position, _uniform_evaluator, 32)
    moves = position.legal_moves()
    assert len(visits) == len(moves)
    assert sum(visits) > 0
    assert max(visits) <= inc_sims
