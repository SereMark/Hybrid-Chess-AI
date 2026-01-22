from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("chesscore", reason="missing chesscore extension")

import encoder
import self_play
from inference import BatchedEvaluator
from self_play import SelfPlayEngine


def test_selfplay_generates_games_and_samples(ensure_chesscore) -> None:
    ev = BatchedEvaluator(torch.device("cpu"))
    try:
        eng = SelfPlayEngine(ev)
        eng.set_num_workers(1)
        stats = eng.play_games(2)
        assert stats["games"] >= 1
        empty = eng.sample_batch(8, 1.0, 1.0)
        assert isinstance(empty, tuple) and len(empty) == 4
    finally:
        ev.close()


@pytest.mark.usefixtures("ensure_chesscore")
def test_self_play_opening_book_bias(tmp_path) -> None:
    ev = BatchedEvaluator(torch.device("cpu"))
    try:
        eng = SelfPlayEngine(ev)
        eng._opening_book = [("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1.0)]
        eng._opening_cumulative = np.array([1.0])
        eng._curriculum_prob = 0.0
        fens = [eng.sample_start_fen(np.random.default_rng(i)) for i in range(6)]
        assert all(len(f.split()) >= 6 for f in fens)
    finally:
        ev.close()


@pytest.mark.usefixtures("ensure_chesscore")
def test_self_play_resignation_and_adjudication(monkeypatch) -> None:
    ev = BatchedEvaluator(torch.device("cpu"))
    try:
        eng = SelfPlayEngine(ev)
        eng.set_num_workers(1)
        eng.resign_enabled = True
        eng.resign_threshold = 0.4
        eng.resign_min_plies = 0
        monkeypatch.setattr(self_play, "_material_balance", lambda pos: 5.0)
        stats = eng.play_games(4)
        assert stats["games"] > 0 and stats["term_adjudicated"] >= 0
    finally:
        ev.close()


@pytest.mark.usefixtures("ensure_chesscore")
def test_selfplay_play_games_populates_buffer_and_stats(monkeypatch, ensure_chesscore) -> None:
    ev = BatchedEvaluator(torch.device("cpu"))
    example_state = np.zeros((encoder.INPUT_PLANES, encoder.BOARD_SIZE, encoder.BOARD_SIZE), dtype=np.uint8)
    idx = np.array([0, 1], dtype=np.int32)
    cnt = np.array([10, 6], dtype=np.uint16)

    def fake_play_single_game(self, seed: int):
        payload = [
            (example_state.copy(), idx, cnt, True),
            (example_state.copy(), idx, cnt, False),
        ]
        self._store_examples(payload, ensure_chesscore.WHITE_WIN)
        return ensure_chesscore.WHITE_WIN, 5, "natural", 20.0

    monkeypatch.setattr(SelfPlayEngine, "_play_single_game", fake_play_single_game)

    try:
        eng = SelfPlayEngine(ev)
        eng.set_num_workers(2)
        stats = eng.play_games(3)
        assert stats["games"] == 3
        assert stats["white_wins"] == 3
        assert stats["term_natural"] == 3
        assert pytest.approx(stats["visit_per_move"], rel=1e-6) == 4.0
        assert eng.size() == 6
    finally:
        try:
            eng.close()
        finally:
            ev.close()


class _StubMCTS:
    def set_simulations(self, sims: int) -> None:
        self._sims = sims

    def search_batched_legal(self, position, infer_fn, batch_cap):
        moves = list(position.legal_moves())
        _ = infer_fn([position], [moves])
        return np.ones(len(moves) + 1, dtype=np.float32)

    def advance_root(self, position, move) -> None:
        return None


@pytest.mark.usefixtures("ensure_chesscore")
def test_selfplay_handles_exhausted_games(monkeypatch) -> None:
    ev = BatchedEvaluator(torch.device("cpu"))
    monkeypatch.setattr(SelfPlayEngine, "_build_mcts", lambda self, rng: _StubMCTS())
    try:
        eng = SelfPlayEngine(ev)
        eng.set_num_workers(1)
        stats = eng.play_games(1)
        assert stats["term_exhausted"] >= 1
        assert stats["draws"] >= 1
    finally:
        try:
            eng.close()
        finally:
            ev.close()


@pytest.mark.usefixtures("ensure_chesscore")
def test_selfplay_state_dict_roundtrip(monkeypatch, ensure_chesscore) -> None:
    ev = BatchedEvaluator(torch.device("cpu"))
    eng2: SelfPlayEngine | None = None

    example_state = np.zeros((encoder.INPUT_PLANES, encoder.BOARD_SIZE, encoder.BOARD_SIZE), dtype=np.uint8)
    idx = np.array([2, 3], dtype=np.int32)
    cnt = np.array([5, 9], dtype=np.uint16)

    def fake_game(self, seed: int):
        data = [
            (example_state.copy(), idx, cnt, True),
            (example_state.copy(), idx, cnt, False),
        ]
        self._store_examples(data, ensure_chesscore.BLACK_WIN)
        return ensure_chesscore.BLACK_WIN, 7, "resign", 18.0

    monkeypatch.setattr(SelfPlayEngine, "_play_single_game", fake_game)

    try:
        eng = SelfPlayEngine(ev)
        eng.set_num_workers(1)
        _ = eng.play_games(3)
        snap = eng.state_dict()

        eng2 = SelfPlayEngine(ev)
        eng2.load_state_dict(snap)

        assert eng2.size() == eng.size() == 6
        assert eng2.adjudication_phase == eng.adjudication_phase

        batch1 = eng.sample_batch(2, 1.0, 1.0)
        batch2 = eng2.sample_batch(2, 1.0, 1.0)
        assert len(batch1[0]) == len(batch2[0]) == 2
        assert np.array_equal(batch1[1][0], batch2[1][0])
    finally:
        if eng2 is not None:
            eng2.close()
        eng.close()
        ev.close()
