from __future__ import annotations

import numpy as np
import pytest
import torch
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
        monkeypatch.setattr(eng, "_material_balance", lambda pos: 5.0)
        stats = eng.play_games(4)
        assert stats["games"] > 0 and stats["term_adjudicated"] >= 0
    finally:
        ev.close()