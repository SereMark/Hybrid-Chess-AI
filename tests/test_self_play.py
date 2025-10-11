from __future__ import annotations

import numpy as np
import pytest
import torch
from inference import BatchedEvaluator
from self_play import SelfPlayEngine


def test_selfplay_generates_games(ensure_chesscore) -> None:  # noqa: ARG001
    evaluator = BatchedEvaluator(torch.device("cpu"))
    try:
        engine = SelfPlayEngine(evaluator)
        engine.set_num_workers(1)
        stats = engine.play_games(2)
        assert stats["games"] >= 1
    finally:
        evaluator.close()


@pytest.mark.usefixtures("ensure_chesscore")
def test_self_play_opening_book_usage(tmp_path) -> None:
    book_path = tmp_path / "book.json"
    book_path.write_text(
        '[{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "weight": 1.0}]', encoding="utf-8"
    )
    evaluator = BatchedEvaluator(torch.device("cpu"))
    try:
        engine = SelfPlayEngine(evaluator)
        engine._opening_book = [("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1.0)]
        engine._opening_cumulative = np.array([1.0])
        engine._curriculum_prob = 0.0
        samples = [engine.sample_start_fen(np.random.default_rng(i)) for i in range(8)]
        assert all("w" in fen.split(" ")[1] or "b" in fen.split(" ")[1] for fen in samples)
    finally:
        evaluator.close()


@pytest.mark.usefixtures("ensure_chesscore")
def test_self_play_resignation_and_adjudication(monkeypatch) -> None:
    evaluator = BatchedEvaluator(torch.device("cpu"))
    try:
        engine = SelfPlayEngine(evaluator)
        engine.set_num_workers(1)
        engine.resign_enabled = True
        engine.resign_threshold = 0.4
        engine.resign_min_plies = 0
        monkeypatch.setattr(engine, "_material_balance", lambda position: 5.0)
        stats = engine.play_games(4)
        assert stats["games"] > 0
        assert stats["term_adjudicated"] >= 0
    finally:
        evaluator.close()
