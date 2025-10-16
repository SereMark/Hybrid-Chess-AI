from __future__ import annotations

import math
from pathlib import Path

import chesscore as ccore
import numpy as np
import pytest
from arena import (
    DEFAULT_START_FEN,
    ArenaResult,
    _prepare_pgn_directory,
    _result_to_str,
    _save_pgn,
    _ScoreTracker,
    _select_move,
    _square_file,
    _square_name,
    _square_rank,
)
from numpy.random import default_rng


def test_select_move_prefers_argmax_when_temperature_low() -> None:
    counts = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    rng = default_rng(3)
    assert _select_move(counts, temperature=0.0, rng=rng) == 1

    empty_choice = _select_move(np.array([], dtype=np.float32), temperature=1.0, rng=rng)
    assert empty_choice == 0


def test_select_move_sampling_uses_probabilities() -> None:
    counts = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rng_impl = default_rng(42)
    rng_expected = default_rng(42)
    index = _select_move(counts, temperature=1.0, rng=rng_impl)

    scaled = np.maximum(counts, 0.0) ** 1.0
    probabilities = scaled / scaled.sum()
    expected_index = rng_expected.choice(len(probabilities), p=probabilities)

    assert index == expected_index
    assert float(probabilities.sum()) == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_square_helpers_round_trip() -> None:
    square = 27  # d4
    assert _square_file(square) == 3
    assert _square_rank(square) == 3
    assert _square_name(square) == "d4"

    last_square = 63
    assert _square_name(last_square) == "h8"


def test_score_tracker_summarises_results() -> None:
    tracker = _ScoreTracker(total_games=4)
    tracker.record(True, ccore.WHITE_WIN)  # candidate win
    tracker.record(True, ccore.DRAW)  # draw
    tracker.record(False, ccore.BLACK_WIN)  # candidate win as black
    tracker.record(False, ccore.WHITE_WIN)  # baseline win

    result = tracker.to_result(12.5, notes=["ok"])
    assert isinstance(result, ArenaResult)
    assert result.games == 4
    assert result.candidate_wins == 2
    assert result.baseline_wins == 1
    assert result.draws == 1
    assert math.isclose(result.score_pct, 62.5)
    assert math.isclose(result.draw_pct, 25.0)
    assert math.isclose(result.decisive_pct, 75.0)
    assert result.notes == ["ok"]
    assert math.isclose(result.elapsed_s, 12.5)


def test_prepare_pgn_directory_and_save(tmp_path: Path) -> None:
    directory = _prepare_pgn_directory(tmp_path / "pgn")
    assert directory is not None
    assert directory.exists()

    path = _save_pgn(
        directory,
        label="candidate",
        game_index=0,
        start_fen=DEFAULT_START_FEN,
        moves_san=["e4", "e5"],
        result_str=_result_to_str(ccore.WHITE_WIN),
        white_name="Candidate",
        black_name="Baseline",
    )
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert '[Event "Arena Evaluation"]' in text
    assert "1. e4 e5 1-0" in text


def test_result_to_str_handled_outcomes() -> None:
    assert _result_to_str(ccore.WHITE_WIN) == "1-0"
    assert _result_to_str(ccore.BLACK_WIN) == "0-1"
    assert _result_to_str(ccore.DRAW) == "1/2-1/2"
