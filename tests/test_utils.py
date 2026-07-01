from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.random import default_rng
from optimization import EMA, WarmupCosine, build_optimizer
from utils import MetricsReporter, flip_fen_perspective, sanitize_fen, select_visit_count_move

try:
    import chesscore as ccore
    from arena import (
        DEFAULT_START_FEN,
        ArenaResult,
        _prepare_pgn_directory,
        _result_to_str,
        _save_pgn,
        _ScoreTracker,
        _square_file,
        _square_name,
        _square_rank,
    )

    _ARENA_AVAILABLE = True
except ImportError:
    _ARENA_AVAILABLE = False


def test_metrics_reporter_writes_csv_and_jsonl(tmp_path: Path) -> None:
    csv_path = tmp_path / "m" / "train.csv"
    jsonl_path = tmp_path / "m" / "train.jsonl"
    rep = MetricsReporter(str(csv_path), jsonl_path=str(jsonl_path))
    row = {"a": 1, "b": 2.5, "c": "x"}
    rep.append(row, field_order=["a", "b", "c"])
    rep.append_json({"event": "tick"})
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert rows and rows[0]["a"] == "1"
    assert jsonl_path.read_text(encoding="utf-8").strip().endswith("}")


def test_fen_sanitize_and_flip() -> None:
    raw = "8/8/8/8/8/8/8/8 w - -"
    san = sanitize_fen(raw)
    assert len(san.split()) >= 6
    flipped = flip_fen_perspective(san)
    assert len(flipped.split()) >= 6 and flipped != san


def test_build_optimizer_param_groups() -> None:
    m = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, bias=False),
        torch.nn.BatchNorm2d(4),
        torch.nn.Linear(4, 2),
    )
    opt = build_optimizer(m)
    assert len(opt.param_groups) == 2
    wds = sorted({pg["weight_decay"] for pg in opt.param_groups})
    assert wds == [0.0, wds[-1]]


def test_warmup_cosine_and_ema() -> None:
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.0)
    sch = WarmupCosine(opt, base_lr=1e-2, warmup_steps=2, final_lr=1e-3, total_steps=10)
    lrs = []
    for _ in range(6):
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert math.isclose(lrs[1], 1e-2, rel_tol=1e-5)
    assert all(next_lr <= curr_lr + 1e-9 for curr_lr, next_lr in zip(lrs[1:], lrs[2:]))

    model = torch.nn.Linear(2, 2)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)


def test_select_move_prefers_argmax_when_temperature_low() -> None:
    counts = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    rng = default_rng(3)
    assert select_visit_count_move(counts, temperature=0.0, rng=rng) == 1

    empty_choice = select_visit_count_move(np.array([], dtype=np.float32), temperature=1.0, rng=rng)
    assert empty_choice == 0


def test_select_move_sampling_uses_probabilities() -> None:
    counts = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rng_impl = default_rng(42)
    rng_expected = default_rng(42)
    index = select_visit_count_move(counts, temperature=1.0, rng=rng_impl)

    scaled = np.maximum(counts, 0.0) ** 1.0
    probabilities = scaled / scaled.sum()
    expected_index = rng_expected.choice(len(probabilities), p=probabilities)

    assert index == expected_index
    assert float(probabilities.sum()) == pytest.approx(1.0, rel=1e-6, abs=1e-6)


@pytest.mark.skipif(not _ARENA_AVAILABLE, reason="missing chesscore or arena module")
def test_square_helpers_round_trip() -> None:
    square = 27
    assert _square_file(square) == 3
    assert _square_rank(square) == 3
    assert _square_name(square) == "d4"

    last_square = 63
    assert _square_name(last_square) == "h8"


@pytest.mark.skipif(not _ARENA_AVAILABLE, reason="missing chesscore or arena module")
def test_score_tracker_summarises_results() -> None:
    tracker = _ScoreTracker(total_games=4)
    tracker.record(True, ccore.WHITE_WIN)
    tracker.record(True, ccore.DRAW)
    tracker.record(False, ccore.BLACK_WIN)
    tracker.record(False, ccore.WHITE_WIN)

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


@pytest.mark.skipif(not _ARENA_AVAILABLE, reason="missing chesscore or arena module")
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


@pytest.mark.skipif(not _ARENA_AVAILABLE, reason="missing chesscore or arena module")
def test_result_to_str_handled_outcomes() -> None:
    assert _result_to_str(ccore.WHITE_WIN) == "1-0"
    assert _result_to_str(ccore.BLACK_WIN) == "0-1"
    assert _result_to_str(ccore.DRAW) == "1/2-1/2"
