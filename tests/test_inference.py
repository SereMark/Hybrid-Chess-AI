from __future__ import annotations

import numpy as np
import pytest
import torch
from inference import BatchedEvaluator


def _make_positions(ensure_chesscore, count: int) -> list:
    seed = ensure_chesscore.Position()
    positions: list = []
    for offset in range(count):
        pos = ensure_chesscore.Position(seed)
        for step in range(offset + 1):
            moves = list(pos.legal_moves())
            if not moves:
                break
            pos.make_move(moves[step % len(moves)])
        positions.append(pos)
    return positions


def test_batched_evaluator_caches_and_metrics(ensure_chesscore) -> None:
    ev = BatchedEvaluator(torch.device("cpu"))
    try:
        pos = ensure_chesscore.Position()
        moves = list(pos.legal_moves())
        pol1, val1 = ev.infer_positions_legal([pos], [moves])
        assert len(pol1) == 1 and val1.shape == (1,)
        m1 = ev.get_metrics()
        pol2, val2 = ev.infer_positions_legal([pos], [moves])
        np.testing.assert_allclose(val1, val2)
        m2 = ev.get_metrics()
        assert m2["cache_hits_total"] >= m1["cache_hits_total"]
    finally:
        ev.close()


def test_batched_evaluator_refresh_clears_caches(ensure_chesscore) -> None:
    ev = BatchedEvaluator(torch.device("cpu"))
    try:
        pos = ensure_chesscore.Position()
        _ = ev.infer_values([pos])
        before = ev.get_metrics()["cache_misses_total"]
        ev.refresh_from(ev.eval_model)  # no-op weights, but clears caches
        _ = ev.infer_values([pos])
        after = ev.get_metrics()["cache_misses_total"]
        assert after >= before + 1
    finally:
        ev.close()


def test_batched_evaluator_handles_shutdown_and_coalesce(ensure_chesscore) -> None:
    ev = BatchedEvaluator(torch.device("cpu"))
    pos = ensure_chesscore.Position()
    moves = [list(pos.legal_moves())]

    # Force immediate shutdown to trigger error path
    ev.close()
    with pytest.raises(RuntimeError):
        ev.infer_positions_legal([pos], moves)


def test_batched_evaluator_respects_cache_capacity(ensure_chesscore) -> None:
    ev = BatchedEvaluator(torch.device("cpu"))
    try:
        ev.set_cache_capacity(capacity=2, value_capacity=3, encode_capacity=2)
        positions = _make_positions(ensure_chesscore, 4)
        moves = [list(p.legal_moves()) for p in positions]
        _ = ev.infer_positions_legal(positions, moves)

        metrics = ev.get_metrics()
        assert metrics["cache_misses_total"] >= 4

        ev.set_cache_capacity(capacity=1, value_capacity=1, encode_capacity=1)
        ev_metrics = ev.get_metrics()
        assert ev_metrics["cache_misses_total"] >= metrics["cache_misses_total"]
    finally:
        ev.close()


def test_batched_evaluator_cpu_fp16_uses_fp32_inputs(monkeypatch, ensure_chesscore) -> None:
    ev = BatchedEvaluator(torch.device("cpu"))
    try:
        monkeypatch.setattr(ev, "_dtype", torch.float16, raising=False)
        monkeypatch.setattr(ev, "_host_infer_dtype", torch.float32, raising=False)
        monkeypatch.setattr(ev, "_should_autocast", False, raising=False)

        pos = ensure_chesscore.Position()
        result = ev.infer_values([pos])
        assert result.shape == (1,)
    finally:
        ev.close()