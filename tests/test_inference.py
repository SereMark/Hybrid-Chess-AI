from __future__ import annotations

import numpy as np
import torch
from inference import BatchedEvaluator


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
