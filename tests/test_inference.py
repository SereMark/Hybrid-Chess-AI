from __future__ import annotations

import numpy as np
import torch
from inference import BatchedEvaluator


def test_batched_evaluator_caches(ensure_chesscore) -> None:  # noqa: ARG001
    evaluator = BatchedEvaluator(torch.device("cpu"))
    try:
        position = ensure_chesscore.Position()
        moves = list(position.legal_moves())
        pol_list, values = evaluator.infer_positions_legal([position], [moves])
        assert len(pol_list) == 1
        assert values.shape == (1,)
        pol_list2, values2 = evaluator.infer_positions_legal([position], [moves])
        np.testing.assert_almost_equal(values, values2)
    finally:
        evaluator.close()
