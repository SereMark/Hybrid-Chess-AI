from __future__ import annotations

import importlib
import os

import config as C


def test_load_override_nested() -> None:
    fresh = importlib.reload(C)
    override = {"TRAIN": {"learning_rate_init": 1e-3, "total_iterations": 10}, "MCTS": {"train_simulations": 42}}
    fresh.load_override(override)
    assert fresh.TRAIN.learning_rate_init == 1e-3
    assert fresh.TRAIN.total_iterations == 10
    assert fresh.MCTS.train_simulations == 42


def test_manager_reset_and_snapshot() -> None:
    fresh = importlib.reload(C)
    snap = fresh.snapshot()
    fresh.load_override({"TRAIN": {"total_iterations": 5}})
    assert fresh.TRAIN.total_iterations == 5
    fresh.reset()
    assert fresh.TRAIN.total_iterations == snap["TRAIN"].total_iterations


def test_load_file_yaml_and_env(tmp_path) -> None:
    fresh = importlib.reload(C)
    y = tmp_path / "over.yaml"
    y.write_text("TRAIN:\n  learning_rate_init: 0.0005\nMODEL:\n  channels: 64\n", encoding="utf-8")
    fresh.load_file(y)
    assert fresh.TRAIN.learning_rate_init == 5e-4 and fresh.MODEL.channels == 64

    y2 = tmp_path / "env.yaml"
    y2.write_text("MCTS:\n  train_simulations: 77\n", encoding="utf-8")
    os.environ["HYBRID_CHESS_CONFIG"] = str(y2)
    fresh = importlib.reload(C)
    assert fresh.MCTS.train_simulations == 77