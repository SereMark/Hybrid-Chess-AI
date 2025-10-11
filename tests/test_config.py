from __future__ import annotations

import importlib

import config as C


def test_load_override_nested() -> None:
    fresh = importlib.reload(C)
    assert fresh.TRAIN.learning_rate_init == 7.5e-4
    override = {
        "TRAIN": {"learning_rate_init": 1e-3, "total_iterations": 10},
        "MCTS": {"train_simulations": 42},
    }
    fresh.load_override(override)
    assert fresh.TRAIN.learning_rate_init == 1e-3
    assert fresh.TRAIN.total_iterations == 10
    assert fresh.MCTS.train_simulations == 42


def test_manager_reset_and_snapshot() -> None:
    fresh = importlib.reload(C)
    initial = fresh.snapshot()
    fresh.load_override({"TRAIN": {"total_iterations": 5}})
    assert fresh.TRAIN.total_iterations == 5
    fresh.reset()
    assert fresh.TRAIN.total_iterations == initial["TRAIN"].total_iterations


def test_load_file_yaml(tmp_path) -> None:
    fresh = importlib.reload(C)
    yaml_path = tmp_path / "override.yaml"
    yaml_path.write_text("TRAIN:\n  learning_rate_init: 0.0005\nMODEL:\n  channels: 64\n", encoding="utf-8")
    fresh.load_file(yaml_path)
    assert fresh.TRAIN.learning_rate_init == 5e-4
    assert fresh.MODEL.channels == 64
