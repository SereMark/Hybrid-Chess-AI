from __future__ import annotations

from dataclasses import replace

import config as C
import torch
from train_loop import run_training_iteration


class DummySelfPlay:
    def __init__(self) -> None:
        self._games = 0
        self.resign_threshold = 0.0
        self.resign_min_plies = 0
        self.resign_consecutive = 0
        self.adjudication_phase = "disabled"
        self.adjudication_enabled = False
        self.adjudication_min_plies = 0
        self.adjudication_value_margin = 0.0
        self.adjudication_persist = 0
        self.adjudication_material_margin = 0.0

    def get_capacity(self) -> int:
        return 1

    def size(self) -> int:
        return 1

    def play_games(self, games: int) -> dict[str, int]:
        self._games += games
        return {"games": games}

    def sample_batch(self, batch_size: int, recent_ratio: float, recent_window: float):
        return ([], [], [], [])

    def enable_resign(self, enabled: bool) -> None:
        return None

    def set_resign_params(self, threshold: float, min_plies: int) -> None:
        self.resign_threshold = threshold
        self.resign_min_plies = min_plies

    def update_adjudication(self, iteration: int) -> None:
        self.adjudication_phase = "active" if iteration >= 0 else "disabled"
        self.adjudication_enabled = True


class DummyTrainer:
    def __init__(self) -> None:
        self.selfplay_engine = DummySelfPlay()
        self.iteration = 0
        self.train_batch_size = 1
        self.model = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = type("Sched", (), {"step": lambda self: None})()
        self.device = torch.device("cpu")
        self.scaler = torch.amp.GradScaler(enabled=False)
        self.ema = None
        self._amp_enabled = False
        self._autocast_dtype = torch.float32


def test_run_training_iteration_handles_no_batches(monkeypatch) -> None:
    tr = DummyTrainer()
    monkeypatch.setattr(type(C.TRAIN), "update_steps_min", 1, raising=False)
    stats = run_training_iteration(tr)
    assert stats["train_steps_actual"] == 0
    entropy_coef = float(stats["entropy_coef"])
    assert 0.0 <= entropy_coef <= float(C.TRAIN.loss_entropy_coef)


def test_run_training_iteration_sets_resign_params(monkeypatch) -> None:
    tr = DummyTrainer()
    tr.iteration = C.RESIGN.cooldown_iters
    monkeypatch.setattr(
        C,
        "RESIGN",
        replace(C.RESIGN, enabled=True, value_threshold=-0.5, min_plies=10, cooldown_iters=0),
        raising=False,
    )
    stats = run_training_iteration(tr)
    assert tr.selfplay_engine.resign_threshold == -0.5
    assert tr.selfplay_engine.resign_min_plies == 10
    assert "selfplay_stats" in stats
