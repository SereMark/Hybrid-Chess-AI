from __future__ import annotations

from dataclasses import replace
from typing import TypedDict

import config as C
import numpy as np
import pytest
import torch

pytest.importorskip("chesscore", reason="chesscore extension missing")

import encoder
from train_loop import run_training_iteration


class _Stats(TypedDict, total=False):
    games: int
    moves: int
    visit_per_move: float


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

    def play_games(self, games: int) -> _Stats:
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
        self.model: torch.nn.Module = torch.nn.Linear(2, 2)
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


def test_run_training_iteration_enables_adjudication_and_buffer(monkeypatch) -> None:
    tr = DummyTrainer()

    class _StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            features = encoder.INPUT_PLANES * encoder.BOARD_SIZE * encoder.BOARD_SIZE
            self.policy = torch.nn.Linear(features, encoder.POLICY_SIZE)
            self.value = torch.nn.Linear(features, 1)

        def forward(self, x: torch.Tensor):
            flat = x.reshape(x.shape[0], -1)
            policy_logits = self.policy(flat)
            value = torch.tanh(self.value(flat)).squeeze(-1)
            return policy_logits, value

    tr.model = _StubModel().to(tr.device)
    tr.optimizer = torch.optim.SGD(tr.model.parameters(), lr=0.01)
    tr.scheduler = type("Sched", (), {"step": lambda self: None})()

    calls = {"play": 0, "sample": 0}

    def fake_play(games: int) -> _Stats:
        calls["play"] += 1
        return {"games": games, "moves": games * 4, "visit_per_move": 32.0}

    def fake_sample(batch_size: int, recent_ratio: float, recent_window: float):
        calls["sample"] += 1
        state = np.zeros((encoder.INPUT_PLANES, encoder.BOARD_SIZE, encoder.BOARD_SIZE), dtype=np.uint8)
        idx = np.array([1, 2], dtype=np.int32)
        cnt = np.array([10, 11], dtype=np.uint16)
        return ([state], [idx], [cnt], [np.int8(1)])

    tr.selfplay_engine.play_games = fake_play  # type: ignore[assignment]
    tr.selfplay_engine.sample_batch = fake_sample  # type: ignore[assignment]
    monkeypatch.setattr(type(C.TRAIN), "update_steps_min", 1, raising=False)
    monkeypatch.setattr(type(C.TRAIN), "update_steps_max", 2, raising=False)
    monkeypatch.setattr(type(C.TRAIN), "samples_per_new_game", 1.0, raising=False)

    stats = run_training_iteration(tr)

    assert calls["play"] == 1
    assert calls["sample"] >= 1
    assert int(stats["train_steps_actual"]) >= 1
    assert stats["adjudication_enabled"] == 1


def test_run_training_iteration_handles_bad_gradients(monkeypatch) -> None:
    tr = DummyTrainer()

    class ExplodingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(encoder.INPUT_PLANES * encoder.BOARD_SIZE * encoder.BOARD_SIZE, 1)

        def forward(self, x: torch.Tensor):
            flat = x.reshape(x.shape[0], -1)
            w = self.lin(flat).squeeze(-1)
            return torch.full((x.shape[0], encoder.POLICY_SIZE), 0.0, device=x.device), torch.tanh(w)

    tr.model = ExplodingModel().to(tr.device)
    tr.optimizer = torch.optim.SGD(tr.model.parameters(), lr=0.1)

    def fake_sample(batch_size: int, recent_ratio: float, recent_window: float):
        state = np.ones((encoder.INPUT_PLANES, encoder.BOARD_SIZE, encoder.BOARD_SIZE), dtype=np.uint8)
        idx = np.array([0], dtype=np.int32)
        cnt = np.array([1], dtype=np.uint16)
        return ([state], [idx], [cnt], [np.int8(1)])

    tr.selfplay_engine.sample_batch = fake_sample  # type: ignore[assignment]

    original_clip = torch.nn.utils.clip_grad_norm_

    def clip_and_raise(*args, **kwargs):
        raise RuntimeError("grad error")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_and_raise)

    with pytest.raises(RuntimeError):
        run_training_iteration(tr)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", original_clip)


def test_run_training_iteration_updates_ema(monkeypatch) -> None:
    tr = DummyTrainer()
    tr.ema = type("EMA", (), {"update": lambda self, model: setattr(self, "touched", True), "touched": False})()

    class SimpleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(encoder.INPUT_PLANES * encoder.BOARD_SIZE * encoder.BOARD_SIZE, 1)

        def forward(self, x: torch.Tensor):
            flat = x.reshape(x.shape[0], -1)
            logits = torch.zeros((x.shape[0], encoder.POLICY_SIZE), device=x.device)
            val = torch.tanh(self.lin(flat)).squeeze(-1)
            return logits, val

    tr.model = SimpleModel().to(tr.device)
    tr.optimizer = torch.optim.SGD(tr.model.parameters(), lr=0.01)

    def fake_sample(batch_size: int, recent_ratio: float, recent_window: float):
        state = np.zeros((encoder.INPUT_PLANES, encoder.BOARD_SIZE, encoder.BOARD_SIZE), dtype=np.uint8)
        idx = np.array([0], dtype=np.int32)
        cnt = np.array([1], dtype=np.uint16)
        return ([state], [idx], [cnt], [np.int8(1)])

    tr.selfplay_engine.sample_batch = fake_sample  # type: ignore[assignment]

    stats = run_training_iteration(tr)
    assert int(stats["train_steps_actual"]) >= 1
    assert getattr(tr.ema, "touched", False)
