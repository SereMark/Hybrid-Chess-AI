from __future__ import annotations

from pathlib import Path

import checkpoint
import pytest
import torch


class DummyTrainer:
    def __init__(self, run_root: Path) -> None:
        self.iteration = 5
        self.total_games = 12
        self.device = torch.device("cpu")
        self.device_name = "cpu"
        self.start_time = 0.0
        self.metrics = type("Metrics", (), {"csv_path": str(run_root / "metrics" / "training.csv")})()
        self.model = torch.nn.Linear(2, 2)
        self.best_model = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        class _Sched:
            def __init__(self) -> None:
                self.t = 0
                self.total = 1

            def set_total_steps(self, total: int) -> None:
                self.total = total

            def step(self) -> None:
                return None

        self.scheduler = _Sched()
        self.scaler = torch.amp.GradScaler(enabled=False)
        self.ema = None
        self.selfplay_engine = type("SP", (), {"get_capacity": lambda self: 1, "size": lambda self: 0})()
        self.run_root = str(run_root)
        self.log = type("Log", (), {"info": lambda *a, **k: None, "warning": lambda *a, **k: None})()


def test_checkpoint_roundtrip_and_best(tmp_path: Path) -> None:
    tr = DummyTrainer(tmp_path)
    tr.iteration = 5
    tr.total_games = 12
    checkpoint.save_checkpoint(tr)
    checkpoint.save_best_model(tr)
    root = Path(tr.run_root)
    assert (root / "run_info.json").is_file()
    assert (root / "checkpoints" / "latest.pt").is_file()
    assert (root / "checkpoints" / "best.pt").is_file()
    assert (root / "config" / "merged.json").is_file()

    tr.iteration = 0
    tr.total_games = 0
    checkpoint.try_resume(tr)
    assert tr.iteration == 5
    assert tr.total_games == 12


def test_resume_uses_full_checkpoint_unpickle(monkeypatch, tmp_path: Path) -> None:
    tr = DummyTrainer(tmp_path)
    checkpoint.save_checkpoint(tr)

    real_load = checkpoint.torch.load
    load_kwargs: list[dict[str, object]] = []

    def recording_load(*args, **kwargs):
        load_kwargs.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint.torch, "load", recording_load)
    checkpoint.try_resume(tr)

    assert load_kwargs
    assert all(kwargs.get("weights_only") is False for kwargs in load_kwargs)


def test_trainer_resume_initializes_restore_targets_before_resume(monkeypatch, tmp_path: Path) -> None:
    trainer_mod = pytest.importorskip("trainer")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Conv2d(1, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer(x)

    class DummyEvaluator:
        def __init__(self, device: torch.device) -> None:
            self.device = device

        def set_batching_params(self, **kwargs) -> None:
            return None

        def set_cache_capacity(self, *args, **kwargs) -> None:
            return None

        def refresh_from(self, model: torch.nn.Module) -> None:
            return None

        def close(self) -> None:
            return None

    class DummySelfPlay:
        def __init__(self, evaluator, *, seed=None) -> None:
            return None

        def close(self) -> None:
            return None

    checked = {"called": False}

    def fake_try_resume(tr) -> None:
        checked["called"] = True
        for name in ("ema", "best_model", "iteration", "total_games", "start_time", "_arena_rng"):
            assert hasattr(tr, name), f"Trainer missing {name} before resume"
        tr.iteration = 7
        tr.total_games = 42

    monkeypatch.setenv("HYBRID_CHESS_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(trainer_mod, "ChessNet", TinyModel)
    monkeypatch.setattr(trainer_mod, "BatchedEvaluator", DummyEvaluator)
    monkeypatch.setattr(trainer_mod, "SelfPlayEngine", DummySelfPlay)
    monkeypatch.setattr(trainer_mod, "try_resume", fake_try_resume)

    tr = trainer_mod.Trainer(device="cpu", resume=True)

    assert checked["called"]
    assert tr.iteration == 7
    assert tr.total_games == 42
