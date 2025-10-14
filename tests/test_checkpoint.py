from __future__ import annotations

from pathlib import Path

import checkpoint
import torch


class DummyTrainer:
    def __init__(self, run_root: Path) -> None:
        self.iteration = 5
        self.total_games = 12
        self.device = torch.device("cpu")
        self.device_name = "cpu"
        self.start_time = 0.0
        self.gate = type("Gate", (), {"accepted": 0, "rejected": 0})()
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
    checkpoint.save_checkpoint(tr)
    checkpoint.save_best_model(tr)
    root = Path(tr.run_root)
    assert (root / "run_info.json").is_file()
    assert (root / "checkpoints" / "latest.pt").is_file()
    assert (root / "checkpoints" / "best.pt").is_file()
    assert (root / "config" / "merged.json").is_file()
    checkpoint.try_resume(tr)