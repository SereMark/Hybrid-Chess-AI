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

        class _Scheduler:
            def __init__(self) -> None:
                self.t = 0
                self.total = 1

            def set_total_steps(self, total: int) -> None:
                self.total = total

            def step(self) -> None:
                return None

        self.scheduler = _Scheduler()
        self.scaler = torch.amp.GradScaler(enabled=False)
        self.ema = None

        class _SelfPlay:
            def get_capacity(self) -> int:
                return 1

            def size(self) -> int:
                return 0

            def play_games(self, games: int) -> dict[str, int]:
                return {"games": games}

            def sample_batch(
                self,
                batch_size: int,
                recent_ratio: float,
                recent_window: float,
            ) -> tuple[list, list, list, list]:  # noqa: ARG002
                return ([], [], [], [])

        self.selfplay_engine = _SelfPlay()
        self.run_root = str(run_root)

        class _Log:
            def info(self, *args, **kwargs) -> None:
                return None

            def warning(self, *args, **kwargs) -> None:
                return None

        self.log = _Log()


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    trainer = DummyTrainer(tmp_path)
    checkpoint.save_checkpoint(trainer)
    run_root = Path(trainer.run_root)
    metadata = run_root / "run_info.json"
    assert metadata.is_file()
    ckpt_path = run_root / "checkpoints" / "latest.pt"
    assert ckpt_path.is_file()
    config_json = run_root / "config" / "merged.json"
    assert config_json.is_file()
    arena_dir = run_root / "arena_games"
    assert arena_dir.is_dir()
    checkpoint.try_resume(trainer)
