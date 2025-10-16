from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from optimization import EMA, WarmupCosine, build_optimizer
from utils import MetricsReporter, flip_fen_perspective, sanitize_fen


def test_metrics_reporter_writes_csv_and_jsonl(tmp_path: Path) -> None:
    csv_path = tmp_path / "m" / "train.csv"
    jsonl_path = tmp_path / "m" / "train.jsonl"
    rep = MetricsReporter(str(csv_path), jsonl_path=str(jsonl_path))
    row = {"a": 1, "b": 2.5, "c": "x"}
    rep.append(row, field_order=["a", "b", "c"])
    rep.append_json({"event": "tick"})
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert rows and rows[0]["a"] == "1"
    assert jsonl_path.read_text(encoding="utf-8").strip().endswith("}")


def test_fen_sanitize_and_flip() -> None:
    raw = "8/8/8/8/8/8/8/8 w - -"
    san = sanitize_fen(raw)
    assert len(san.split()) >= 6
    flipped = flip_fen_perspective(san)
    assert len(flipped.split()) >= 6 and flipped != san


def test_build_optimizer_param_groups() -> None:
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, bias=False), torch.nn.BatchNorm2d(4), torch.nn.Linear(4, 2))
    opt = build_optimizer(m)
    assert len(opt.param_groups) == 2
    wds = sorted({pg["weight_decay"] for pg in opt.param_groups})
    assert wds == [0.0, wds[-1]]


def test_warmup_cosine_and_ema() -> None:
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.0)
    sch = WarmupCosine(opt, base_lr=1e-2, warmup_steps=2, final_lr=1e-3, total_steps=10)
    lrs = []
    for _ in range(6):
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert math.isclose(lrs[1], 1e-2, rel_tol=1e-5)  # end warmup
    assert all(next_lr <= curr_lr + 1e-9 for curr_lr, next_lr in zip(lrs[1:], lrs[2:]))

    model = torch.nn.Linear(2, 2)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)  # shadow should move toward params without error
