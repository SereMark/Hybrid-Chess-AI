from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import utils
from torch import nn


def test_format_helpers() -> None:
    assert utils.format_gb(1.234) == "1.2G"
    assert utils.format_si(1_500) == "1.5k"
    assert utils.format_si(-2_500_000, digits=2) == "-2.50M"
    assert utils.format_time(45) == "45.0s"
    assert utils.format_time(120) == "2.0m"


def test_metrics_reporter_appends(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    json_path = tmp_path / "metrics.jsonl"
    reporter = utils.MetricsReporter(str(csv_path), jsonl_path=str(json_path))

    first_row = {"step": 1, "value": 0.5}
    second_row = {"step": 2, "value": 0.25}
    reporter.append(first_row, field_order=["step", "value"])
    reporter.append(second_row)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"step": "1", "value": "0.5"},
        {"step": "2", "value": "0.25"},
    ]

    with json_path.open(encoding="utf-8") as handle:
        parsed = [json.loads(line) for line in handle if line.strip()]
    assert parsed == [first_row, second_row]


def test_prepare_model_channels_last() -> None:
    model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU())
    prepared = utils.prepare_model(model, torch.device("cpu"), channels_last=True, eval_mode=True)
    assert prepared is model
    assert not prepared.training
    for param in prepared.parameters():
        assert param.requires_grad is True
