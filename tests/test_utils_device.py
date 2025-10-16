from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import utils
from utils import (
    _json_safe,
    format_time,
    prepare_model,
    select_autocast_dtype,
    select_inference_dtype,
    startup_summary,
)


def test_prepare_model_applies_channels_last_dtype_and_freeze() -> None:
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3), torch.nn.BatchNorm2d(4))
    prepared = prepare_model(
        model,
        torch.device("cpu"),
        channels_last=True,
        dtype=torch.float64,
        eval_mode=True,
        freeze=True,
    )

    assert not prepared.training
    conv = cast(torch.nn.Conv2d, list(prepared.children())[0])
    assert conv.weight.dtype == torch.float64
    try:
        assert conv.weight.is_contiguous(memory_format=torch.channels_last)
    except RuntimeError:
        pytest.skip("channels_last memory format not supported on this platform for conv weights")
    assert all(not p.requires_grad for p in prepared.parameters())


def test_select_autocast_dtype_branches(monkeypatch) -> None:
    assert select_autocast_dtype(torch.device("cpu")) is torch.bfloat16

    monkeypatch.setattr(utils, "_cuda_supports_bf16", lambda: True)
    assert select_autocast_dtype(torch.device("cuda")) is torch.bfloat16

    monkeypatch.setattr(utils, "_cuda_supports_bf16", lambda: False)
    assert select_autocast_dtype(torch.device("cuda")) is torch.float16


def test_select_inference_dtype_honours_cpu_override(monkeypatch) -> None:
    assert select_inference_dtype(torch.device("cpu"), cpu_dtype=torch.float64) is torch.float64

    monkeypatch.setattr(utils, "select_autocast_dtype", lambda device: torch.float16)
    assert select_inference_dtype(torch.device("cuda")) is torch.float16


def test_json_safe_serialises_complex_objects() -> None:
    payload = {
        "path": Path("runs/train"),
        "scalar": np.float32(1.5),
        "array": np.array([1, 2, 3], dtype=np.int16),
        "nested": {"values": (np.float16(0.25), "x")},
    }
    safe = cast(dict[str, Any], _json_safe(payload))
    nested = cast(dict[str, Any], safe["nested"])
    values = cast(list[Any], nested["values"])

    assert safe["path"] == "runs/train"
    assert safe["scalar"] == pytest.approx(1.5)
    assert safe["array"] == [1, 2, 3]
    assert values[0] == pytest.approx(0.25)


def test_format_time_ranges() -> None:
    assert format_time(12.0) == "12.0s"
    assert format_time(120.0) == "2.0m"
    assert format_time(7200.0) == "2.0h"


def test_startup_summary_contains_key_metrics(monkeypatch) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    trainer = type(
        "Trainer",
        (),
        {
            "device": torch.device("cpu"),
            "device_name": "cpu",
            "model": model,
            "train_batch_size": 32,
            "_amp_enabled": False,
            "_autocast_dtype": torch.float32,
            "metrics": type("Metrics", (), {"csv_path": "metrics/train.csv"})(),
        },
    )()

    summary = startup_summary(trainer)
    assert "Hybrid Chess AI training" in summary
    assert "device=cpu" in summary
    assert "metrics_csv=metrics/train.csv" in summary
