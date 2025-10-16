from __future__ import annotations

"""Shared helpers for benchmarking scripts in the tools package."""

import sys
import csv
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

_CANDIDATE_EXT_DIRS: tuple[Path, ...] = (
    REPO_ROOT / "build" / "python" / "Release",
    REPO_ROOT / "build" / "python" / "Debug",
    REPO_ROOT / "build" / "python",
    REPO_ROOT / "src" / "python",
)


def _append_sys_path(paths: Iterable[Path]) -> None:
    """Add existing directories to sys.path in order."""
    for path in paths:
        if not path.exists():
            continue
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def prepare_extension_import_paths(extra: Sequence[Path] | None = None) -> None:
    """Ensure candidate extension directories (and optional extras) are on sys.path."""
    _append_sys_path(_CANDIDATE_EXT_DIRS)
    if extra:
        _append_sys_path(extra)


def import_chesscore(module_name: str = "chesscore") -> ModuleType:
    """Import the compiled chesscore extension, raising a helpful error when missing."""
    prepare_extension_import_paths()
    try:
        module = __import__(module_name)
    except ImportError as exc:  # pragma: no cover - defensive
        tried = ", ".join(str(p) for p in _CANDIDATE_EXT_DIRS)
        raise SystemExit(
            "Failed to import chesscore. Build the extension with CMake and ensure one of these directories is on "
            f"PYTHONPATH: {tried}"
        ) from exc
    return module


def import_python_chess() -> ModuleType:
    """Import python-chess, prompting the user to install it if missing."""
    try:
        import chess  # type: ignore[import,unused-ignore]
    except ImportError as exc:  # pragma: no cover - defensive
        raise SystemExit("Missing dependency: python-chess (pip install python-chess)") from exc
    return chess


__all__ = [
    "REPO_ROOT",
    "prepare_extension_import_paths",
    "import_chesscore",
    "import_python_chess",
    "Measurement",
    "benchmark_callable",
    "summarize_measurements",
    "render_markdown_table",
    "write_reports",
]


# ---------------------------------------------------------------------------#
# Benchmark helpers


@dataclass(slots=True)
class Measurement:
    """Timing samples collected for a specific benchmark scenario."""

    name: str
    samples: list[float]
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        mean = statistics.fmean(self.samples) if self.samples else 0.0
        stdev = statistics.pstdev(self.samples) if len(self.samples) > 1 else 0.0
        return {
            "name": self.name,
            "sample_count": len(self.samples),
            "mean_s": mean,
            "stdev_s": stdev,
            "min_s": min(self.samples) if self.samples else 0.0,
            "max_s": max(self.samples) if self.samples else 0.0,
            **self.extras,
        }


def benchmark_callable(
    name: str,
    fn: Callable[[], Any],
    *,
    warmup: int = 1,
    repeat: int = 5,
) -> Measurement:
    """Execute a callable repeatedly and capture timing samples."""

    for _ in range(max(0, warmup)):
        fn()

    samples: list[float] = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return Measurement(name=name, samples=samples)


def summarize_measurements(measurements: Sequence[Measurement], *, metadata: dict[str, Any] | None = None) -> dict:
    """Return a JSON-serialisable report describing benchmark outcomes."""

    summaries: list[dict[str, Any]] = []
    for measurement in measurements:
        summary = measurement.summary()
        summary.setdefault("sample_count", len(measurement.samples))
        summaries.append(summary)
    return {
        "metadata": metadata or {},
        "measurements": summaries,
    }


def render_markdown_table(measurements: Sequence[Measurement]) -> str:
    """Render a simple Markdown table summarising measurement statistics."""

    if not measurements:
        return "No measurements collected."

    rows = [m.summary() for m in measurements]
    extra_keys: list[str] = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in {"name", "samples", "mean_s", "stdev_s", "min_s", "max_s"}
        }
    )

    headers = [
        "Scenario",
        "Mean (ms)",
        "Std (ms)",
        "Min (ms)",
        "Max (ms)",
        *(key.replace("_", " ").title() for key in extra_keys),
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]

    for row in rows:
        base = [
            row["name"],
            f"{row['mean_s'] * 1000:.3f}",
            f"{row['stdev_s'] * 1000:.3f}",
            f"{row['min_s'] * 1000:.3f}",
            f"{row['max_s'] * 1000:.3f}",
        ]
        extras = [str(row.get(key, "")) for key in extra_keys]
        lines.append("| " + " | ".join(base + extras) + " |")
    return "\n".join(lines)


def write_reports(
    report: dict,
    *,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
    csv_path: Path | None = None,
) -> None:
    """Persist a benchmark report to optional JSON/Markdown/CSV outputs."""

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2))

    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        summaries = report.get("measurements", [])
        measurements = []
        for entry in summaries:
            extras = {
                k: v
                for k, v in entry.items()
                if k not in {"name", "mean_s", "stdev_s", "min_s", "max_s", "sample_count"}
            }
            measurement = Measurement(name=entry["name"], samples=[], extras=extras)
            measurements.append(measurement)
        markdown_path.write_text(render_markdown_table(measurements))

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = report.get("measurements", [])
        if rows:
            clean_rows: list[dict[str, Any]] = []
            fieldnames: set[str] = set()
            for row in rows:
                clean = {k: v for k, v in row.items() if k != "samples"}
                clean_rows.append(clean)
                fieldnames.update(clean.keys())
            ordered_fields = sorted(fieldnames)
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=ordered_fields)
                writer.writeheader()
                writer.writerows(clean_rows)
