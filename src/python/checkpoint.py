"""Checkpoint persistence utilities for training state."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import config as C
import numpy as np
import torch
from torch.amp import GradScaler

__all__ = ["save_checkpoint", "save_best_model", "try_resume", "get_run_root"]


def _timestamped_path(base_path: str, iteration: int | None = None) -> str:
    stem, ext = os.path.splitext(base_path)
    iteration_suffix = f"_iter{int(iteration):05d}" if iteration is not None else ""
    return f"{stem}{iteration_suffix}_{int(time.time())}{ext or '.pt'}"


def _resolve_run_root(trainer: Any) -> str:
    existing = getattr(trainer, "run_root", None)
    if existing:
        return str(existing)
    env_root = os.environ.get("HYBRID_CHESS_RUN_DIR")
    run_id = os.environ.get("HYBRID_CHESS_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
    base_dir = env_root or C.LOG.runs_dir
    root = os.path.join(base_dir, run_id)
    trainer.run_root = root
    return root


def get_run_root(trainer: Any) -> str:
    """Return the directory used for the current training run."""
    return _resolve_run_root(trainer)


def _experiment_paths(trainer: Any, iteration: int | None = None) -> dict[str, str]:
    root = _resolve_run_root(trainer)
    artifacts = {
        "root": root,
        "checkpoint": os.path.join(root, "checkpoints", "latest.pt"),
        "best_model": os.path.join(root, "checkpoints", "best.pt"),
        "metadata": os.path.join(root, "run_info.json"),
    }
    if iteration is not None:
        artifacts["checkpoint_archive"] = _timestamped_path(artifacts["checkpoint"], iteration)
    return artifacts


def _write_config_snapshot(snapshot: dict[str, Any], config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    merged_json = config_dir / "merged.json"
    with merged_json.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - optional
        yaml = None
    if yaml is not None:
        merged_yaml = config_dir / "merged.yaml"
        with merged_yaml.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(snapshot, handle, sort_keys=False)


def _write_metadata(trainer: Any, artifacts: dict[str, str]) -> None:
    meta = {
        "iteration": int(trainer.iteration),
        "total_games": int(trainer.total_games),
        "device": str(trainer.device),
        "device_name": getattr(trainer, "device_name", str(trainer.device)),
        "run_root": artifacts["root"],
        "artifacts": {
            "checkpoint": artifacts["checkpoint"],
            "best_model": artifacts["best_model"],
            "metadata": artifacts["metadata"],
        },
        "seed": int(getattr(C, "SEED", 0)),
        "log_paths": {},
        "timestamps": {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": float(max(0.0, time.time() - trainer.start_time)),
        },
    }
    default_metrics = Path(artifacts["root"]) / "metrics" / "training.csv"
    metrics_path = getattr(getattr(trainer, "metrics", None), "csv_path", str(default_metrics))
    arena_dir = Path(getattr(trainer, "arena_dir", Path(Path(artifacts["root"]) / "arena_games"))).as_posix()
    meta["log_paths"] = {
        "metrics_csv": str(metrics_path),
        "arena_dir": arena_dir,
    }
    snapshot: dict[str, Any] = {}
    for attr_name in (
        "LOG",
        "TORCH",
        "DATA",
        "MODEL",
        "EVAL",
        "SELFPLAY",
        "REPLAY",
        "SAMPLING",
        "AUGMENT",
        "MCTS",
        "RESIGN",
        "TRAIN",
        "ARENA",
    ):
        cfg = getattr(C, attr_name, None)
        if cfg is None:
            continue
        if is_dataclass(cfg) and not isinstance(cfg, type):
            snapshot[attr_name] = asdict(cfg)
        else:
            snapshot[attr_name] = cfg
    meta["config_snapshot"] = snapshot

    metadata_path = Path(artifacts["metadata"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    _write_config_snapshot(snapshot, metadata_path.parent / "config")


def save_checkpoint(trainer: Any) -> None:
    """Persist the trainer state to disk."""
    try:
        artifacts = _experiment_paths(trainer, trainer.iteration)
        ckpt_path = artifacts["checkpoint"]
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        base = getattr(trainer.model, "_orig_mod", trainer.model)
        payload = {
            "iter": int(trainer.iteration),
            "total_games": int(trainer.total_games),
            "elapsed_s": float(max(0.0, time.time() - trainer.start_time)),
            "model": base.state_dict(),
            "best_model": trainer.best_model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": {
                "t": int(trainer.scheduler.t),
                "total": int(trainer.scheduler.total),
            },
            "scaler": trainer.scaler.state_dict(),
            "ema": trainer.ema.shadow if trainer.ema is not None else None,
            "gate": {
                "accepted": int(trainer.gate.accepted),
                "rejected": int(trainer.gate.rejected),
            },
            "rng": {
                "py": __import__("random").getstate(),
                "np": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        tmp_path = f"{ckpt_path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)
        if C.LOG.archive_checkpoints and "checkpoint_archive" in artifacts:
            torch.save(payload, artifacts["checkpoint_archive"])
        _write_metadata(trainer, artifacts)
        trainer.log.info("[CKPT] saved checkpoint -> %s", ckpt_path)
    except Exception as exc:  # pragma: no cover - defensive
        trainer.log.warning("Failed to save checkpoint: %s", exc)


def save_best_model(trainer: Any) -> None:
    """Persist the best-performing model snapshot."""
    try:
        artifacts = _experiment_paths(trainer, trainer.iteration)
        best_path = artifacts["best_model"]
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        payload = {
            "iter": int(trainer.iteration),
            "total_games": int(trainer.total_games),
            "model": trainer.best_model.state_dict(),
        }
        tmp_path = f"{best_path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, best_path)
        trainer.log.info("[BEST] saved best model -> %s", best_path)
    except Exception as exc:  # pragma: no cover - defensive
        trainer.log.warning("Failed to save best model: %s", exc)


def try_resume(trainer: Any) -> None:
    """Restore trainer state from a previously saved checkpoint if present."""
    candidates: list[str] = []
    resume_root = os.environ.get("HYBRID_CHESS_RESUME")
    if resume_root:
        candidates.append(os.path.join(resume_root, "checkpoints", "latest.pt"))
    if getattr(trainer, "run_root", None):
        candidates.append(os.path.join(trainer.run_root, "checkpoints", "latest.pt"))
    runs_dir = Path(C.LOG.runs_dir)
    if runs_dir.is_dir():
        run_dirs = sorted(
            (p for p in runs_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run in run_dirs:
            candidate = run / "checkpoints" / "latest.pt"
            if candidate.is_file():
                candidates.append(str(candidate))
                break

    path = None
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(candidate):
            path = candidate
            break

    if path is None:
        trainer.log.info("[CKPT] no checkpoint found; starting fresh")
        return
    try:
        ckpt = torch.load(path, map_location="cpu")
        base = getattr(trainer.model, "_orig_mod", trainer.model)
        base.load_state_dict(ckpt.get("model", {}), strict=True)
        if trainer.ema is not None and ckpt.get("ema") is not None:
            trainer.ema.shadow = ckpt["ema"]
        run_root = Path(path).resolve().parent.parent if "checkpoints" in Path(path).parts else None
        best_candidates: list[str] = []
        if run_root:
            trainer.run_root = str(run_root)
            best_candidates.append(str(run_root / "checkpoints" / "best.pt"))
            meta_path = run_root / "run_info.json"
            if meta_path.is_file():
                trainer.log.info("[CKPT] loaded run metadata from %s", meta_path)
        if "best_model" in ckpt:
            trainer.best_model.load_state_dict(ckpt["best_model"], strict=True)
        else:
            for best_path in best_candidates:
                if os.path.isfile(best_path):
                    best_ckpt = torch.load(best_path, map_location="cpu")
                    state_dict = best_ckpt.get("model", best_ckpt)
                    trainer.best_model.load_state_dict(state_dict, strict=True)
                    break
        trainer.best_model.eval()
        if "optimizer" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer"])
        sched = ckpt.get("scheduler")
        if isinstance(sched, dict):
            trainer.scheduler.set_total_steps(int(sched.get("total", trainer.scheduler.total)))
            trainer.scheduler.t = int(sched.get("t", trainer.scheduler.t))
        if "scaler" in ckpt and isinstance(trainer.scaler, GradScaler):
            trainer.scaler.load_state_dict(ckpt["scaler"])
        trainer.iteration = int(ckpt.get("iter", 0))
        trainer.total_games = int(ckpt.get("total_games", 0))
        elapsed_s = float(ckpt.get("elapsed_s", 0.0))
        if elapsed_s > 0.0:
            trainer.start_time = time.time() - elapsed_s
        gate = ckpt.get("gate", {})
        trainer.gate.accepted = int(gate.get("accepted", 0))
        trainer.gate.rejected = int(gate.get("rejected", 0))
        rng = ckpt.get("rng", {})
        if "py" in rng:
            __import__("random").setstate(rng["py"])
        if "np" in rng:
            np.random.set_state(rng["np"])
        if "torch_cpu" in rng:
            torch.set_rng_state(rng["torch_cpu"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        trainer.log.info("[CKPT] resumed from %s @ iter %s", path, trainer.iteration)
    except Exception as exc:  # pragma: no cover - defensive
        trainer.log.warning("[CKPT] failed to resume: %s", exc)
