from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler

import config as C

def _timestamped_path(base_path: str, iteration: int | None = None) -> str:
    stem, ext = os.path.splitext(base_path)
    tag = f"_iter{int(iteration):05d}" if iteration is not None else ""
    return f"{stem}{tag}_{int(time.time())}{ext or '.pt'}"


def _resolve_run_root(trainer: Any) -> str:
    existing = getattr(trainer, "run_root", None)
    if existing:
        return str(existing)
    env_root = os.environ.get("HYBRID_CHESS_RUN_DIR")
    run_id = os.environ.get("HYBRID_CHESS_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
    root = os.path.join(env_root or C.LOG.runs_dir, run_id)
    trainer.run_root = root
    return root


def get_run_root(trainer: Any) -> str:
    return _resolve_run_root(trainer)


def _experiment_paths(trainer: Any, iteration: int | None = None) -> dict[str, str]:
    root = _resolve_run_root(trainer)
    paths = {
        "root": root,
        "checkpoint": os.path.join(root, "checkpoints", "latest.pt"),
        "best_model": os.path.join(root, "checkpoints", "best.pt"),
        "metadata": os.path.join(root, "run_info.json"),
    }
    if iteration is not None:
        paths["checkpoint_archive"] = _timestamped_path(paths["checkpoint"], iteration)
    return paths


def _write_config_snapshot(snapshot: dict[str, Any], config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "merged.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    try:
        import yaml
    except Exception:
        yaml = None
    if yaml is not None:
        with (config_dir / "merged.yaml").open("w", encoding="utf-8") as h:
            yaml.safe_dump(snapshot, h, sort_keys=False)


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
    meta["log_paths"] = {"metrics_csv": str(metrics_path), "arena_dir": arena_dir}

    snapshot: dict[str, Any] = {}
    for name in ("LOG", "TORCH", "DATA", "MODEL", "EVAL", "SELFPLAY", "REPLAY", "SAMPLING", "AUGMENT", "MCTS", "RESIGN", "TRAIN", "ARENA"):
        cfg = getattr(C, name, None)
        if cfg is None:
            continue
        snapshot[name] = asdict(cfg) if is_dataclass(cfg) and not isinstance(cfg, type) else cfg
    meta["config_snapshot"] = snapshot

    meta_path = Path(artifacts["metadata"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_config_snapshot(snapshot, meta_path.parent / "config")


def save_checkpoint(trainer: Any) -> None:
    """Persist full trainer state."""
    try:
        artifacts = _experiment_paths(trainer, trainer.iteration)
        ckpt_path = artifacts["checkpoint"]
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

        base = getattr(trainer.model, "_orig_mod", trainer.model)
        payload = {
            "iter": int(trainer.iteration),
            "total_games": int(trainer.total_games),
            "elapsed_s": float(max(0.0, time.time() - trainer.start_time)),
            "model": base.state_dict(),
            "best_model": trainer.best_model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": {"t": int(trainer.scheduler.t), "total": int(trainer.scheduler.total)},
            "scaler": trainer.scaler.state_dict(),
            "ema": trainer.ema.shadow if trainer.ema is not None else None,
            "gate": {"accepted": int(trainer.gate.accepted), "rejected": int(trainer.gate.rejected)},
            "rng": {
                "py": __import__("random").getstate(),
                "np": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        tmp = f"{ckpt_path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, ckpt_path)
        if C.LOG.archive_checkpoints and "checkpoint_archive" in artifacts:
            torch.save(payload, artifacts["checkpoint_archive"])
        _write_metadata(trainer, artifacts)
        trainer.log.info("[CKPT] saved -> %s", ckpt_path)
    except Exception as exc:
        trainer.log.warning("Checkpoint save failed: %s", exc)


def save_best_model(trainer: Any) -> None:
    """Persist best-model snapshot."""
    try:
        artifacts = _experiment_paths(trainer, trainer.iteration)
        best_path = artifacts["best_model"]
        Path(best_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"iter": int(trainer.iteration), "total_games": int(trainer.total_games), "model": trainer.best_model.state_dict()}
        tmp = f"{best_path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, best_path)
        trainer.log.info("[BEST] saved -> %s", best_path)
    except Exception as exc:
        trainer.log.warning("Best-model save failed: %s", exc)


def try_resume(trainer: Any) -> None:
    """Restore trainer state from latest checkpoint if available."""
    candidates: list[str] = []
    env_root = os.environ.get("HYBRID_CHESS_RESUME")
    if env_root:
        candidates.append(os.path.join(env_root, "checkpoints", "latest.pt"))
    if getattr(trainer, "run_root", None):
        candidates.append(os.path.join(trainer.run_root, "checkpoints", "latest.pt"))

    runs_dir = Path(C.LOG.runs_dir)
    if runs_dir.is_dir():
        recent = sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        for run in recent:
            cand = run / "checkpoints" / "latest.pt"
            if cand.is_file():
                candidates.append(str(cand))
                break

    seen: set[str] = set()
    path: str | None = None
    for c in candidates:
        if not c:
            continue
        norm = os.path.normpath(c)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(c):
            path = c
            break

    if path is None:
        trainer.log.info("[CKPT] none found; fresh run")
        return

    try:
        ckpt = torch.load(path, map_location="cpu")
        base = getattr(trainer.model, "_orig_mod", trainer.model)
        base.load_state_dict(ckpt.get("model", {}), strict=True)

        if trainer.ema is not None and ckpt.get("ema") is not None:
            trainer.ema.shadow = ckpt["ema"]

        run_root = Path(path).resolve().parent.parent if "checkpoints" in Path(path).parts else None
        if "best_model" in ckpt:
            trainer.best_model.load_state_dict(ckpt["best_model"], strict=True)
        elif run_root:
            best_path = run_root / "checkpoints" / "best.pt"
            if best_path.is_file():
                best_ckpt = torch.load(best_path, map_location="cpu")
                state = best_ckpt.get("model", best_ckpt)
                trainer.best_model.load_state_dict(state, strict=True)
            trainer.run_root = str(run_root)

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
    except Exception as exc:
        trainer.log.warning("[CKPT] resume failed: %s", exc)