from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch

import config as C


def save_checkpoint(trainer: Any) -> None:
    try:
        try:
            _dir = os.path.dirname(C.LOG.CHECKPOINT_FILE_PATH)
            if _dir:
                os.makedirs(_dir, exist_ok=True)
        except Exception:
            pass
        base = getattr(trainer.model, "_orig_mod", trainer.model)
        ckpt = {
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
            "ema": (trainer.ema.shadow if trainer.ema is not None else None),
            "gate": {
                "active": bool(trainer._gate_active),
                "w": int(trainer._gate.w),
                "d": int(trainer._gate.d),
                "losses": int(trainer._gate.losses),
                "z": float(trainer._gate.z),
                "started_iter": int(trainer._gate_started_iter),
                "rounds": int(trainer._gate_rounds),
            },
            "pending_challenger": (
                None
                if (not trainer._gate_active or trainer._pending_challenger is None)
                else trainer._pending_challenger.state_dict()
            ),
            "rng": {
                "py": __import__("random").getstate(),
                "np": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
        }
        tmp_path = f"{C.LOG.CHECKPOINT_FILE_PATH}.tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, C.LOG.CHECKPOINT_FILE_PATH)
        trainer.log.info("[CKPT] saved checkpoint")
    except Exception as e:
        trainer.log.warning(f"Failed to save checkpoint: {e}")


def save_best_model(trainer: Any) -> None:
    try:
        try:
            _dir = os.path.dirname(C.LOG.BEST_MODEL_FILE_PATH)
            if _dir:
                os.makedirs(_dir, exist_ok=True)
        except Exception:
            pass
        state_dict = trainer.best_model.state_dict()
        payload = {
            "iter": int(trainer.iteration),
            "total_games": int(trainer.total_games),
            "model": state_dict,
        }
        tmp_path = f"{C.LOG.BEST_MODEL_FILE_PATH}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, C.LOG.BEST_MODEL_FILE_PATH)
        trainer.log.info("[BEST] saved best model")
    except Exception as e:
        trainer.log.warning(f"Failed to save best model: {e}")


def try_resume(trainer: Any) -> None:
    path = C.LOG.CHECKPOINT_FILE_PATH
    fallback = os.path.basename(path)
    if not os.path.isfile(path) and fallback and os.path.isfile(fallback):
        path = fallback
    if not os.path.isfile(path):
        trainer.log.info("[CKPT] no checkpoint found; starting fresh")
        return
    try:
        ckpt = torch.load(path, map_location="cpu")
        base = getattr(trainer.model, "_orig_mod", trainer.model)
        base.load_state_dict(ckpt.get("model", {}), strict=True)
        if trainer.ema is not None and ckpt.get("ema") is not None:
            trainer.ema.shadow = ckpt["ema"]
        if os.path.isfile(C.LOG.BEST_MODEL_FILE_PATH):
            best_ckpt = torch.load(C.LOG.BEST_MODEL_FILE_PATH, map_location="cpu")
            state_dict = best_ckpt.get("model", best_ckpt)
            trainer.best_model.load_state_dict(state_dict, strict=True)
            trainer.best_model.eval()
            trainer.evaluator.refresh_from(trainer.best_model)
        elif "best_model" in ckpt:
            trainer.best_model.load_state_dict(ckpt["best_model"], strict=True)
            trainer.best_model.eval()
            trainer.evaluator.refresh_from(trainer.best_model)
        if "optimizer" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sd = ckpt["scheduler"]
            trainer.scheduler.set_total_steps(int(sd.get("total", trainer.scheduler.total)))
            trainer.scheduler.t = int(sd.get("t", trainer.scheduler.t))
        if "scaler" in ckpt and isinstance(trainer.scaler, torch.amp.GradScaler):
            trainer.scaler.load_state_dict(ckpt["scaler"])
        trainer.iteration = int(ckpt.get("iter", 0))
        trainer.total_games = int(ckpt.get("total_games", 0))
        elapsed_s = float(ckpt.get("elapsed_s", 0.0))
        if elapsed_s > 0.0:
            trainer.start_time = time.time() - elapsed_s
        g = ckpt.get("gate", None)
        if isinstance(g, dict):
            try:
                trainer._gate.w = int(g.get("w", 0))
                trainer._gate.d = int(g.get("d", 0))
                trainer._gate.losses = int(g.get("losses", 0))
                trainer._gate.z = float(g.get("z", trainer._gate.z))
                trainer._gate_started_iter = int(g.get("started_iter", 0))
                trainer._gate_rounds = int(g.get("rounds", 0))
                trainer._gate_active = bool(g.get("active", False))
            except Exception:
                trainer._gate_active = False
        pc = ckpt.get("pending_challenger", None)
        if trainer._gate_active and pc is not None:
            try:
                _pc_model = trainer._clone_model()
                _pc_model.load_state_dict(pc, strict=True)
                _pc_model.eval()
                trainer._pending_challenger = _pc_model
            except Exception:
                trainer._pending_challenger = None
        rng = ckpt.get("rng", {})
        if "py" in rng:
            __import__("random").setstate(rng["py"])
        if "np" in rng:
            np.random.set_state(rng["np"])
        if "torch_cpu" in rng:
            torch.set_rng_state(rng["torch_cpu"])
        if "torch_cuda" in rng:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        trainer._prev_eval_m = trainer.evaluator.get_metrics()
        if not getattr(trainer, "_prev_eval_m", None):
            trainer._prev_eval_m = {}
        trainer.log.info(f"[CKPT] resumed from {path} @ iter {trainer.iteration}")
    except Exception as e:
        trainer.log.warning(f"[CKPT] failed to resume: {e}")
