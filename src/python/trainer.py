"""Training orchestration for Hybrid Chess AI."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import config as C
import numpy as np
import torch
from arena import ArenaResult, play_match
from checkpoint import get_run_root, save_best_model, save_checkpoint, try_resume
from inference import BatchedEvaluator
from network import ChessNet
from optimization import EMA, WarmupCosine, build_optimizer
from self_play import SelfPlayEngine
from torch.amp import GradScaler
from train_loop import run_training_iteration
from utils import (
    MetricsReporter,
    format_time,
    prepare_model,
    select_autocast_dtype,
    startup_summary,
)

__all__ = ["Trainer"]


@dataclass(slots=True)
class GateHistory:
    """Lightweight record keeping for arena evaluations."""

    last: ArenaResult | None = None
    accepted: int = 0
    rejected: int = 0
    notes: list[str] = field(default_factory=list)


class Trainer:
    """Coordinates neural network training, self-play, evaluation, and gating."""

    def __init__(self, device: str | None = None, resume: bool = False) -> None:
        self.log = logging.getLogger("hybridchess.trainer")
        self.device = self._resolve_device(device)
        self.device_name = self._device_name()
        self.model = prepare_model(
            ChessNet(),
            self.device,
            channels_last=C.TORCH.model_channels_last,
        )
        self.optimizer = build_optimizer(self.model)

        total_steps = int(C.TRAIN.total_iterations * C.TRAIN.lr_steps_per_iter_estimate)
        warmup_steps = int(min(max(1, C.TRAIN.learning_rate_warmup_steps), max(1, total_steps - 1)))
        restart_steps = 0
        if C.TRAIN.lr_restart_interval_iters > 0:
            restart_steps = int(max(1, C.TRAIN.lr_restart_interval_iters * C.TRAIN.lr_steps_per_iter_estimate))

        self.scheduler = WarmupCosine(
            self.optimizer,
            C.TRAIN.learning_rate_init,
            warmup_steps,
            C.TRAIN.learning_rate_final,
            total_steps,
            restart_interval=restart_steps,
            restart_decay=C.TRAIN.lr_restart_decay,
        )

        self._amp_enabled = bool(C.TORCH.amp_enabled and self.device.type == "cuda")
        self._autocast_dtype = select_autocast_dtype(self.device)
        scaler: GradScaler | None = None
        try:
            scaler = GradScaler(enabled=self._amp_enabled)
        except TypeError:
            scaler = GradScaler(enabled=self._amp_enabled)
        self.scaler = scaler

        self.evaluator = BatchedEvaluator(self.device)
        self.evaluator.set_batching_params(
            batch_size_max=C.EVAL.batch_size_max,
            coalesce_ms=C.EVAL.coalesce_ms,
        )
        self.evaluator.set_cache_capacity(
            C.EVAL.cache_capacity,
            value_capacity=C.EVAL.value_cache_capacity,
            encode_capacity=C.EVAL.encode_cache_capacity,
        )

        self.selfplay_engine = SelfPlayEngine(self.evaluator)

        initial_root = Path(get_run_root(self))
        self._configure_run_paths(initial_root)
        if resume:
            try_resume(self)
            resumed_root = Path(get_run_root(self))
            if resumed_root != initial_root:
                self._configure_run_paths(resumed_root)
        self.metrics.append_json(
            {
                "event": "startup",
                "timestamp": time.time(),
                "run_root": self.run_root,
            }
        )

        self.ema: EMA | None = EMA(self.model) if C.TRAIN.ema_enabled else None
        self.best_model = self._clone_model()
        self.best_model.load_state_dict(self.model.state_dict(), strict=True)
        self.best_model.eval()

        self.iteration = 0
        self.total_games = 0
        self.start_time = time.time()
        self.gate = GateHistory()
        self._arena_rng = np.random.default_rng()

        self._sync_selfplay_evaluator()

        base_batch = int(C.TRAIN.batch_size)
        self.train_batch_size = int(np.clip(base_batch, C.TRAIN.batch_size_min, C.TRAIN.batch_size_max))
        if self.device.type != "cuda":
            self.train_batch_size = int(C.TRAIN.batch_size_min)

        self.log.info(startup_summary(self))

    # ------------------------------------------------------------------ public API
    def train(self) -> None:
        max_iters = int(C.TRAIN.total_iterations)
        while self.iteration < max_iters:
            stats = run_training_iteration(self)
            self.iteration += 1
            sp_raw = stats.get("selfplay_stats")
            if isinstance(sp_raw, dict):
                self.total_games += int(float(sp_raw.get("games", 0)))
            self._sync_selfplay_evaluator()
            arena_result = None
            if self._should_run_arena():
                arena_result = self._run_arena()
            self._log_iteration(stats, arena_result)
            self._write_metrics(stats, arena_result)
            self._maybe_checkpoint()
            self._maybe_flush_cuda_cache()
        self.log.info("Training completed after %s iterations.", self.iteration)

    # ------------------------------------------------------------------ helpers
    def _resolve_device(self, override: str | None) -> torch.device:
        if override:
            device = torch.device(override)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            self.log.warning("CUDA unavailable; training will run on CPU")
        return device

    def _device_name(self) -> str:
        if self.device.type != "cuda":
            return "CPU"
        try:
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            return torch.cuda.get_device_name(index)
        except Exception:
            return "cuda"

    def _clone_model(self) -> torch.nn.Module:
        model = ChessNet()
        model.to(self.device)
        model.eval()
        return model

    def _sync_selfplay_evaluator(self) -> None:
        model = self.model
        if self.ema is not None:
            shadow = self._clone_model()
            shadow.load_state_dict(self.ema.shadow, strict=True)
            model = shadow
        self.evaluator.refresh_from(model)
        self.evaluator.set_cache_capacity(
            C.EVAL.cache_capacity,
            value_capacity=C.EVAL.value_cache_capacity,
            encode_capacity=C.EVAL.encode_cache_capacity,
        )

    def _candidate_eval_model(self) -> torch.nn.Module:
        model = self._clone_model()
        if self.ema is not None:
            model.load_state_dict(self.ema.shadow, strict=True)
        else:
            model.load_state_dict(self.model.state_dict(), strict=True)
        model.eval()
        return model

    def _configure_run_paths(self, root: Path) -> None:
        resolved = Path(root).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        self.run_root = resolved
        checkpoints_dir = resolved / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = resolved / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        arena_dir = resolved / "arena_games"
        arena_dir.mkdir(parents=True, exist_ok=True)
        self.arena_dir = arena_dir
        metrics_jsonl = metrics_dir / "training.jsonl"
        metrics_csv = metrics_dir / "training.csv"
        self.metrics = MetricsReporter(str(metrics_csv), jsonl_path=str(metrics_jsonl))

    def _run_arena(self) -> ArenaResult:
        candidate_model = self._candidate_eval_model()
        baseline_model = self._clone_model()
        baseline_model.load_state_dict(self.best_model.state_dict(), strict=True)
        baseline_model.eval()

        candidate_eval = BatchedEvaluator(self.device)
        candidate_eval.refresh_from(candidate_model)
        candidate_eval.set_batching_params(
            batch_size_max=C.EVAL.batch_size_max,
            coalesce_ms=C.EVAL.coalesce_ms,
        )
        candidate_eval.set_cache_capacity(
            min(256, C.EVAL.cache_capacity),
            value_capacity=min(8_000, C.EVAL.value_cache_capacity),
            encode_capacity=min(8_000, C.EVAL.encode_cache_capacity),
        )
        candidate_eval.clear_caches()

        baseline_eval = BatchedEvaluator(self.device)
        baseline_eval.refresh_from(baseline_model)
        baseline_eval.set_batching_params(
            batch_size_max=C.EVAL.batch_size_max,
            coalesce_ms=C.EVAL.coalesce_ms,
        )
        baseline_eval.set_cache_capacity(
            min(256, C.EVAL.cache_capacity),
            value_capacity=min(8_000, C.EVAL.value_cache_capacity),
            encode_capacity=min(8_000, C.EVAL.encode_cache_capacity),
        )
        baseline_eval.clear_caches()

        pgn_dir = self.arena_dir
        label = f"iter{self.iteration:04d}"
        result = play_match(
            candidate_eval,
            baseline_eval,
            games=C.ARENA.games_per_eval,
            seed=int(self._arena_rng.integers(0, np.iinfo(np.int64).max)),
            start_fen_fn=self.selfplay_engine.sample_start_fen,
            pgn_dir=pgn_dir,
            label=label,
        )

        decisive_games = result.candidate_wins + result.baseline_wins
        min_games = int(C.ARENA.gate_min_games)
        min_decisive = int(C.ARENA.gate_min_decisive)
        required = C.ARENA.gate_baseline_p + C.ARENA.gate_margin
        score = result.score_pct / 100.0
        accept = result.games >= max(1, min_games) and decisive_games >= max(1, min_decisive) and score >= required

        if accept:
            self.best_model.load_state_dict(candidate_model.state_dict(), strict=True)
            save_best_model(self)
            decision = "accept"
            self.gate.accepted += 1
        else:
            decision = "reject"
            self.gate.rejected += 1
        result.notes.append(decision)
        self.gate.last = result
        candidate_eval.close()
        baseline_eval.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.gate.last

    def _should_run_arena(self) -> bool:
        every = int(max(1, C.ARENA.eval_every_iters))
        if every <= 0:
            return False
        return self.iteration % every == 0

    def _maybe_checkpoint(self) -> None:
        if C.LOG.checkpoint_interval_iters <= 0:
            return
        if self.iteration % C.LOG.checkpoint_interval_iters == 0:
            save_checkpoint(self)

    def _maybe_flush_cuda_cache(self) -> None:
        if self.device.type != "cuda":
            return
        interval = int(C.LOG.empty_cache_interval_iters)
        if interval > 0 and self.iteration % interval == 0:
            torch.cuda.empty_cache()

    def _log_iteration(
        self,
        stats: dict[str, float | int | str],
        arena_result: ArenaResult | None,
    ) -> None:
        sp_raw = stats.get("selfplay_stats")
        sp = cast(dict[str, float | int | str], sp_raw if isinstance(sp_raw, dict) else {})
        games = max(1, int(float(sp.get("games", 0))))
        sp_w = int(float(sp.get("white_wins", 0)))
        sp_b = int(float(sp.get("black_wins", 0)))
        sp_d = int(float(sp.get("draws", 0)))
        natural_pct = 100.0 * int(float(sp.get("term_natural", 0))) / games
        adjudicated_pct = 100.0 * int(float(sp.get("term_adjudicated", 0))) / games
        resigned_pct = 100.0 * int(float(sp.get("term_resign", 0))) / games
        exhausted_pct = 100.0 * int(float(sp.get("term_exhausted", 0))) / games
        visit_per_move = float(sp.get("visit_per_move", 0.0))

        train_line = (
            f"TRN iter={self.iteration}/{C.TRAIN.total_iterations} "
            f"steps={stats['train_steps_actual']}/{stats['train_steps_planned']} "
            f"lossP={stats['policy_loss']:.3f} lossV={stats['value_loss']:.3f} "
            f"entropy={stats['entropy']:.3f} lr={stats['learning_rate']:.2e} "
            f"grad={stats['avg_grad_norm']:.2f} samples/s={stats['samples_per_sec']:.0f} "
            f"buffer={stats['buffer_percent']:.0f}% recent={stats['train_recent_pct']:.0f}% "
            f"resign={stats['resign_status']}"
        )
        sp_line = (
            f"SP  games={games} W/D/L={sp_w}/{sp_d}/{sp_b} "
            f"natural={natural_pct:.1f}% adjudicated={adjudicated_pct:.1f}% "
            f"resign={resigned_pct:.1f}% exhausted={exhausted_pct:.1f}% "
            f"visits/move={visit_per_move:.1f}"
        )
        arena_line = "ARENA skipped"
        if arena_result is not None:
            arena_line = (
                f"ARENA {arena_result.notes[0].upper():<6} "
                f"score={arena_result.score_pct:.1f}% "
                f"W/D/L={arena_result.candidate_wins}/{arena_result.draws}/{arena_result.baseline_wins} "
                f"draw={arena_result.draw_pct:.1f}% decisive={arena_result.decisive_pct:.1f}% "
                f"time={format_time(arena_result.elapsed_s)}"
            )

        self.log.info(sp_line)
        self.log.info(train_line)
        self.log.info(arena_line)
        self.log.info("")

    def _write_metrics(
        self,
        stats: dict[str, float | int | str],
        arena_result: ArenaResult | None,
    ) -> None:
        if not C.LOG.metrics_csv_enable or not self.metrics.csv_path:
            return
        elapsed = time.time() - self.start_time
        sp_raw = stats.get("selfplay_stats")
        sp = cast(dict[str, float | int | str], sp_raw if isinstance(sp_raw, dict) else {})
        games = max(1, int(float(sp.get("games", 0))))
        row = {
            "iter": int(self.iteration),
            "elapsed_s": float(elapsed),
            "optimizer_steps": int(stats["optimizer_steps"]),
            "train_steps_planned": int(stats["train_steps_planned"]),
            "train_steps_actual": int(stats["train_steps_actual"]),
            "train_samples": int(stats["train_samples"]),
            "train_time_s": float(stats["train_time_s"]),
            "selfplay_time_s": float(stats["selfplay_time_s"]),
            "policy_loss": float(stats["policy_loss"]),
            "policy_loss_std": float(stats["policy_loss_std"]),
            "value_loss": float(stats["value_loss"]),
            "value_loss_std": float(stats["value_loss_std"]),
            "entropy": float(stats["entropy"]),
            "entropy_std": float(stats["entropy_std"]),
            "entropy_coef": float(stats["entropy_coef"]),
            "avg_grad_norm": float(stats["avg_grad_norm"]),
            "grad_norm_std": float(stats["grad_norm_std"]),
            "learning_rate": float(stats["learning_rate"]),
            "samples_per_sec": float(stats["samples_per_sec"]),
            "train_recent_pct": float(stats["train_recent_pct"]),
            "buffer_percent": float(stats["buffer_percent"]),
            "buffer_size": int(stats["buffer_size"]),
            "buffer_capacity": int(stats["buffer_capacity"]),
            "sp_games": int(float(sp.get("games", 0))),
            "sp_white_wins": int(float(sp.get("white_wins", 0))),
            "sp_black_wins": int(float(sp.get("black_wins", 0))),
            "sp_draws": int(float(sp.get("draws", 0))),
            "sp_visit_per_move": float(sp.get("visit_per_move", 0.0)),
            "sp_term_natural_pct": 100.0 * int(float(sp.get("term_natural", 0))) / games,
            "sp_term_adjudicated_pct": 100.0 * int(float(sp.get("term_adjudicated", 0))) / games,
            "sp_term_resign_pct": 100.0 * int(float(sp.get("term_resign", 0))) / games,
            "sp_term_exhausted_pct": 100.0 * int(float(sp.get("term_exhausted", 0))) / games,
            "arena_score_pct": float(arena_result.score_pct) if arena_result else 0.0,
            "arena_draw_pct": float(arena_result.draw_pct) if arena_result else 0.0,
            "arena_decisive_pct": float(arena_result.decisive_pct) if arena_result else 0.0,
            "arena_elapsed_s": float(arena_result.elapsed_s) if arena_result else 0.0,
            "arena_decision": arena_result.notes[0] if arena_result else "skipped",
        }
        field_order = list(row.keys())
        self.metrics.append(row, field_order=field_order)
