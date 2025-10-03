from __future__ import annotations

import ctypes
import gc
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import psutil
import torch
from torch.amp import GradScaler

import config as C
from arena import EloGater, arena_match
from checkpoint import save_best_model, save_checkpoint, try_resume
from inference import BatchedEvaluator
from metrics import MetricsReporter
from network import ChessNet
from optimization import EMA, WarmupCosine, build_optimizer
from pipeline import PipelineSettings, load_settings
from reporting import (
    format_gb,
    get_mem_info,
    get_sys_info,
    startup_summary,
)
from self_play import SelfPlayEngine
from train_loop import run_training_iteration, train_step as core_train_step


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    base = getattr(module, "_orig_mod", module)
    if hasattr(base, "module"):
        base = base.module
    return cast(torch.nn.Module, base)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class ArenaResult:
    ran: bool = False
    elapsed: float = 0.0
    w: int = 0
    d: int = 0
    losses: int = 0
    score_pct: float = 0.0
    draw_pct: float = 0.0
    decisive_pct: float = 0.0
    decision: str = "skipped"
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def games_played(self) -> int:
        return self.w + self.d + self.losses


class Trainer:
    def __init__(self, device: str | None = None, resume: bool = False) -> None:
        self.log = logging.getLogger("hybridchess.trainer")
        if device:
            device_obj = torch.device(device)
        else:
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
            device_obj = torch.device(default_device)
            if device_obj.type != "cuda":
                self.log.warning("CUDA unavailable; falling back to CPU for training")
        self.device = device_obj
        self.settings: PipelineSettings = load_settings()
        self.metrics = MetricsReporter(self.settings.logging.csv_path)
        net_any: Any = ChessNet().to(self.device)
        if C.TORCH.MODEL_CHANNELS_LAST:
            self.model = net_any.to(memory_format=torch.channels_last)
        else:
            self.model = net_any

        self.optimizer = build_optimizer(self.model)

        total_expected_train_steps = int(C.TRAIN.TOTAL_ITERATIONS * C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST)
        warmup_steps_clamped = int(max(1, min(C.TRAIN.LR_WARMUP_STEPS, max(1, total_expected_train_steps - 1))))
        restart_interval_iters = int(max(0, getattr(C.TRAIN, "LR_RESTART_INTERVAL_ITERS", 0)))
        if restart_interval_iters > 0:
            steps_per_iter_est = float(max(1.0, float(C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST)))
            restart_interval_steps = int(max(1, round(restart_interval_iters * steps_per_iter_est)))
        else:
            restart_interval_steps = 0

        self.scheduler = WarmupCosine(
            self.optimizer,
            C.TRAIN.LR_INIT,
            warmup_steps_clamped,
            C.TRAIN.LR_FINAL,
            total_expected_train_steps,
            restart_interval=restart_interval_steps,
            restart_decay=getattr(C.TRAIN, "LR_RESTART_DECAY", 1.0),
        )
        self._amp_enabled = bool(C.TORCH.AMP_ENABLED and self.device.type == "cuda")
        self._scaler_device_type = (
            self.device.type if self.device.type in {"cuda", "cpu", "mps"} else "cuda"
        )
        self.scaler = GradScaler(self._scaler_device_type, enabled=self._amp_enabled)

        self.evaluator = BatchedEvaluator(self.device)
        self._eval_batch_cap = int(C.EVAL.BATCH_SIZE_MAX)
        self._eval_coalesce_ms = int(C.EVAL.COALESCE_MS)
        try:
            self.evaluator.set_batching_params(self._eval_batch_cap, self._eval_coalesce_ms)
        except Exception as exc:
            self.log.debug("Failed to apply evaluator batching params: %s", exc, exc_info=True)
        try:
            self.evaluator.set_cache_capacity(int(C.EVAL.CACHE_CAPACITY))
        except Exception as exc:
            self.log.debug("Failed to apply evaluator cache capacity: %s", exc, exc_info=True)
        self.train_batch_size: int = int(C.TRAIN.BATCH_SIZE)
        if self.device.type != "cuda":
            self.train_batch_size = int(min(self.train_batch_size, 128))
        self._current_eval_cache_cap: int = int(C.EVAL.CACHE_CAPACITY)
        self._current_replay_cap: int = int(C.REPLAY.BUFFER_CAPACITY)
        self._arena_eval_cache_cap: int = int(C.EVAL.ARENA_CACHE_CAPACITY)
        self._oom_cooldown_iters: int = 0
        ratio_max = float(getattr(C.SAMPLING, "TRAIN_RECENT_SAMPLE_RATIO_MAX", C.SAMPLING.TRAIN_RECENT_SAMPLE_RATIO))
        ratio_min = float(getattr(C.SAMPLING, "TRAIN_RECENT_SAMPLE_RATIO_MIN", ratio_max))
        self._recent_sample_ratio: float = float(np.clip(ratio_max, ratio_min, ratio_max))

        self._proc = psutil.Process(os.getpid())
        psutil.cpu_percent(0.0)
        self._proc.cpu_percent(0.0)

        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            self.device_name = props.name
            self.device_total_gb = props.total_memory / 1024**3
        else:
            self.device_name = str(self.device)
            self.device_total_gb = psutil.virtual_memory().total / 1024**3

        self.ema = EMA(self.model, C.TRAIN.EMA_DECAY) if C.TRAIN.EMA_ENABLED else None
        self.best_model = self._clone_model()
        self.selfplay_model = self._clone_model()
        self._sync_selfplay_evaluator()

        self.selfplay_engine = SelfPlayEngine(self.evaluator)
        try:
            self.selfplay_engine.update_adjudication(0)
        except Exception as exc:
            self.log.debug("Self-play adjudication priming failed: %s", exc, exc_info=True)
        try:
            self.selfplay_engine.set_game_length(self.settings.selfplay.game_max_plies)
        except Exception as exc:
            self.log.debug("Unable to set self-play game length: %s", exc, exc_info=True)
        self.iteration = 0
        self.total_games = 0
        self.start_time = time.time()
        self._prev_eval_m: dict[str, float] = {}
        self._color_window: deque[tuple[int, int]] = deque(
            maxlen=int(getattr(C.SELFPLAY, "COLOR_BALANCE_WINDOW_ITERS", 8))
        )
        self._recent_len_window: deque[float] = deque(
            maxlen=max(1, int(getattr(C.RESIGN, "GUARD_WINDOW_ITERS", 32)))
        )
        self._resign_cooldown = 0
        self._resign_threshold = float(getattr(C.RESIGN, "VALUE_THRESHOLD_INIT", C.RESIGN.VALUE_THRESHOLD))
        self._resign_min_plies = int(max(0, getattr(C.RESIGN, "MIN_PLIES_INIT", 0)))
        self._resign_eval_losses: deque[float] = deque(
            maxlen=int(max(32, getattr(C.RESIGN, "TARGET_WINDOW_GAMES", 320)))
        )
        self._resign_target_pct = float(
            np.clip(getattr(C.RESIGN, "TARGET_LOSS_VALUE_PCTL", 0.25), 0.05, 0.5)
        )
        self._grad_skip_count = 0
        self._loop_cooldown = 0

        self._gate = EloGater(
            z=C.ARENA.GATE_Z_EARLY,
            min_games=C.ARENA.GATE_MIN_GAMES,
            draw_w=C.ARENA.GATE_DRAW_WEIGHT,
            baseline_p=C.ARENA.GATE_BASELINE_P,
            decisive_secondary=C.ARENA.GATE_DECISIVE_SECONDARY,
            min_decisive=C.ARENA.GATE_MIN_DECISIVES,
        )
        self._gate_active = False
        self._pending_challenger: torch.nn.Module | None = None
        self._gate_started_iter = 0
        self._gate_rounds = 0

        if resume:
            try_resume(self)
        else:
            if self.settings.logging.enable_csv:
                log_dir = os.path.dirname(self.settings.logging.csv_path)
                if log_dir:
                    try:
                        os.makedirs(log_dir, exist_ok=True)
                    except OSError as exc:
                        self.log.debug("Unable to create metrics directory %s: %s", log_dir, exc, exc_info=True)
                if os.path.isfile(self.settings.logging.csv_path):
                    try:
                        os.remove(self.settings.logging.csv_path)
                    except OSError as exc:
                        self.log.debug(
                            "Unable to remove existing metrics CSV %s: %s",
                            self.settings.logging.csv_path,
                            exc,
                            exc_info=True,
                        )

    def _startup_summary(self) -> str:
        return startup_summary(self)

    def _clone_model(self) -> torch.nn.Module:
        clone = ChessNet().to(self.device)
        src = _unwrap_module(self.model)
        clone.load_state_dict(src.state_dict(), strict=True)
        clone.eval()
        return clone

    def _sync_selfplay_evaluator(self) -> None:
        model_src = getattr(self, "selfplay_model", None)
        if model_src is None:
            return
        if self.ema is not None:
            self.ema.copy_to(model_src)
        else:
            base = _unwrap_module(self.model)
            model_src.load_state_dict(base.state_dict(), strict=True)
        model_src.eval()
        self.evaluator.refresh_from(model_src)

    def _gate_state_metrics(self, expected_games: int, games_played: int) -> dict[str, Any]:
        games_played = max(0, int(games_played))
        expected_games = max(0, int(expected_games))
        total_games = self._gate.w + self._gate.d + self._gate.losses
        return {
            "round": float(self._gate_rounds),
            "age_iter": float(max(0, self.iteration - self._gate_started_iter)),
            "gate_z": float(self._gate.z),
            "gate_w": float(self._gate.w),
            "gate_d": float(self._gate.d),
            "gate_l": float(self._gate.losses),
            "gate_games": float(total_games),
            "gate_min_games": float(self._gate.min_games),
            "gate_min_decisive": float(self._gate.min_decisive),
            "gate_force_decisive": 1.0 if self._gate.force_decisive else 0.0,
            "gate_draw_weight": float(self._gate.draw_w),
            "gate_baseline_p": float(self._gate.baseline_p),
            "gate_baseline_margin": float(self._gate.baseline_margin),
            "gate_started_iter": float(self._gate_started_iter),
            "expected_games": float(expected_games),
            "actual_games": float(games_played),
        }

    def _baseline_arena_metrics(self, expected_games: int) -> dict[str, Any]:
        baseline = {
            "n": 0.0,
            "p": 0.0,
            "lb": 0.0,
            "ub": 0.0,
            "elo": 0.0,
            "se_elo": 0.0,
            "draw_pct": 0.0,
            "decisive_pct": 0.0,
            "score_pct": 0.0,
            "decisive": 0.0,
            "margin_target": 0.0,
            "term_resign": 0.0,
            "term_natural": 0.0,
            "term_exhausted": 0.0,
            "term_threefold": 0.0,
            "term_fifty": 0.0,
            "term_adjudicated": 0.0,
            "term_natural_pct": 0.0,
            "term_adjudicated_pct": 0.0,
            "natural_gate_blocked": 0.0,
            "curriculum_gate_blocked": 0.0,
            "curriculum_games": 0.0,
            "curriculum_wins": 0.0,
            "curriculum_decisive": 0.0,
            "notes": "",
        }
        baseline.update(self._gate_state_metrics(expected_games, 0))
        return baseline

    def _maybe_run_arena(self, iteration: int, expected_games: int, run_eval: bool) -> ArenaResult:
        metrics = self._baseline_arena_metrics(expected_games)
        result = ArenaResult(metrics=dict(metrics))
        if not run_eval:
            return result

        if (
            (not self._gate_active)
            or (self._gate_rounds >= C.ARENA.CANDIDATE_MAX_ROUNDS)
            or ((self._gate.w + self._gate.d + self._gate.losses) >= C.ARENA.CANDIDATE_MAX_GAMES)
        ):
            if self._gate_active:
                self.log.info("[Arena   ] reset: timeboxing stuck challenger")
            self._pending_challenger = self._clone_from_ema()
            self._gate.reset()
            self._gate_active = True
            self._gate_started_iter = iteration
            self._gate_rounds = 0

        assert self._pending_challenger is not None
        start = time.time()
        score_raw, aw, ad, al, arena_details = arena_match(
            self._pending_challenger,
            self.best_model,
            device=self.device,
            eval_cache_cap=self._arena_eval_cache_cap,
            batch_size_cap=self._eval_batch_cap,
        )
        result.elapsed = time.time() - start
        result.ran = True
        result.w = int(aw)
        result.d = int(ad)
        result.losses = int(al)
        games_played = result.games_played
        if games_played != expected_games:
            self.log.warning(
                "[Arena   ] games_played mismatch: got %d expected %d",
                games_played,
                expected_games,
            )

        self._gate.update(result.w, result.d, result.losses)
        self._gate_rounds += 1
        decision, decision_metrics = self._gate.decision()
        result.decision = str(decision)

        total_games = max(1, games_played)
        result.draw_pct = 100.0 * result.d / total_games
        result.decisive_pct = 100.0 * (result.w + result.losses) / total_games
        score_pct = 100.0 * _as_float(decision_metrics.get("p"), float(score_raw))
        result.score_pct = score_pct

        reason_counts = {
            str(k): int(v)
            for k, v in ((arena_details or {}).get("reason_counts", {}) or {}).items()
        }
        curriculum_games = int((arena_details or {}).get("curriculum_games", 0))
        curriculum_wins = int((arena_details or {}).get("curriculum_wins", 0))
        curriculum_decisive = int((arena_details or {}).get("curriculum_decisive", 0))
        term_resign = int(reason_counts.get("resign", 0))
        term_natural = int(reason_counts.get("natural", 0))
        term_exhausted = int(reason_counts.get("exhausted", 0))
        term_threefold = int(reason_counts.get("threefold", 0))
        term_fifty = int(reason_counts.get("fifty_move", 0))
        term_adjudicated = int(reason_counts.get("adjudicated", 0))
        term_loop_forced = int(reason_counts.get("loop", 0))
        term_loop_loss = int(reason_counts.get("loop_loss", 0))
        term_loop = term_loop_forced + term_loop_loss
        total_reason_games = max(1, sum(reason_counts.values()))
        natural_like = term_natural + term_resign + term_threefold + term_fifty + term_loop
        if bool(getattr(C.ARENA, "COUNT_ADJUDICATED_AS_NATURAL", False)):
            natural_like += term_adjudicated
        arena_notes: list[str] = []

        natural_threshold = float(getattr(C.ARENA, "GATE_MIN_NATURAL_PCT", 0.0))
        natural_gate_blocked = False
        natural_ratio = (natural_like / max(1, total_reason_games)) if total_reason_games else 0.0
        if (
            natural_threshold > 0.0
            and natural_ratio < natural_threshold
            and decision in {"accept", "reject"}
        ):
            natural_gate_blocked = True
            decision = "undecided"
            result.decision = "undecided"
            self.log.info(
                "[Arena   ] gate paused: natural term %.1f%% < target %.1f%%",
                100.0 * natural_ratio,
                natural_threshold * 100.0,
            )
            decision_metrics["natural_gate_blocked"] = 1.0
            arena_notes.append("insufficient-natural")

        curriculum_gate_blocked = False
        min_curr_wins = int(getattr(C.ARENA, "CURRICULUM_MIN_WINS", 0))
        min_curr_dec = int(getattr(C.ARENA, "CURRICULUM_MIN_DECISIVES", 0))
        if (
            decision == "accept"
            and (
                (min_curr_wins > 0 and curriculum_wins < min_curr_wins)
                or (min_curr_dec > 0 and curriculum_decisive < min_curr_dec)
            )
        ):
            decision = "undecided"
            result.decision = "undecided"
            curriculum_gate_blocked = True
            self.log.info(
                "[Arena   ] gate paused: curriculum wins %d/%d, decisives %d (needs %d wins, %d decisives)",
                curriculum_wins,
                max(1, curriculum_games),
                curriculum_decisive,
                min_curr_wins,
                min_curr_dec,
            )
            arena_notes.append("curriculum-block")
            decision_metrics["curriculum_gate_blocked"] = 1.0

        if term_loop > 0:
            arena_notes.append("loop-loss")

        margin_raw = decision_metrics.get("margin_target")
        gate_margin = (
            float(margin_raw)
            if margin_raw is not None
            else float(C.ARENA.GATE_BASELINE_P + C.ARENA.GATE_BASELINE_MARGIN)
        )

        metrics.update(
            {
                "n": _as_float(decision_metrics.get("n"), 0.0),
                "p": _as_float(decision_metrics.get("p"), float(score_raw)),
                "lb": _as_float(decision_metrics.get("lb"), 0.0),
                "ub": _as_float(decision_metrics.get("ub"), 0.0),
                "elo": _as_float(decision_metrics.get("elo"), 0.0),
                "se_elo": _as_float(decision_metrics.get("se_elo"), 0.0),
                "draw_pct": result.draw_pct,
                "decisive_pct": result.decisive_pct,
                "score_pct": score_pct,
                "decisive": float(result.w + result.losses),
                "margin_target": gate_margin,
                "term_resign": float(term_resign),
                "term_natural": float(term_natural),
                "term_exhausted": float(term_exhausted),
                "term_threefold": float(term_threefold),
                "term_fifty": float(term_fifty),
                "term_adjudicated": float(term_adjudicated),
                "term_loop": float(term_loop),
                "term_natural_pct": (
                    100.0 * natural_like / total_reason_games if total_reason_games else 0.0
                ),
                "term_adjudicated_pct": (
                    100.0 * term_adjudicated / total_reason_games if total_reason_games else 0.0
                ),
                "term_loop_pct": (
                    100.0 * term_loop / total_reason_games if total_reason_games else 0.0
                ),
                "natural_gate_blocked": 1.0 if natural_gate_blocked else 0.0,
                "curriculum_gate_blocked": 1.0 if curriculum_gate_blocked else 0.0,
                "curriculum_games": float(curriculum_games),
                "curriculum_wins": float(curriculum_wins),
                "curriculum_decisive": float(curriculum_decisive),
            }
        )

        if decision == "accept":
            assert self._pending_challenger is not None
            self.best_model.load_state_dict(self._pending_challenger.state_dict(), strict=True)
            self.best_model.eval()
            self._sync_selfplay_evaluator()
            self._save_checkpoint()
            self._save_best_model()
            self._gate_active = False
            self._pending_challenger = None
            self._gate_rounds = 0
        elif decision == "reject":
            self._gate_active = False
            self._pending_challenger = None
            self._gate_rounds = 0
            self._restore_best_weights()

        metrics.update(self._gate_state_metrics(expected_games, games_played))
        metrics["notes"] = "; ".join(arena_notes)
        result.metrics = metrics
        result.notes = arena_notes
        return result

    def _restore_best_weights(self) -> None:
        base_best = _unwrap_module(self.best_model)
        target = _unwrap_module(self.model)
        target.load_state_dict(base_best.state_dict(), strict=True)
        if self.ema is not None:
            self.ema.shadow = {
                k: v.detach().clone()
                for k, v in base_best.state_dict().items()
            }
        self.optimizer = build_optimizer(self.model)
        self.scheduler.opt = self.optimizer
        current_lr = self.scheduler._lr_at(self.scheduler.t) if hasattr(self.scheduler, "_lr_at") else self.optimizer.param_groups[0]["lr"]
        for pg in self.optimizer.param_groups:
            pg["lr"] = current_lr
        scaler_device_type = getattr(
            self,
            "_scaler_device_type",
            self.device.type if self.device.type in {"cuda", "cpu", "mps"} else "cuda",
        )
        self.scaler = GradScaler(scaler_device_type, enabled=self._amp_enabled)
        self.optimizer.zero_grad(set_to_none=True)
        self._sync_selfplay_evaluator()
        self.log.info("[Arena   ] challenger rejected; restored best weights for continued training")

    def _save_checkpoint(self) -> None:
        save_checkpoint(self)

    def _save_best_model(self) -> None:
        save_best_model(self)

    def _try_resume(self) -> None:
        try_resume(self)

    def _append_csv_row(
        self, path: str, row: dict[str, Any], fieldnames: list[str] | None = None
    ) -> None:
        if not path:
            return
        reporter = self.metrics if path == self.metrics.csv_path else MetricsReporter(path)
        reporter.append(row, fieldnames)

    def train_step(
        self, batch_data: tuple[list[Any], list[np.ndarray], list[np.ndarray], list[np.ndarray]]
    ) -> tuple[torch.Tensor, torch.Tensor, float, float] | None:
        return core_train_step(self, batch_data)

    def training_iteration(self) -> dict[str, int | float | str]:
        return run_training_iteration(self)

    def _clone_from_ema(self) -> torch.nn.Module:
        model_clone = self._clone_model()
        if self.ema is not None:
            self.ema.copy_to(model_clone)
        return model_clone

    def train(self) -> None:
        self.log.info(self._startup_summary())

        for iteration in range(self.iteration + 1, C.TRAIN.TOTAL_ITERATIONS + 1):
            self.iteration = iteration
            iter_stats = self.training_iteration()

            arena_every = max(1, self.settings.arena.eval_every_iters)
            expected_games = int(self.settings.arena.games_per_eval)
            should_eval = self.settings.arena.eval_every_iters > 0 and (iteration % arena_every) == 0
            arena_result = self._maybe_run_arena(iteration, expected_games, should_eval)
            arena_elapsed = arena_result.elapsed
            arena_w, arena_d, arena_l = arena_result.w, arena_result.d, arena_result.losses
            arena_decision = arena_result.decision
            arena_metrics = dict(arena_result.metrics)
            arena_draw_pct = float(arena_result.draw_pct)
            arena_decisive_pct = float(arena_result.decisive_pct)
            score_pct = float(arena_result.score_pct)
            arena_notes = list(arena_result.notes)
            games_played = arena_result.games_played
            arena_metrics.setdefault("notes", "; ".join(arena_notes))

            if self.iteration % C.LOG.CHECKPOINT_SAVE_EVERY_ITERS == 0:
                self._save_checkpoint()

            next_ar = 0
            if self.settings.arena.eval_every_iters > 0:
                k = arena_every
                r = self.iteration % k
                next_ar = (k - r) if r != 0 else k
            if self.device.type == "cuda":
                peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1024**3
                peak_res = torch.cuda.max_memory_reserved(self.device) / 1024**3
            else:
                peak_alloc = 0.0
                peak_res = 0.0
            mem_info_summary = get_mem_info(self._proc, self.device, self.device_total_gb)
            try:
                prev_bs = int(self.train_batch_size)
                alloc_frac = float(peak_alloc) / max(1e-9, float(self.device_total_gb))
                headroom_gb = max(0.0, float(self.device_total_gb) - float(peak_alloc))
                can_increase = (
                    alloc_frac < 0.70
                    and headroom_gb >= 0.5
                    and prev_bs < int(C.TRAIN.BATCH_SIZE_MAX)
                    and int(self._oom_cooldown_iters) == 0
                )
                can_decrease = alloc_frac > 0.90 and prev_bs > int(C.TRAIN.BATCH_SIZE_MIN)
                if can_increase:
                    self.train_batch_size = int(min(int(C.TRAIN.BATCH_SIZE_MAX), prev_bs + 128))
                elif can_decrease:
                    self.train_batch_size = int(max(int(C.TRAIN.BATCH_SIZE_MIN), prev_bs - 256))
                if self.train_batch_size != prev_bs:
                    self.log.info(
                        f"[AUTO    ] train_batch_size {prev_bs} -> {self.train_batch_size} (peak_res {format_gb(peak_res)})"
                    )
                if int(self._oom_cooldown_iters) > 0:
                    self._oom_cooldown_iters = int(self._oom_cooldown_iters) - 1
                else:
                    pass
            except Exception:
                pass
            try:
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

            sys_info = get_sys_info(self._proc)

            sp_games = int(iter_stats.get("games", 0))
            sp_w = int(iter_stats.get("white_wins", 0))
            sp_b = int(iter_stats.get("black_wins", 0))
            sp_d = int(iter_stats.get("draws", 0))
            sp_len = float(iter_stats.get("sp_avg_len", 0.0))
            sp_time = float(iter_stats.get("selfplay_time", 0.0))
            sp_games_per_min = float(iter_stats.get("games_per_min", 0.0))
            sp_moves_per_sec = float(iter_stats.get("moves_per_sec", 0.0))
            sp_mps_k = sp_moves_per_sec / 1000.0
            sp_curriculum_games = int(iter_stats.get("sp_curriculum_games", 0))
            sp_curriculum_pct = _as_float(iter_stats.get("sp_curriculum_pct"), 0.0)
            sp_term_nat_pct = float(iter_stats.get("sp_term_natural_pct", 0.0))
            sp_term_adj_pct = float(iter_stats.get("sp_term_adjudicated_pct", 0.0))
            sp_term_resign_pct = float(iter_stats.get("sp_term_resign_pct", 0.0))
            sp_term_loop_pct = float(iter_stats.get("sp_term_loop_pct", 0.0))
            sp_loop_pct = float(iter_stats.get("sp_loop_pct", 0.0))
            sp_loop_flag_pct = float(iter_stats.get("sp_loop_flag_pct", 0.0))
            sp_loop_unique_pct = float(iter_stats.get("sp_loop_unique_ratio_pct", 0.0))
            sp_loop_bias_avg = float(iter_stats.get("sp_loop_bias_avg", 0.0))
            sp_eval_mean = float(iter_stats.get("sp_terminal_eval_mean", 0.0))
            sp_eval_abs = float(iter_stats.get("sp_terminal_eval_abs_mean", 0.0))
            sp_tempo_bonus = float(iter_stats.get("sp_tempo_bonus_avg", 0.0))
            sp_loss_eval_mean = float(iter_stats.get("sp_loss_eval_mean", 0.0))
            sp_loss_eval_p25 = float(iter_stats.get("sp_loss_eval_p25", 0.0))
            sp_value_trend = float(iter_stats.get("sp_value_trend_hit_pct", 0.0))
            sp_sims = float(iter_stats.get("sp_sims_avg", 0.0))
            resign_status_str = str(iter_stats.get("resign_status", "n/a"))
            resign_threshold = _as_float(iter_stats.get("resign_threshold"), C.RESIGN.VALUE_THRESHOLD)
            resign_min_plies = int(iter_stats.get("resign_min_plies", getattr(C.RESIGN, "MIN_PLIES_FINAL", 0)))
            resign_cooldown = int(iter_stats.get("resign_cooldown_iters", 0))
            replay_reset_flag = int(iter_stats.get("replay_reset", 0))

            train_steps = int(iter_stats.get("train_steps_actual", 0))
            train_plan = int(iter_stats.get("train_steps_planned", 0))
            loss_p = float(iter_stats.get("policy_loss", 0.0))
            loss_v = float(iter_stats.get("value_loss", 0.0))
            lr_now = float(iter_stats.get("learning_rate", 0.0))
            grad = float(iter_stats.get("avg_grad_norm", 0.0))
            grad_skips = int(iter_stats.get("grad_skip_count", 0))
            entropy = float(iter_stats.get("avg_entropy", 0.0))
            ent_coef = float(iter_stats.get("entropy_coef", 0.0))
            samples_ratio = float(iter_stats.get("samples_ratio", 0.0))
            batches_per_sec = float(iter_stats.get("batches_per_sec", 0.0))
            samples_per_sec = float(iter_stats.get("samples_per_sec", 0.0))
            train_time = float(iter_stats.get("train_time_s", 0.0))
            buffer_pct = float(iter_stats.get("buffer_percent", 0.0))
            train_recent_pct = float(iter_stats.get("train_recent_pct", 0.0))
            lr_sched_drift_pct = float(iter_stats.get("lr_sched_drift_pct", 0.0))

            eval_hit = float(iter_stats.get("eval_hit_rate", 0.0))
            eval_avg_batch = float(iter_stats.get("eval_avg_batch", 0.0))
            ram_used = float(sys_info.get("ram_used_gb", 0.0))
            ram_total = float(sys_info.get("ram_total_gb", 0.0))
            ram_pct = float(sys_info.get("ram_pct", 0.0))

            arena_termN = float(arena_metrics.get("term_natural", 0.0))
            arena_termA = float(arena_metrics.get("term_adjudicated", 0.0))
            arena_notes_str = arena_metrics.get("notes", "")
            arena_draw_pct = float(arena_metrics.get("draw_pct", 0.0))
            arena_decisive_pct = float(arena_metrics.get("decisive_pct", 0.0))

            sp_summary = (
                f"[Iter {iteration:03d}/{C.TRAIN.TOTAL_ITERATIONS}] "
                f"SP  games={sp_games} W/D/L={sp_w}/{sp_d}/{sp_b} len={sp_len:.1f} "
                f"gpm={sp_games_per_min:.1f} mps={sp_mps_k:.2f}k"
            )
            sp_loop_line = (
                f"           loop={sp_loop_pct:.0f}% flag={sp_loop_flag_pct:.0f}% uniq={sp_loop_unique_pct:.0f}% bias={sp_loop_bias_avg:+.2f} "
                f"nat/adj/res={sp_term_nat_pct:.0f}/{sp_term_adj_pct:.0f}/{sp_term_resign_pct:.0f}% loopTerm={sp_term_loop_pct:.0f}% eval={sp_eval_mean:.2f}±{sp_eval_abs:.2f} "
                f"sims={sp_sims:.0f} tempo={sp_tempo_bonus:.3f} lossEval={sp_loss_eval_mean:.2f}/{sp_loss_eval_p25:.2f} trend={sp_value_trend:.1f}% "
                f"curr={sp_curriculum_games}({sp_curriculum_pct:.0f}%) time={sp_time:.1f}s"
            )
            trn_line = (
                f"           TRN steps={train_steps}/{train_plan} lossP={loss_p:.3f} lossV={loss_v:.3f} lr={lr_now:.2e} grad={grad:.2f} ent={entropy:.2f} entc={ent_coef:.2e} "
                f"sps={samples_per_sec:.0f} b/s={batches_per_sec:.1f} buf={buffer_pct:.0f}% recent={train_recent_pct:.0f}% mix={samples_ratio:.1f} skips={grad_skips} "
                f"res={resign_status_str} thr={resign_threshold:.2f}@{resign_min_plies} reset={replay_reset_flag}"
            )
            if arena_result.ran:
                arena_n = max(1, games_played)
                arena_draw_pct_val = 100.0 * arena_d / arena_n if arena_n else 0.0
                arena_decisive_pct_val = 100.0 * (arena_w + arena_l) / arena_n if arena_n else 0.0
                arena_line = (
                    f"           ARENA {arena_decision.upper():<7} score={score_pct:.1f}% W/D/L={arena_w}/{arena_d}/{arena_l} "
                    f"draw={arena_draw_pct_val:.1f}% dec={arena_decisive_pct_val:.1f}% nat={arena_termN:.0f} adj={arena_termA:.0f} "
                    f"time={arena_elapsed:.1f}s"
                )
                if arena_notes_str:
                    arena_line += f" note={arena_notes_str}"
            else:
                arena_line = f"           ARENA skipped (next in {next_ar})"
            sys_line = (
                f"           SYS  gpu={peak_alloc:.1f}/{peak_res:.1f}G ram={ram_used:.1f}/{ram_total:.1f}G ({ram_pct:.0f}%) "
                f"eval={eval_hit:.1f}% batch={eval_avg_batch:.1f}/{self._eval_batch_cap} sched={lr_sched_drift_pct:+.1f}%"
            )

            self.log.info(sp_summary)
            self.log.info(sp_loop_line)
            self.log.info(trn_line)
            self.log.info(arena_line)
            self.log.info(sys_line)
            self.log.info("")

            self._prev_eval_m = self.evaluator.get_metrics()

            if self.settings.logging.enable_csv:
                try:
                    try:
                        log_dir = os.path.dirname(self.settings.logging.csv_path)
                        if log_dir:
                            os.makedirs(log_dir, exist_ok=True)
                    except Exception:
                        pass
                    eval_metrics_now = self.evaluator.get_metrics()
                    try:
                        buf_cap = int(self.selfplay_engine.get_capacity())
                    except Exception:
                        buf_cap = 1
                    sys_info = get_sys_info(self._proc)
                    lr = float(iter_stats.get("learning_rate", 0.0))
                    opt_steps = int(iter_stats.get("optimizer_steps", 0))
                    train_bs = int(self.train_batch_size)
                    train_time = float(iter_stats.get("train_time_s", 0.0))
                    samples_per_sec = float(iter_stats.get("samples_per_sec", 0.0))
                    if samples_per_sec == 0.0 and train_time > 0:
                        samples_per_sec = (opt_steps * train_bs) / train_time
                    fieldnames = [
                        "iter",
                        "elapsed_s",
                        "next_ar",
                        "train_batch_size",
                        "optimizer_steps",
                        "learning_rate",
                        "policy_loss",
                        "value_loss",
                        "batches_per_sec",
                        "samples_per_sec",
                        "train_time_s",
                        "train_steps_planned",
                        "train_steps_actual",
                        "train_samples",
                        "samples_ratio",
                        "avg_grad_norm",
                        "avg_entropy",
                        "entropy_coef",
                        "lr_sched_drift_pct",
                        "train_recent_pct",
                        "lr_sched_t",
                        "lr_sched_total",
                        "sp_games",
                        "sp_white_wins",
                        "sp_white_win_pct",
                        "sp_draws",
                        "sp_draws_true",
                        "sp_draws_cap",
                        "sp_draw_pct",
                        "sp_draw_true_pct",
                        "sp_draw_cap_pct",
                        "sp_black_wins",
                        "sp_black_win_pct",
                        "sp_decisive_pct",
                        "sp_gpm",
                        "sp_mps_k",
                        "sp_avg_len",
                        "sp_new_moves",
                        "sp_white_starts",
                        "sp_black_starts",
                        "sp_white_start_pct",
                        "sp_black_start_pct",
                        "sp_color_bias_black_pct",
                        "sp_color_bias_target_pct",
                        "sp_terminal_eval_mean",
                        "sp_terminal_eval_abs_mean",
                        "sp_tempo_bonus_avg",
                        "sp_sims_avg",
                        "sp_unique_positions_avg",
                        "sp_unique_ratio_pct",
                        "sp_halfmove_avg",
                        "sp_halfmove_max",
                        "sp_threefold_pct",
                        "sp_threefold_games",
                        "sp_threefold_draws",
                        "sp_loop_games",
                        "sp_loop_alerts",
                        "sp_loop_pct",
                        "sp_loop_flag_games",
                        "sp_loop_flag_pct",
                        "sp_loop_unique_ratio_pct",
                        "sp_loop_bias_avg",
                        "sp_value_trend_hit_pct",
                        "sp_loss_eval_mean",
                        "sp_loss_eval_p05",
                        "sp_loss_eval_p25",
                        "resign_status",
                        "resign_threshold",
                        "resign_min_plies",
                        "resign_cooldown_iters",
                        "replay_reset",
                        "sp_adjudicate_margin",
                        "sp_adjudicate_min_plies",
                        "sp_term_natural",
                        "sp_term_exhausted",
                        "sp_term_resign",
                        "sp_term_threefold",
                        "sp_term_fifty",
                        "sp_term_adjudicated",
                        "sp_term_loop",
                        "sp_term_natural_pct",
                        "sp_term_exhausted_pct",
                        "sp_term_resign_pct",
                        "sp_term_threefold_pct",
                        "sp_term_fifty_pct",
                        "sp_term_adjudicated_pct",
                        "sp_term_loop_pct",
                        "sp_adjudicated_games",
                        "sp_adjudicated_white",
                        "sp_adjudicated_black",
                        "sp_adjudicated_draw",
                        "sp_adjudicated_pct",
                        "sp_adjudicated_white_pct",
                        "sp_adjudicated_black_pct",
                        "sp_adjudicated_draw_pct",
                        "sp_adjudicated_balance_mean",
                        "sp_adjudicated_balance_abs_mean",
                        "sp_adjudicated_balance_total",
                        "sp_adjudicated_balance_abs_total",
                        "selfplay_time",
                        "buffer_size",
                        "buffer_capacity",
                        "buffer_percent",
                        "eval_requests_total",
                        "eval_cache_hits_total",
                        "eval_requests_delta",
                        "eval_cache_hits_delta",
                        "eval_hit_rate",
                        "eval_hit_rate_delta",
                        "eval_batches_total",
                        "eval_batches_delta",
                        "eval_positions_total",
                        "eval_positions_delta",
                        "eval_batch_size_max",
                        "eval_batch_cap",
                        "eval_coalesce_ms",
                        "eval_avg_batch",
                        "arena_ran",
                        "arena_time_s",
                        "arena_w",
                        "arena_d",
                        "arena_l",
                        "arena_n",
                        "arena_p",
                        "arena_score_pct",
                        "arena_draw_pct",
                        "arena_decisive_pct",
                        "arena_decisive_games",
                        "arena_margin_target",
                        "arena_term_resign",
                        "arena_term_natural",
                        "arena_term_exhausted",
                        "arena_term_threefold",
                        "arena_term_fifty",
                        "arena_term_adjudicated",
                        "arena_term_loop",
                        "arena_term_natural_pct",
                        "arena_term_adjudicated_pct",
                        "arena_term_loop_pct",
                        "arena_natural_gate_blocked",
                        "arena_lb",
                        "arena_ub",
                        "arena_elo",
                        "arena_se_elo",
                        "arena_decision",
                        "arena_round",
                        "arena_age_iter",
                        "arena_expected_games",
                        "arena_actual_games",
                        "arena_gate_z",
                        "arena_gate_w",
                        "arena_gate_d",
                        "arena_gate_l",
                        "arena_gate_games",
                        "arena_gate_min_games",
                        "arena_gate_min_decisive",
                        "arena_gate_force_decisive",
                        "arena_gate_draw_weight",
                        "arena_gate_baseline_p",
                        "arena_gate_baseline_margin",
                        "arena_gate_started_iter",
                        "arena_notes",
                        "gate_rounds",
                        "gpu_peak_alloc_gb",
                        "gpu_peak_reserved_gb",
                        "gpu_allocated_gb",
                        "gpu_reserved_gb",
                        "gpu_total_gb",
                        "rss_gb",
                        "ram_used_gb",
                        "ram_total_gb",
                        "ram_pct",
                        "cpu_sys_pct",
                        "cpu_proc_pct",
                        "load1",
                    ]
                    arena_n = max(1, int(arena_metrics.get("n", 0.0)))
                    arena_draw_pct = (arena_d / arena_n * 100.0) if arena_n > 0 else 0.0
                    arena_decisive_pct = ((arena_w + arena_l) / arena_n * 100.0) if arena_n > 0 else 0.0

                    row = {
                        "iter": int(self.iteration),
                        "elapsed_s": float(time.time() - self.start_time),
                        "next_ar": int(next_ar),
                        "train_batch_size": train_bs,
                        "optimizer_steps": opt_steps,
                        "learning_rate": lr,
                        "policy_loss": float(iter_stats.get("policy_loss", 0.0)),
                        "value_loss": float(iter_stats.get("value_loss", 0.0)),
                        "batches_per_sec": float(iter_stats.get("batches_per_sec", 0.0)),
                        "samples_per_sec": float(samples_per_sec),
                        "train_time_s": train_time,
                        "train_steps_planned": int(iter_stats.get("train_steps_planned", 0)),
                        "train_steps_actual": int(iter_stats.get("train_steps_actual", 0)),
                        "train_samples": int(iter_stats.get("train_samples", opt_steps * train_bs)),
                        "samples_ratio": float(iter_stats.get("samples_ratio", 0.0)),
                        "avg_grad_norm": float(iter_stats.get("avg_grad_norm", 0.0)),
                        "avg_entropy": float(iter_stats.get("avg_entropy", 0.0)),
                        "entropy_coef": float(iter_stats.get("entropy_coef", 0.0)),
                        "lr_sched_drift_pct": float(iter_stats.get("lr_sched_drift_pct", 0.0)),
                        "train_recent_pct": float(iter_stats.get("train_recent_pct", 0.0)),
                        "lr_sched_t": int(iter_stats.get("lr_sched_t", 0)),
                        "lr_sched_total": int(iter_stats.get("lr_sched_total", 0)),
                        "sp_games": int(iter_stats.get("games", 0)),
                        "sp_white_wins": int(iter_stats.get("white_wins", 0)),
                        "sp_white_win_pct": float(iter_stats.get("sp_white_win_pct", 0.0)),
                        "sp_draws": int(iter_stats.get("draws", 0)),
                        "sp_draws_true": int(iter_stats.get("draws_true", 0)),
                        "sp_draws_cap": int(iter_stats.get("draws_cap", 0)),
                        "sp_draw_pct": float(iter_stats.get("sp_draw_pct", 0.0)),
                        "sp_draw_true_pct": float(iter_stats.get("sp_draw_true_pct", 0.0)),
                        "sp_draw_cap_pct": float(iter_stats.get("sp_draw_cap_pct", 0.0)),
                        "sp_black_wins": int(iter_stats.get("black_wins", 0)),
                        "sp_black_win_pct": float(iter_stats.get("sp_black_win_pct", 0.0)),
                        "sp_decisive_pct": (
                            100.0
                            * (int(iter_stats.get("white_wins", 0)) + int(iter_stats.get("black_wins", 0)))
                            / max(1, int(iter_stats.get("games", 0)))
                        ),
                        "sp_gpm": float(iter_stats.get("games_per_min", 0.0)),
                        "sp_mps_k": float(iter_stats.get("moves_per_sec", 0.0)) / 1000.0,
                        "sp_avg_len": sp_len,
                        "sp_new_moves": int(iter_stats.get("sp_new_moves", iter_stats.get("moves", 0))),
                        "sp_white_starts": int(iter_stats.get("sp_white_starts", 0)),
                        "sp_black_starts": int(iter_stats.get("sp_black_starts", 0)),
                        "sp_white_start_pct": float(iter_stats.get("sp_white_start_pct", 0.0)),
                        "sp_black_start_pct": float(iter_stats.get("sp_black_start_pct", 0.0)),
                        "sp_color_bias_black_pct": float(iter_stats.get("sp_color_bias_black_pct", 0.0)),
                        "sp_color_bias_target_pct": float(iter_stats.get("sp_color_bias_target_pct", 0.0)),
                        "sp_terminal_eval_mean": float(iter_stats.get("sp_terminal_eval_mean", 0.0)),
                        "sp_terminal_eval_abs_mean": float(
                            iter_stats.get("sp_terminal_eval_abs_mean", 0.0)
                        ),
                        "sp_tempo_bonus_avg": float(iter_stats.get("sp_tempo_bonus_avg", 0.0)),
                        "sp_sims_avg": float(iter_stats.get("sp_sims_avg", 0.0)),
                        "sp_unique_positions_avg": float(iter_stats.get("sp_unique_positions_avg", 0.0)),
                        "sp_unique_ratio_pct": float(iter_stats.get("sp_unique_ratio_pct", 0.0)),
                        "sp_halfmove_avg": float(iter_stats.get("sp_halfmove_avg", 0.0)),
                        "sp_halfmove_max": int(iter_stats.get("sp_halfmove_max", 0)),
                        "sp_threefold_pct": float(iter_stats.get("sp_threefold_pct", 0.0)),
                        "sp_threefold_games": int(iter_stats.get("sp_threefold_games", 0)),
                        "sp_threefold_draws": int(iter_stats.get("sp_threefold_draws", 0)),
                        "sp_loop_games": int(iter_stats.get("sp_loop_games", 0)),
                        "sp_loop_alerts": int(iter_stats.get("sp_loop_alerts", 0)),
                        "sp_loop_pct": float(iter_stats.get("sp_loop_pct", 0.0)),
                        "sp_loop_flag_games": int(iter_stats.get("sp_loop_flag_games", 0)),
                        "sp_loop_flag_pct": float(iter_stats.get("sp_loop_flag_pct", 0.0)),
                        "sp_loop_unique_ratio_pct": float(iter_stats.get("sp_loop_unique_ratio_pct", 0.0)),
                        "sp_loop_bias_avg": float(iter_stats.get("sp_loop_bias_avg", 0.0)),
                        "sp_value_trend_hit_pct": float(
                            iter_stats.get("sp_value_trend_hit_pct", 0.0)
                        ),
                        "sp_loss_eval_mean": float(iter_stats.get("sp_loss_eval_mean", 0.0)),
                        "sp_loss_eval_p05": float(iter_stats.get("sp_loss_eval_p05", 0.0)),
                        "sp_loss_eval_p25": float(iter_stats.get("sp_loss_eval_p25", 0.0)),
                        "resign_status": resign_status_str,
                        "resign_threshold": resign_threshold,
                        "resign_min_plies": resign_min_plies,
                        "resign_cooldown_iters": resign_cooldown,
                        "replay_reset": replay_reset_flag,
                        "sp_adjudicate_margin": float(iter_stats.get("sp_adjudicate_margin", 0.0)),
                        "sp_adjudicate_min_plies": int(iter_stats.get("sp_adjudicate_min_plies", 0)),
                        "sp_term_natural": int(iter_stats.get("sp_term_natural", 0)),
                        "sp_term_exhausted": int(iter_stats.get("sp_term_exhausted", 0)),
                        "sp_term_resign": int(iter_stats.get("sp_term_resign", 0)),
                        "sp_term_threefold": int(iter_stats.get("sp_term_threefold", 0)),
                        "sp_term_fifty": int(iter_stats.get("sp_term_fifty", 0)),
                        "sp_term_adjudicated": int(iter_stats.get("sp_term_adjudicated", 0)),
                        "sp_term_loop": int(iter_stats.get("sp_term_loop", 0)),
                        "sp_term_natural_pct": float(iter_stats.get("sp_term_natural_pct", 0.0)),
                        "sp_term_exhausted_pct": float(iter_stats.get("sp_term_exhausted_pct", 0.0)),
                        "sp_term_resign_pct": float(iter_stats.get("sp_term_resign_pct", 0.0)),
                        "sp_term_threefold_pct": float(iter_stats.get("sp_term_threefold_pct", 0.0)),
                        "sp_term_fifty_pct": float(iter_stats.get("sp_term_fifty_pct", 0.0)),
                        "sp_term_adjudicated_pct": float(iter_stats.get("sp_term_adjudicated_pct", 0.0)),
                        "sp_term_loop_pct": float(iter_stats.get("sp_term_loop_pct", 0.0)),
                        "sp_adjudicated_games": int(iter_stats.get("sp_adjudicated_games", 0)),
                        "sp_adjudicated_white": int(iter_stats.get("sp_adjudicated_white", 0)),
                        "sp_adjudicated_black": int(iter_stats.get("sp_adjudicated_black", 0)),
                        "sp_adjudicated_draw": int(iter_stats.get("sp_adjudicated_draw", 0)),
                        "sp_adjudicated_pct": float(iter_stats.get("sp_adjudicated_pct", 0.0)),
                        "sp_adjudicated_white_pct": float(iter_stats.get("sp_adjudicated_white_pct", 0.0)),
                        "sp_adjudicated_black_pct": float(iter_stats.get("sp_adjudicated_black_pct", 0.0)),
                        "sp_adjudicated_draw_pct": float(iter_stats.get("sp_adjudicated_draw_pct", 0.0)),
                        "sp_adjudicated_balance_mean": float(
                            iter_stats.get("sp_adjudicated_balance_mean", 0.0)
                        ),
                        "sp_adjudicated_balance_abs_mean": float(
                            iter_stats.get("sp_adjudicated_balance_abs_mean", 0.0)
                        ),
                        "sp_adjudicated_balance_total": float(
                            iter_stats.get("sp_adjudicated_balance_total", 0.0)
                        ),
                        "sp_adjudicated_balance_abs_total": float(
                            iter_stats.get("sp_adjudicated_balance_abs_total", 0.0)
                        ),
                        "selfplay_time": float(iter_stats.get("selfplay_time", 0.0)),
                        "buffer_size": int(
                            iter_stats.get("buffer_size", getattr(self.selfplay_engine, "size", lambda: 0)())
                        ),
                        "buffer_capacity": int(buf_cap),
                        "buffer_percent": float(
                            iter_stats.get(
                                "buffer_percent",
                                100.0 * float(getattr(self.selfplay_engine, "size", lambda: 0)()) / max(1, buf_cap),
                            )
                        ),
                        "eval_requests_total": int(iter_stats.get("eval_requests_total", eval_metrics_now.get("requests_total", 0))),
                        "eval_cache_hits_total": int(iter_stats.get("eval_cache_hits_total", eval_metrics_now.get("cache_hits_total", 0))),
                        "eval_requests_delta": int(iter_stats.get("eval_requests_delta", 0)),
                        "eval_cache_hits_delta": int(iter_stats.get("eval_cache_hits_delta", 0)),
                        "eval_hit_rate": float(iter_stats.get("eval_hit_rate", 0.0)),
                        "eval_hit_rate_delta": float(iter_stats.get("eval_hit_rate_delta", 0.0)),
                        "eval_batches_total": int(iter_stats.get("eval_batches_total", eval_metrics_now.get("batches_total", 0))),
                        "eval_batches_delta": int(iter_stats.get("eval_batches_delta", 0)),
                        "eval_positions_total": int(iter_stats.get("eval_positions_total", eval_metrics_now.get("eval_positions_total", 0))),
                        "eval_positions_delta": int(iter_stats.get("eval_positions_delta", 0)),
                        "eval_batch_size_max": int(iter_stats.get("eval_batch_size_max", eval_metrics_now.get("batch_size_max", 0))),
                        "eval_batch_cap": int(iter_stats.get("eval_batch_cap", self._eval_batch_cap)),
                        "eval_coalesce_ms": int(iter_stats.get("eval_coalesce_ms", self._eval_coalesce_ms)),
                        "eval_avg_batch": float(iter_stats.get("eval_avg_batch", 0.0)),
                        "arena_ran": (1 if arena_result.ran else 0),
                        "arena_time_s": float(arena_elapsed),
                        "arena_w": int(arena_w),
                        "arena_d": int(arena_d),
                        "arena_l": int(arena_l),
                        "arena_n": int(_as_float(arena_metrics.get("n"), 0.0)),
                        "arena_p": _as_float(arena_metrics.get("p"), 0.0),
                        "arena_score_pct": float(score_pct),
                        "arena_draw_pct": arena_draw_pct,
                        "arena_decisive_pct": arena_decisive_pct,
                        "arena_decisive_games": int(_as_float(arena_metrics.get("decisive"), 0.0)),
                        "arena_margin_target": _as_float(
                            arena_metrics.get("margin_target"), C.ARENA.GATE_BASELINE_P
                        ),
                        "arena_term_resign": _as_float(arena_metrics.get("term_resign"), 0.0),
                        "arena_term_natural": _as_float(arena_metrics.get("term_natural"), 0.0),
                        "arena_term_exhausted": _as_float(arena_metrics.get("term_exhausted"), 0.0),
                        "arena_term_threefold": _as_float(arena_metrics.get("term_threefold"), 0.0),
                        "arena_term_fifty": _as_float(arena_metrics.get("term_fifty"), 0.0),
                        "arena_term_adjudicated": _as_float(arena_metrics.get("term_adjudicated"), 0.0),
                        "arena_term_loop": _as_float(arena_metrics.get("term_loop"), 0.0),
                        "arena_term_natural_pct": _as_float(arena_metrics.get("term_natural_pct"), 0.0),
                        "arena_term_adjudicated_pct": _as_float(
                            arena_metrics.get("term_adjudicated_pct"), 0.0
                        ),
                        "arena_term_loop_pct": _as_float(arena_metrics.get("term_loop_pct"), 0.0),
                        "arena_natural_gate_blocked": _as_float(
                            arena_metrics.get("natural_gate_blocked"), 0.0
                        ),
                        "arena_lb": _as_float(arena_metrics.get("lb"), 0.0),
                        "arena_ub": _as_float(arena_metrics.get("ub"), 0.0),
                        "arena_elo": _as_float(arena_metrics.get("elo"), 0.0),
                        "arena_se_elo": _as_float(arena_metrics.get("se_elo"), 0.0),
                        "arena_decision": str(arena_decision),
                        "arena_round": _as_float(arena_metrics.get("round"), 0.0),
                        "arena_age_iter": _as_float(arena_metrics.get("age_iter"), 0.0),
                        "arena_expected_games": _as_float(arena_metrics.get("expected_games"), float(expected_games)),
                        "arena_actual_games": _as_float(arena_metrics.get("actual_games"), float(games_played)),
                        "arena_gate_z": _as_float(arena_metrics.get("gate_z"), self._gate.z),
                        "arena_gate_w": _as_float(arena_metrics.get("gate_w"), self._gate.w),
                        "arena_gate_d": _as_float(arena_metrics.get("gate_d"), self._gate.d),
                        "arena_gate_l": _as_float(arena_metrics.get("gate_l"), self._gate.losses),
                        "arena_gate_games": _as_float(
                            arena_metrics.get("gate_games"), self._gate.w + self._gate.d + self._gate.losses
                        ),
                        "arena_gate_min_games": _as_float(
                            arena_metrics.get("gate_min_games"), self._gate.min_games
                        ),
                        "arena_gate_min_decisive": _as_float(
                            arena_metrics.get("gate_min_decisive"), self._gate.min_decisive
                        ),
                        "arena_gate_force_decisive": _as_float(
                            arena_metrics.get("gate_force_decisive"), 1.0 if self._gate.force_decisive else 0.0
                        ),
                        "arena_gate_draw_weight": _as_float(
                            arena_metrics.get("gate_draw_weight"), self._gate.draw_w
                        ),
                        "arena_gate_baseline_p": _as_float(
                            arena_metrics.get("gate_baseline_p"), self._gate.baseline_p
                        ),
                        "arena_gate_baseline_margin": _as_float(
                            arena_metrics.get("gate_baseline_margin"), self._gate.baseline_margin
                        ),
                        "arena_gate_started_iter": _as_float(
                            arena_metrics.get("gate_started_iter"), self._gate_started_iter
                        ),
                        "arena_notes": arena_notes_str,
                        "gate_rounds": int(self._gate_rounds),
                        "gpu_peak_alloc_gb": float(peak_alloc),
                        "gpu_peak_reserved_gb": float(peak_res),
                        "gpu_allocated_gb": float(mem_info_summary.get("allocated_gb", 0.0)),
                        "gpu_reserved_gb": float(mem_info_summary.get("reserved_gb", 0.0)),
                        "gpu_total_gb": float(mem_info_summary.get("total_gb", 0.0)),
                        "rss_gb": float(mem_info_summary.get("rss_gb", 0.0)),
                        "ram_used_gb": float(sys_info.get("ram_used_gb", 0.0)),
                        "ram_total_gb": float(sys_info.get("ram_total_gb", 0.0)),
                        "ram_pct": float(sys_info.get("ram_pct", 0.0)),
                        "cpu_sys_pct": float(sys_info.get("cpu_sys_pct", 0.0)),
                        "cpu_proc_pct": float(sys_info.get("cpu_proc_pct", 0.0)),
                        "load1": float(sys_info.get("load1", 0.0)),
                    }
                    self._append_csv_row(self.settings.logging.csv_path, row, fieldnames)
                except Exception as exc:
                    self.log.debug("Failed to append metrics row: %s", exc, exc_info=True)


if __name__ == "__main__":
    import sys

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(2 * 1024 * 1024 * 1024))
    if C.SEED != 0:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")

    root = logging.getLogger()
    log_level = getattr(logging, str(C.LOG.LEVEL).upper(), logging.INFO)
    root.setLevel(log_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(stdout_handler)

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.set_float32_matmul_precision(C.TORCH.MATMUL_FLOAT32_PRECISION)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    else:
        logging.getLogger("hybridchess.trainer").warning("CUDA unavailable; running training in CPU mode")
    if C.SEED != 0:
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(C.TORCH.THREADS_INTRA)
    torch.set_num_interop_threads(C.TORCH.THREADS_INTER)
    import random as _py_random

    if C.SEED != 0:
        _py_random.seed(C.SEED)
        np.random.seed(C.SEED)
        torch.manual_seed(C.SEED)
        if has_cuda:
            torch.cuda.manual_seed_all(C.SEED)
    resume_flag = any(a in ("--resume", "resume") for a in sys.argv[1:])
    Trainer(resume=resume_flag).train()
