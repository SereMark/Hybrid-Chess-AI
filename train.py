from __future__ import annotations

import contextlib
import csv
import ctypes
import gc
import logging
import os
import time
from typing import Any

import numpy as np
import psutil
import torch

import config as C
from arena import EloGater, arena_match
from checkpoint import save_best_model, save_checkpoint, try_resume
from inference import BatchedEvaluator
from network import ChessNet
from optimization import EMA, WarmupCosine, build_optimizer
from reporting import (
    format_gb,
    format_iteration_summary,
    format_time,
    get_mem_info,
    get_sys_info,
    startup_summary,
)
from self_play import SelfPlayEngine
from train_loop import run_training_iteration, train_step as core_train_step


class Trainer:
    def __init__(self, device: str | None = None, resume: bool = False) -> None:
        self.log = logging.getLogger("hybridchess.trainer")
        device_obj = torch.device(device or "cuda")
        self.device = device_obj
        net_any: Any = ChessNet().to(self.device)
        if C.TORCH.MODEL_CHANNELS_LAST:
            self.model = net_any.to(memory_format=torch.channels_last)
        else:
            self.model = net_any

        self.optimizer = build_optimizer(self.model)

        total_expected_train_steps = int(C.TRAIN.TOTAL_ITERATIONS * C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST)
        warmup_steps_clamped = int(max(1, min(C.TRAIN.LR_WARMUP_STEPS, max(1, total_expected_train_steps - 1))))

        self.scheduler = WarmupCosine(
            self.optimizer,
            C.TRAIN.LR_INIT,
            warmup_steps_clamped,
            C.TRAIN.LR_FINAL,
            total_expected_train_steps,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=C.TORCH.AMP_ENABLED)

        self.evaluator = BatchedEvaluator(self.device)
        self._eval_batch_cap = int(C.EVAL.BATCH_SIZE_MAX)
        self._eval_coalesce_ms = int(C.EVAL.COALESCE_MS)
        with contextlib.suppress(Exception):
            self.evaluator.set_batching_params(self._eval_batch_cap, self._eval_coalesce_ms)
        with contextlib.suppress(Exception):
            self.evaluator.set_cache_capacity(int(C.EVAL.CACHE_CAPACITY))
        self.train_batch_size: int = int(C.TRAIN.BATCH_SIZE)
        self._current_eval_cache_cap: int = int(C.EVAL.CACHE_CAPACITY)
        self._current_replay_cap: int = int(C.REPLAY.BUFFER_CAPACITY)
        self._arena_eval_cache_cap: int = int(C.EVAL.ARENA_CACHE_CAPACITY)
        self._oom_cooldown_iters: int = 0

        self._proc = psutil.Process(os.getpid())
        psutil.cpu_percent(0.0)
        self._proc.cpu_percent(0.0)

        self.ema = EMA(self.model, C.TRAIN.EMA_DECAY) if C.TRAIN.EMA_ENABLED else None
        self.best_model = self._clone_model()
        self.evaluator.refresh_from(self.best_model)

        self.selfplay_engine = SelfPlayEngine(self.evaluator)
        self.iteration = 0
        self.total_games = 0
        self.start_time = time.time()
        props = torch.cuda.get_device_properties(self.device)
        self.device_name = props.name
        self.device_total_gb = props.total_memory / 1024**3
        self._prev_eval_m: dict[str, float] = {}

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
            try:
                if C.LOG.METRICS_LOG_CSV_ENABLE and os.path.isfile(C.LOG.METRICS_LOG_CSV_PATH):
                    os.remove(C.LOG.METRICS_LOG_CSV_PATH)
            except Exception:
                pass

    def _startup_summary(self) -> str:
        return startup_summary(self)

    def _clone_model(self) -> torch.nn.Module:
        clone = ChessNet().to(self.device)
        src = getattr(self.model, "_orig_mod", self.model)
        if hasattr(src, "module"):
            src = src.module
        clone.load_state_dict(src.state_dict(), strict=True)
        clone.eval()
        return clone

    def _save_checkpoint(self) -> None:
        save_checkpoint(self)

    def _save_best_model(self) -> None:
        save_best_model(self)

    def _try_resume(self) -> None:
        try_resume(self)

    def train_step(
        self, batch_data: tuple[list[Any], list[np.ndarray], list[np.ndarray], list[np.ndarray]]
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        return core_train_step(self, batch_data)

    def training_iteration(self) -> dict[str, int | float]:
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

            do_eval = (iteration % C.ARENA.EVAL_EVERY_ITERS) == 0 if C.ARENA.EVAL_EVERY_ITERS > 0 else False
            arena_elapsed = 0.0
            arena_w = arena_d = arena_l = 0
            arena_decision = "skipped"
            arena_metrics: dict[str, float] = {}
            if do_eval:
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
                t_ar = time.time()
                assert self._pending_challenger is not None
                _, aw, ad, al = arena_match(
                    self._pending_challenger,
                    self.best_model,
                    device=self.device,
                    eval_cache_cap=self._arena_eval_cache_cap,
                )
                arena_elapsed = time.time() - t_ar

                self._gate.update(aw, ad, al)
                self._gate_rounds += 1
                decision, m = self._gate.decision()
                self.log.info(
                    f"[Arena   ] W/D/L {aw}/{ad}/{al} | n {int(m.get('n', 0))} | p {100.0 * m.get('p', 0):>5.1f}% | elo {m.get('elo', 0):>6.1f} ±{m.get('se_elo', 0):.1f} | decision {decision.upper()} | time {format_time(arena_elapsed)} | age_iter {iteration - self._gate_started_iter} | rounds {self._gate_rounds}"
                )
                arena_w, arena_d, arena_l = int(aw), int(ad), int(al)
                arena_decision = str(decision)
                arena_metrics = {
                    "n": float(m.get("n", 0.0)),
                    "p": float(m.get("p", 0.0)),
                    "lb": float(m.get("lb", 0.0)),
                    "ub": float(m.get("ub", 0.0)),
                    "elo": float(m.get("elo", 0.0)),
                    "se_elo": float(m.get("se_elo", 0.0)),
                }

                if C.LOG.ARENA_LOG_CSV_ENABLE:
                    try:
                        write_header = not os.path.isfile(C.LOG.ARENA_LOG_CSV_PATH)
                        arena_record = {
                            "iter": int(self.iteration),
                            "age_iter": int(self.iteration - self._gate_started_iter),
                            "round": int(self._gate_rounds),
                            "n": int(m.get("n", 0)),
                            "w": int(aw),
                            "d": int(ad),
                            "l": int(al),
                            "p": float(m.get("p", 0.0)),
                            "lb": float(m.get("lb", 0.0)),
                            "ub": float(m.get("ub", 0.0)),
                            "elo": float(m.get("elo", 0.0)),
                            "se_elo": float(m.get("se_elo", 0.0)),
                            "decision": str(decision),
                            "z": float(self._gate.z),
                            "draw_w": float(C.ARENA.GATE_DRAW_WEIGHT),
                            "baseline_p": float(C.ARENA.GATE_BASELINE_P),
                            "deterministic": bool(C.ARENA.DETERMINISTIC),
                            "mcts_sims": int(C.ARENA.MCTS_EVAL_SIMULATIONS),
                        }
                        with open(C.LOG.ARENA_LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                            w = csv.DictWriter(f, fieldnames=list(arena_record.keys()))
                            if write_header:
                                w.writeheader()
                            w.writerow(arena_record)
                    except Exception:
                        pass

                if decision == "accept":
                    assert self._pending_challenger is not None
                    self.best_model.load_state_dict(self._pending_challenger.state_dict(), strict=True)
                    self.best_model.eval()
                    self.evaluator.refresh_from(self.best_model)
                    self._save_checkpoint()
                    self._save_best_model()
                    self._gate_active = False
                    self._pending_challenger = None
                    self._gate_rounds = 0
                elif decision == "reject":
                    self._gate_active = False
                    self._pending_challenger = None
                    self._gate_rounds = 0
            else:
                self.log.info(f"[Arena   ] skipped | games 0 | time {format_time(arena_elapsed)}")

            if self.iteration % C.LOG.CHECKPOINT_SAVE_EVERY_ITERS == 0:
                self._save_checkpoint()

            next_ar = 0
            if C.ARENA.EVAL_EVERY_ITERS > 0:
                k = C.ARENA.EVAL_EVERY_ITERS
                r = self.iteration % k
                next_ar = (k - r) if r != 0 else k
            peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1024**3
            peak_res = torch.cuda.max_memory_reserved(self.device) / 1024**3
            mem_info_summary = get_mem_info(self._proc, self.device, self.device_total_gb)
            try:
                target_lo = 0.60 * self.device_total_gb
                prev_bs = int(self.train_batch_size)
                headroom_gb = max(0.0, float(self.device_total_gb) - float(peak_res))
                if (
                    peak_res < target_lo
                    and headroom_gb >= 8.0
                    and prev_bs < int(C.TRAIN.BATCH_SIZE_MAX)
                    and int(self._oom_cooldown_iters) == 0
                ):
                    self.train_batch_size = int(min(int(C.TRAIN.BATCH_SIZE_MAX), prev_bs + 512))
                elif peak_res > 0.92 * self.device_total_gb and prev_bs > int(C.TRAIN.BATCH_SIZE_MIN):
                    self.train_batch_size = int(max(int(C.TRAIN.BATCH_SIZE_MIN), prev_bs - 1024))
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

            for _line in format_iteration_summary(
                self,
                iter_stats,
                arena_elapsed,
                next_ar,
                peak_alloc,
                peak_res,
                mem_info_summary,
                get_sys_info(self._proc),
            ):
                self.log.info(_line)

            self._prev_eval_m = self.evaluator.get_metrics()

            if C.LOG.METRICS_LOG_CSV_ENABLE:
                try:
                    eval_metrics_now = self.evaluator.get_metrics()
                    try:
                        buf_cap = int(self.selfplay_engine.get_capacity())
                    except Exception:
                        buf_cap = 1
                    sys_info = get_sys_info(self._proc)
                    lr = float(iter_stats.get("learning_rate", 0.0))
                    opt_steps = int(iter_stats.get("optimizer_steps", 0))
                    train_bs = int(self.train_batch_size)
                    train_time = float(iter_stats.get("training_time", 0.0))
                    samples_per_sec = (opt_steps * train_bs) / train_time if train_time > 0 else 0.0
                    write_header = not os.path.isfile(C.LOG.METRICS_LOG_CSV_PATH)
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
                        "lr_sched_t",
                        "lr_sched_total",
                        "sp_games",
                        "sp_white_wins",
                        "sp_draws",
                        "sp_black_wins",
                        "sp_gpm",
                        "sp_mps_k",
                        "sp_avg_len",
                        "sp_new_moves",
                        "selfplay_time",
                        "buffer_size",
                        "buffer_capacity",
                        "buffer_percent",
                        "eval_requests_total",
                        "eval_cache_hits_total",
                        "eval_hit_rate",
                        "eval_batches_total",
                        "eval_positions_total",
                        "eval_batch_size_max",
                        "eval_batch_cap",
                        "eval_coalesce_ms",
                        "arena_ran",
                        "arena_time_s",
                        "arena_w",
                        "arena_d",
                        "arena_l",
                        "arena_n",
                        "arena_p",
                        "arena_lb",
                        "arena_ub",
                        "arena_elo",
                        "arena_se_elo",
                        "arena_decision",
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
                        "lr_sched_t": int(iter_stats.get("lr_sched_t", 0)),
                        "lr_sched_total": int(iter_stats.get("lr_sched_total", 0)),
                        "sp_games": int(iter_stats.get("games", 0)),
                        "sp_white_wins": int(iter_stats.get("white_wins", 0)),
                        "sp_draws": int(iter_stats.get("draws", 0)),
                        "sp_black_wins": int(iter_stats.get("black_wins", 0)),
                        "sp_gpm": float(iter_stats.get("games_per_min", 0.0)),
                        "sp_mps_k": float(iter_stats.get("moves_per_sec", 0.0)) / 1000.0,
                        "sp_avg_len": float(
                            (iter_stats.get("moves", 0) or 0) / max(1, (iter_stats.get("games", 0) or 0))
                        ),
                        "sp_new_moves": int(iter_stats.get("moves", 0)),
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
                        "eval_requests_total": int(eval_metrics_now.get("requests_total", 0)),
                        "eval_cache_hits_total": int(eval_metrics_now.get("cache_hits_total", 0)),
                        "eval_hit_rate": (
                            100.0
                            * float(eval_metrics_now.get("cache_hits_total", 0))
                            / max(1, float(eval_metrics_now.get("requests_total", 0)))
                        ),
                        "eval_batches_total": int(eval_metrics_now.get("batches_total", 0)),
                        "eval_positions_total": int(eval_metrics_now.get("eval_positions_total", 0)),
                        "eval_batch_size_max": int(eval_metrics_now.get("batch_size_max", 0)),
                        "eval_batch_cap": int(self._eval_batch_cap),
                        "eval_coalesce_ms": int(self._eval_coalesce_ms),
                        "arena_ran": (1 if do_eval else 0),
                        "arena_time_s": float(arena_elapsed),
                        "arena_w": int(arena_w),
                        "arena_d": int(arena_d),
                        "arena_l": int(arena_l),
                        "arena_n": int(arena_metrics.get("n", 0.0)),
                        "arena_p": float(arena_metrics.get("p", 0.0)),
                        "arena_lb": float(arena_metrics.get("lb", 0.0)),
                        "arena_ub": float(arena_metrics.get("ub", 0.0)),
                        "arena_elo": float(arena_metrics.get("elo", 0.0)),
                        "arena_se_elo": float(arena_metrics.get("se_elo", 0.0)),
                        "arena_decision": str(arena_decision),
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
                    with open(C.LOG.METRICS_LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        if write_header:
                            w.writeheader()
                        w.writerow(row)
                except Exception:
                    pass


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
    stdout_handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(stdout_handler)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this training pipeline.")
    torch.set_float32_matmul_precision(C.TORCH.MATMUL_FLOAT32_PRECISION)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = bool(C.TORCH.CUDNN_BENCHMARK and (C.SEED == 0))
    if C.SEED != 0:
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(C.TORCH.THREADS_INTRA)
    torch.set_num_interop_threads(C.TORCH.THREADS_INTER)
    import random as _py_random

    if C.SEED != 0:
        _py_random.seed(C.SEED)
        np.random.seed(C.SEED)
        torch.manual_seed(C.SEED)
        torch.cuda.manual_seed_all(C.SEED)
    resume_flag = any(a in ("--resume", "resume") for a in sys.argv[1:])
    Trainer(resume=resume_flag).train()
