from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from .config import (
    AMP_ENABLED,
    AMP_PREFER_BFLOAT16,
    ARENA_CANDIDATE_MAX_GAMES,
    ARENA_CANDIDATE_MAX_ROUNDS,
    ARENA_DETERMINISTIC,
    ARENA_DIRICHLET_WEIGHT,
    ARENA_DRAW_SCORE,
    ARENA_EVAL_CACHE_CAPACITY,
    ARENA_EVAL_EVERY_ITERS,
    ARENA_GAMES_PER_EVAL,
    ARENA_GATE_BASELINE_P,
    ARENA_GATE_DECISIVE_SECONDARY,
    ARENA_GATE_DRAW_WEIGHT,
    ARENA_GATE_EPS,
    ARENA_GATE_MIN_DECISIVES,
    ARENA_GATE_MIN_GAMES,
    ARENA_GATE_Z_EARLY,
    ARENA_GATE_Z_LATE,
    ARENA_GATE_Z_SWITCH_ITER,
    ARENA_OPENING_TEMPERATURE_EPS,
    ARENA_PAIRING_FACTOR,
    ARENA_STRATIFY_OPENINGS,
    ARENA_TEMP_MOVES,
    ARENA_TEMPERATURE,
    AUGMENT_MIRROR_PROB,
    AUGMENT_ROT180_PROB,
    AUGMENT_VFLIP_CS_PROB,
    BEST_MODEL_FILE_PATH,
    CHECKPOINT_FILE_PATH,
    CHECKPOINT_SAVE_EVERY_ITERS,
    CUDA_EMPTY_CACHE_EVERY_ITERS,
    EMA_DECAY,
    EMA_ENABLED,
    EVAL_BATCH_COALESCE_MS,
    EVAL_BATCH_SIZE_MAX,
    EVAL_CACHE_CAPACITY,
    GAME_MAX_PLIES,
    LOG_FILE_PATH,
    LOG_LEVEL,
    LOG_TO_FILE,
    LOSS_ENTROPY_ANNEAL_ITERS,
    LOSS_ENTROPY_COEF_INIT,
    LOSS_ENTROPY_COEF_MIN,
    LOSS_POLICY_LABEL_SMOOTH,
    LOSS_POLICY_WEIGHT,
    LOSS_VALUE_WEIGHT,
    LOSS_VALUE_WEIGHT_LATE,
    LOSS_VALUE_WEIGHT_SWITCH_ITER,
    MCTS_C_PUCT,
    MCTS_C_PUCT_BASE,
    MCTS_C_PUCT_INIT,
    MCTS_DIRICHLET_ALPHA,
    MCTS_DIRICHLET_WEIGHT,
    MCTS_EVAL_SIMULATIONS,
    MCTS_FPU_REDUCTION,
    MCTS_TRAIN_SIMULATIONS_BASE,
    MCTS_TRAIN_SIMULATIONS_MIN,
    MODEL_BLOCKS,
    MODEL_CHANNELS,
    MODEL_CHANNELS_LAST,
    OPENINGS_FILE_PATH,
    REPLAY_BUFFER_CAPACITY,
    RESIGN_CONSECUTIVE_MIN,
    RESIGN_CONSECUTIVE_PLIES,
    RESIGN_VALUE_THRESHOLD,
    SEED,
    SELFPLAY_GAMES_PER_ITER,
    SELFPLAY_NUM_WORKERS,
    SELFPLAY_TEMP_HIGH,
    SELFPLAY_TEMP_LOW,
    SELFPLAY_TEMP_MOVES,
    TORCH_ALLOW_TF32,
    TORCH_COMPILE,
    TORCH_COMPILE_DYNAMIC,
    TORCH_COMPILE_FULLGRAPH,
    TORCH_COMPILE_MODE,
    TORCH_CUDNN_BENCHMARK,
    TORCH_MATMUL_FLOAT32_PRECISION,
    TORCH_THREADS_INTER,
    TORCH_THREADS_INTRA,
    TRAIN_BATCH_SIZE,
    TRAIN_GRAD_CLIP_NORM,
    TRAIN_LR_FINAL,
    TRAIN_LR_INIT,
    TRAIN_LR_SCHED_DRIFT_ADJUST_THRESHOLD,
    TRAIN_LR_SCHED_STEPS_PER_ITER_EST,
    TRAIN_LR_WARMUP_STEPS,
    TRAIN_MOMENTUM,
    TRAIN_PIN_MEMORY,
    TRAIN_RECENT_SAMPLE_RATIO,
    TRAIN_TARGET_TRAIN_SAMPLES_PER_NEW,
    TRAIN_TOTAL_ITERATIONS,
    TRAIN_UPDATE_STEPS_MAX,
    TRAIN_UPDATE_STEPS_MIN,
    TRAIN_WEIGHT_DECAY,
)
from .model import (
    BatchedEvaluator,
    ChessNet,
)
from .selfplay import (
    Augment,
    SelfPlayEngine,
)


class Trainer:
    def __init__(self, device: str | None = None, resume: bool = False) -> None:
        self.log = logging.getLogger("hybridchess.trainer")
        dev = torch.device(device or "cuda")
        self.device = dev
        m_any: Any = ChessNet().to(self.device)
        if MODEL_CHANNELS_LAST:
            self.model = m_any.to(memory_format=torch.channels_last)
        else:
            self.model = m_any
        self._compiled = False
        try:
            if TORCH_COMPILE:
                self.model = torch.compile(
                    self.model,
                    mode=TORCH_COMPILE_MODE,
                    fullgraph=TORCH_COMPILE_FULLGRAPH,
                    dynamic=TORCH_COMPILE_DYNAMIC,
                )
                self._compiled = True
        except Exception as e:
            self._compiled = False
            self.log.warning(f"torch.compile unavailable; running uncompiled ({e})")

        decay: list[torch.nn.Parameter] = []
        nodecay: list[torch.nn.Parameter] = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (nodecay if (n.endswith(".bias") or "bn" in n.lower() or "batchnorm" in n.lower()) else decay).append(p)

        self.optimizer = torch.optim.SGD(
            [
                {"params": decay, "weight_decay": TRAIN_WEIGHT_DECAY},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=TRAIN_LR_INIT,
            momentum=TRAIN_MOMENTUM,
            nesterov=True,
        )

        TOTAL_EXPECTED_TRAIN_STEPS = int(TRAIN_TOTAL_ITERATIONS * TRAIN_LR_SCHED_STEPS_PER_ITER_EST)
        WARMUP_STEPS_CLAMPED = int(max(1, min(TRAIN_LR_WARMUP_STEPS, max(1, TOTAL_EXPECTED_TRAIN_STEPS - 1))))

        class WarmupCosine:
            def __init__(
                self,
                optimizer: Any,
                base_lr: float,
                warmup_steps: int,
                final_lr: float,
                total_steps: int,
            ) -> None:
                self.opt = optimizer
                self.base = base_lr
                self.warm = max(1, int(warmup_steps))
                self.final = final_lr
                self.total = max(1, int(total_steps))
                self.t = 0

            def step(self) -> None:
                self.t += 1
                if self.t <= self.warm:
                    lr = self.base * (self.t / self.warm)
                else:
                    import math

                    progress = min(1.0, (self.t - self.warm) / max(1, self.total - self.warm))
                    lr = self.final + (self.base - self.final) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for pg in self.opt.param_groups:
                    pg["lr"] = lr

            def peek_next_lr(self) -> float:
                t_next = self.t + 1
                if t_next <= self.warm:
                    lr = self.base * (t_next / self.warm)
                else:
                    import math

                    progress = min(1.0, (t_next - self.warm) / max(1, self.total - self.warm))
                    lr = self.final + (self.base - self.final) * 0.5 * (1.0 + math.cos(math.pi * progress))
                return lr

            def set_total_steps(self, total_steps: int) -> None:
                self.total = max(self.t + 1, int(total_steps))
                if self.warm >= self.total:
                    self.warm = max(1, self.total - 1)

        self.scheduler = WarmupCosine(
            self.optimizer,
            TRAIN_LR_INIT,
            WARMUP_STEPS_CLAMPED,
            TRAIN_LR_FINAL,
            TOTAL_EXPECTED_TRAIN_STEPS,
        )
        use_bf16 = AMP_PREFER_BFLOAT16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(AMP_ENABLED and not use_bf16))

        self.evaluator = BatchedEvaluator(self.device)

        self._proc = psutil.Process(os.getpid())
        psutil.cpu_percent(0.0)
        self._proc.cpu_percent(0.0)

        class EMA:
            def __init__(self, model: torch.nn.Module, decay: float = EMA_DECAY) -> None:
                self.decay = decay
                base = getattr(model, "_orig_mod", model)
                if hasattr(base, "module"):
                    base = base.module
                self.shadow = {k: v.detach().clone() for k, v in base.state_dict().items()}

            @torch.no_grad()
            def update(self, model: torch.nn.Module) -> None:
                base = getattr(model, "_orig_mod", model)
                if hasattr(base, "module"):
                    base = base.module
                for k, v in base.state_dict().items():
                    if not torch.is_floating_point(v):
                        self.shadow[k] = v.detach().clone()
                        continue
                    if self.shadow[k].dtype != v.dtype:
                        self.shadow[k] = self.shadow[k].to(dtype=v.dtype)
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

            def copy_to(self, model: torch.nn.Module) -> None:
                base = getattr(model, "_orig_mod", model)
                if hasattr(base, "module"):
                    base = base.module
                base.load_state_dict(self.shadow, strict=True)

        self.ema = EMA(self.model, EMA_DECAY) if EMA_ENABLED else None
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

        self._arena_openings = self._load_openings()
        if ARENA_EVAL_EVERY_ITERS > 0 and ARENA_GAMES_PER_EVAL > 0 and not self._arena_openings:
            raise RuntimeError("No arena openings found.")

        self._gate = EloGater(
            z=ARENA_GATE_Z_EARLY,
            min_games=ARENA_GATE_MIN_GAMES,
            draw_w=ARENA_GATE_DRAW_WEIGHT,
            baseline_p=ARENA_GATE_BASELINE_P,
            decisive_secondary=ARENA_GATE_DECISIVE_SECONDARY,
            min_decisive=ARENA_GATE_MIN_DECISIVES,
        )
        self._gate_active = False
        self._pending_challenger: torch.nn.Module | None = None
        self._gate_started_iter = 0
        self._gate_rounds = 0

        if resume:
            self._try_resume()

    @staticmethod
    def _format_time(s: float) -> str:
        return f"{s:.1f}s" if s < 60 else (f"{s / 60:.1f}m" if s < 3600 else f"{s / 3600:.1f}h")

    @staticmethod
    def _format_si(n: int | float, digits: int = 1) -> str:
        try:
            x = float(n)
        except Exception:
            return str(n)
        sign = "-" if x < 0 else ""
        x = abs(x)
        if x >= 1_000_000_000:
            return f"{sign}{x / 1_000_000_000:.{digits}f}B"
        if x >= 1_000_000:
            return f"{sign}{x / 1_000_000:.{digits}f}M"
        if x >= 1_000:
            return f"{sign}{x / 1_000:.{digits}f}k"
        if digits <= 0:
            return f"{sign}{int(x)}"
        return f"{sign}{x:.{digits}f}"

    @staticmethod
    def _format_gb(x: float, digits: int = 1) -> str:
        try:
            v = float(x)
        except Exception:
            return str(x)
        return f"{v:.{digits}f}G"

    def _get_mem_info(self) -> dict[str, float]:
        p = self._proc
        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1024**3,
            "total_gb": self.device_total_gb,
            "rss_gb": p.memory_info().rss / 1024**3,
        }

    def _get_sys_info(self) -> dict[str, float]:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        try:
            la1, la5, la15 = os.getloadavg()
        except Exception:
            la1 = la5 = la15 = 0.0
        return {
            "cpu_sys_pct": float(psutil.cpu_percent(0.0)),
            "cpu_proc_pct": float(self._proc.cpu_percent(0.0)),
            "ram_used_gb": float(vm.used) / 1024**3,
            "ram_total_gb": float(vm.total) / 1024**3,
            "ram_pct": float(vm.percent),
            "swap_used_gb": float(sm.used) / 1024**3,
            "swap_total_gb": float(sm.total) / 1024**3,
            "swap_pct": float(sm.percent),
            "load1": float(la1),
            "load5": float(la5),
            "load15": float(la15),
        }

    def _clone_model(self) -> torch.nn.Module:
        clone = ChessNet().to(self.device)
        src = getattr(self.model, "_orig_mod", self.model)
        if hasattr(src, "module"):
            src = src.module
        clone.load_state_dict(src.state_dict(), strict=True)
        clone.eval()
        return clone

    def _save_checkpoint(self) -> None:
        try:
            base = getattr(self.model, "_orig_mod", self.model)
            ckpt = {
                "iter": int(self.iteration),
                "total_games": int(self.total_games),
                "elapsed_s": float(max(0.0, time.time() - self.start_time)),
                "model": base.state_dict(),
                "best_model": self.best_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": {
                    "t": int(self.scheduler.t),
                    "total": int(self.scheduler.total),
                },
                "scaler": self.scaler.state_dict(),
                "ema": (self.ema.shadow if self.ema is not None else None),
                "gate": {
                    "active": bool(self._gate_active),
                    "w": int(self._gate.w),
                    "d": int(self._gate.d),
                    "losses": int(self._gate.losses),
                    "z": float(self._gate.z),
                    "started_iter": int(self._gate_started_iter),
                    "rounds": int(self._gate_rounds),
                },
                "pending_challenger": (None if (not self._gate_active or self._pending_challenger is None) else getattr(self._pending_challenger, "state_dict")()),
                "rng": {
                    "py": random.getstate(),
                    "np": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            }
            tmp_path = f"{CHECKPOINT_FILE_PATH}.tmp"
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, CHECKPOINT_FILE_PATH)
            self.log.info("[CKPT] saved checkpoint")
        except Exception as e:
            self.log.warning(f"Failed to save checkpoint: {e}")

    def _save_best_model(self) -> None:
        try:
            sd = self.best_model.state_dict()
            payload = {
                "iter": int(self.iteration),
                "total_games": int(self.total_games),
                "model": sd,
            }
            tmp_path = f"{BEST_MODEL_FILE_PATH}.tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, BEST_MODEL_FILE_PATH)
            self.log.info("[BEST] saved best model")
        except Exception as e:
            self.log.warning(f"Failed to save best model: {e}")

    def _try_resume(self) -> None:
        path = CHECKPOINT_FILE_PATH
        if not os.path.isfile(path):
            self.log.info("[CKPT] no checkpoint found; starting fresh")
            return
        try:
            ckpt = torch.load(path, map_location="cpu")
            base = getattr(self.model, "_orig_mod", self.model)
            base.load_state_dict(ckpt.get("model", {}), strict=True)
            if self.ema is not None and ckpt.get("ema") is not None:
                self.ema.shadow = ckpt["ema"]
            if os.path.isfile(BEST_MODEL_FILE_PATH):
                try:
                    best_ckpt = torch.load(BEST_MODEL_FILE_PATH, map_location="cpu")
                    sd = best_ckpt.get("model", best_ckpt)
                    self.best_model.load_state_dict(sd, strict=True)
                    self.best_model.eval()
                    self.evaluator.refresh_from(self.best_model)
                except Exception:
                    pass
            elif "best_model" in ckpt:
                try:
                    self.best_model.load_state_dict(ckpt["best_model"], strict=True)
                    self.best_model.eval()
                    self.evaluator.refresh_from(self.best_model)
                except Exception:
                    pass
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                sd = ckpt["scheduler"]
                try:
                    self.scheduler.set_total_steps(int(sd.get("total", self.scheduler.total)))
                except Exception:
                    pass
                try:
                    self.scheduler.t = int(sd.get("t", self.scheduler.t))
                except Exception:
                    pass
            if "scaler" in ckpt and isinstance(self.scaler, torch.amp.GradScaler):
                try:
                    self.scaler.load_state_dict(ckpt["scaler"])
                except Exception:
                    pass
            self.iteration = int(ckpt.get("iter", 0))
            self.total_games = int(ckpt.get("total_games", 0))
            elapsed_s = float(ckpt.get("elapsed_s", 0.0))
            if elapsed_s > 0.0:
                self.start_time = time.time() - elapsed_s
            g = ckpt.get("gate", None)
            if isinstance(g, dict):
                try:
                    self._gate.w = int(g.get("w", 0))
                    self._gate.d = int(g.get("d", 0))
                    self._gate.losses = int(g.get("losses", 0))
                    self._gate.z = float(g.get("z", self._gate.z))
                    self._gate_started_iter = int(g.get("started_iter", 0))
                    self._gate_rounds = int(g.get("rounds", 0))
                    self._gate_active = bool(g.get("active", False))
                except Exception:
                    self._gate_active = False
            pc = ckpt.get("pending_challenger", None)
            if self._gate_active and pc is not None:
                try:
                    _pc_model = self._clone_model()
                    _pc_model.load_state_dict(pc, strict=True)
                    _pc_model.eval()
                    self._pending_challenger = _pc_model
                except Exception:
                    self._pending_challenger = None
            rng = ckpt.get("rng", {})
            try:
                if "py" in rng:
                    random.setstate(rng["py"])
                if "np" in rng:
                    np.random.set_state(rng["np"])
                if "torch_cpu" in rng:
                    torch.set_rng_state(rng["torch_cpu"])
                if ("torch_cuda" in rng) and (rng["torch_cuda"] is not None):
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
            except Exception:
                pass
            try:
                self._prev_eval_m = self.evaluator.get_metrics()
            except Exception:
                self._prev_eval_m = {}
            self.log.info(f"[CKPT] resumed from {path} @ iter {self.iteration}")
        except Exception as e:
            self.log.warning(f"[CKPT] failed to resume: {e}")

    def _load_openings(self) -> list[str]:
        seen: set[str] = set()

        def _add_fenish(s: str) -> None:
            s = s.strip()
            if not s or s.startswith("#") or s.startswith(";"):
                return
            s = s.split(";", 1)[0].strip()
            toks = s.split()
            if len(toks) >= 6:
                fen6 = " ".join(toks[:6])
            elif len(toks) >= 4:
                fen6 = " ".join(toks[:4] + ["0", "1"])
            else:
                return
            seen.add(fen6)

        if not os.path.isfile(OPENINGS_FILE_PATH):
            return []

        with open(OPENINGS_FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                _add_fenish(line)

        return sorted(seen)

    def train_step(self, batch_data: tuple[list[Any], list[np.ndarray], list[float]]) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        states, policies, values = batch_data
        x_np = np.stack(states).astype(np.float32, copy=False)
        x_t = torch.from_numpy(x_np)
        if TRAIN_PIN_MEMORY:
            x_t = x_t.pin_memory()
        x = x_t.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        pi_t = torch.from_numpy(np.stack(policies).astype(np.float32))
        if TRAIN_PIN_MEMORY:
            pi_t = pi_t.pin_memory()
        pi_target = pi_t.to(self.device, non_blocking=True)
        v_t = torch.tensor(values, dtype=torch.float32)
        if TRAIN_PIN_MEMORY:
            v_t = v_t.pin_memory()
        v_target = v_t.to(self.device, non_blocking=True)
        self.model.train()
        with torch.autocast(
            device_type="cuda",
            dtype=(torch.bfloat16 if AMP_PREFER_BFLOAT16 else torch.float16),
            enabled=AMP_ENABLED,
        ):
            pi_pred, v_pred = self.model(x)
            v_pred = v_pred.squeeze(-1)
            if LOSS_POLICY_LABEL_SMOOTH > 0.0:
                A = pi_target.shape[1]
                pi_smooth = (1.0 - LOSS_POLICY_LABEL_SMOOTH) * pi_target + (LOSS_POLICY_LABEL_SMOOTH / A)
            else:
                pi_smooth = pi_target
            policy_loss = F.kl_div(F.log_softmax(pi_pred, dim=1), pi_smooth, reduction="batchmean")
            value_loss = F.mse_loss(v_pred, v_target)
            ent_coef = 0.0
            if LOSS_ENTROPY_COEF_INIT > 0 and self.iteration <= LOSS_ENTROPY_ANNEAL_ITERS:
                ent_coef = LOSS_ENTROPY_COEF_INIT * (1.0 - (self.iteration - 1) / max(1, LOSS_ENTROPY_ANNEAL_ITERS))
            if LOSS_ENTROPY_COEF_MIN > 0:
                ent_coef = max(float(ent_coef), float(LOSS_ENTROPY_COEF_MIN))
            entropy = (-(F.softmax(pi_pred, dim=1) * F.log_softmax(pi_pred, dim=1)).sum(dim=1)).mean()
            value_weight_now = LOSS_VALUE_WEIGHT_LATE if self.iteration >= LOSS_VALUE_WEIGHT_SWITCH_ITER else LOSS_VALUE_WEIGHT
            total_loss = LOSS_POLICY_WEIGHT * policy_loss + value_weight_now * value_loss - ent_coef * entropy
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_total_norm_t = torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAIN_GRAD_CLIP_NORM)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)
        return (
            policy_loss.detach(),
            value_loss.detach(),
            float(grad_total_norm_t.detach().cpu()),
            float(entropy.detach().cpu()),
        )

    def training_iteration(self) -> dict[str, int | float]:
        if self.selfplay_engine.resign_consecutive < RESIGN_CONSECUTIVE_MIN:
            self.selfplay_engine.resign_consecutive = RESIGN_CONSECUTIVE_MIN
        if self.iteration == ARENA_GATE_Z_SWITCH_ITER:
            self._gate.z = float(ARENA_GATE_Z_LATE)
        stats: dict[str, int | float] = {}
        torch.cuda.reset_peak_memory_stats(self.device)
        header_lr = float(self.scheduler.peek_next_lr())
        mem = self._get_mem_info()
        sysi = self._get_sys_info()
        buf_len = len(self.selfplay_engine.buffer)
        buf_cap = int(self.selfplay_engine.buffer.maxlen or 1)
        buf_pct = (buf_len / buf_cap) * 100
        total_elapsed = time.time() - self.start_time
        pct_done = 100.0 * (self.iteration - 1) / max(1, TRAIN_TOTAL_ITERATIONS)
        self.log.info(
            f"[ITR {self.iteration:>3}/{TRAIN_TOTAL_ITERATIONS} {pct_done:>4.1f}%] "
            f"LRnext {header_lr:.2e} | t {self._format_time(total_elapsed)} | "
            f"buf {self._format_si(buf_len)}/{self._format_si(self.selfplay_engine.buffer.maxlen or 0)} ({int(buf_pct)}%) | "
            f"GPU {self._format_gb(mem['allocated_gb'])}/{self._format_gb(mem['reserved_gb'])}/{self._format_gb(mem['total_gb'])} | "
            f"RSS {self._format_gb(mem['rss_gb'])} | CPU {sysi['cpu_sys_pct']:.0f}/{sysi['cpu_proc_pct']:.0f}% | "
            f"RAM {self._format_gb(sysi['ram_used_gb'])}/{self._format_gb(sysi['ram_total_gb'])} ({sysi['ram_pct']:.0f}%) | "
            f"load {sysi['load1']:.2f}"
        )

        t0 = time.time()
        game_stats = self.selfplay_engine.play_games(SELFPLAY_GAMES_PER_ITER)
        self.total_games += int(game_stats["games"])
        sp_elapsed = time.time() - t0

        gpm = game_stats["games"] / max(1e-9, sp_elapsed / 60)
        mps = game_stats["moves"] / max(1e-9, sp_elapsed)
        gc = int(game_stats["games"])
        ww = int(game_stats["white_wins"])
        bb = int(game_stats["black_wins"])
        dd = int(game_stats["draws"])
        wpct = 100.0 * ww / max(1, gc)
        dpct = 100.0 * dd / max(1, gc)
        bpct = 100.0 * bb / max(1, gc)
        avg_len = game_stats["moves"] / max(1, gc)

        eval_m = self.evaluator.get_metrics()
        eval_req = int(eval_m.get("requests_total", 0))
        eval_hit = int(eval_m.get("cache_hits_total", 0))
        eval_batches = int(eval_m.get("batches_total", 0))
        eval_evalN = int(eval_m.get("eval_positions_total", 0))
        eval_bmax = int(eval_m.get("batch_size_max", 0))
        prev = getattr(self, "_prev_eval_m", {}) or {}
        d_req = int(eval_req - int(prev.get("requests_total", 0)))
        d_hit = int(eval_hit - int(prev.get("cache_hits_total", 0)))
        d_batches = int(eval_batches - int(prev.get("batches_total", 0)))
        d_evalN = int(eval_evalN - int(prev.get("eval_positions_total", 0)))
        hit_rate = (100.0 * eval_hit / max(1, eval_req)) if eval_req else 0.0
        hit_rate_d = (100.0 * d_hit / max(1, d_req)) if d_req > 0 else 0.0

        sp_line = f"games {gc:,} | W/D/B {ww}/{dd}/{bb} ({wpct:.0f}%/{dpct:.0f}%/{bpct:.0f}%) | len {avg_len:>4.1f} | gpm {gpm:>5.1f} | mps {mps / 1000:>4.1f}k | t {self._format_time(sp_elapsed)} | new {self._format_si(int(game_stats.get('moves', 0)))}"

        stats.update(game_stats)
        stats["selfplay_time"] = sp_elapsed
        stats["games_per_min"] = gpm
        stats["moves_per_sec"] = mps

        t1 = time.time()
        if (CUDA_EMPTY_CACHE_EVERY_ITERS > 0) and (self.iteration % CUDA_EMPTY_CACHE_EVERY_ITERS == 0):
            torch.cuda.empty_cache()

        losses: list[tuple[torch.Tensor, torch.Tensor]] = []
        snap = self.selfplay_engine.snapshot()
        min_samples = max(1, TRAIN_BATCH_SIZE // 2)
        new_examples = int(game_stats.get("moves", 0))
        desired_train_samples = int(TRAIN_TARGET_TRAIN_SAMPLES_PER_NEW * max(1, new_examples))
        steps = int(np.ceil(desired_train_samples / TRAIN_BATCH_SIZE))
        steps = max(TRAIN_UPDATE_STEPS_MIN, min(TRAIN_UPDATE_STEPS_MAX, steps))
        ratio = 0.0
        if len(snap) < min_samples:
            steps = 0
        else:
            ratio = (steps * TRAIN_BATCH_SIZE) / max(1, new_examples)

        grad_norm_running: float = 0.0
        ent_running: float = 0.0
        for _i_step in range(steps):
            batch = self.selfplay_engine.sample_from_snapshot(snap, TRAIN_BATCH_SIZE, recent_ratio=TRAIN_RECENT_SAMPLE_RATIO)
            if not batch:
                continue
            s, p, v = batch
            if np.random.rand() < AUGMENT_MIRROR_PROB:
                s, p, _ = Augment.apply(s, p, "mirror")
            if np.random.rand() < AUGMENT_ROT180_PROB:
                s, p, _ = Augment.apply(s, p, "rot180")
            if np.random.rand() < AUGMENT_VFLIP_CS_PROB:
                s, p, cs = Augment.apply(s, p, "vflip_cs")
                if cs:
                    v = [-val for val in v]
            pol_loss_t, val_loss_t, grad_norm_val, pred_entropy = self.train_step((s, p, v))
            grad_norm_running += float(grad_norm_val)
            ent_running += float(pred_entropy)
            losses.append((pol_loss_t, val_loss_t))

        tr_elapsed = time.time() - t1
        actual_steps = len(losses)
        pol_loss = float(torch.stack([pair[0] for pair in losses]).mean().detach().cpu()) if losses else float("nan")
        val_loss = float(torch.stack([pair[1] for pair in losses]).mean().detach().cpu()) if losses else float("nan")
        buf_sz = len(self.selfplay_engine.buffer)
        buf_cap2 = int(self.selfplay_engine.buffer.maxlen or 1)
        buf_pct2 = (buf_sz / buf_cap2) * 100
        bps = (len(losses) / max(1e-9, tr_elapsed)) if losses else 0.0
        sps = ((len(losses) * TRAIN_BATCH_SIZE) / max(1e-9, tr_elapsed)) if losses else 0.0
        current_lr = self.optimizer.param_groups[0]["lr"]
        avg_grad_norm = (grad_norm_running / max(1, len(losses))) if losses else 0.0
        avg_entropy = (ent_running / max(1, len(losses))) if losses else 0.0
        ent_coef_current = 0.0
        if LOSS_ENTROPY_COEF_INIT > 0 and self.iteration <= LOSS_ENTROPY_ANNEAL_ITERS:
            ent_coef_current = LOSS_ENTROPY_COEF_INIT * (1.0 - (self.iteration - 1) / max(1, LOSS_ENTROPY_ANNEAL_ITERS))
        if LOSS_ENTROPY_COEF_MIN > 0:
            ent_coef_current = max(float(ent_coef_current), float(LOSS_ENTROPY_COEF_MIN))
        drift_pct = 0.0
        if TRAIN_LR_SCHED_STEPS_PER_ITER_EST > 0:
            drift_pct = 100.0 * (actual_steps - TRAIN_LR_SCHED_STEPS_PER_ITER_EST) / TRAIN_LR_SCHED_STEPS_PER_ITER_EST if actual_steps > 0 else 0.0
        if self.iteration == 1 and actual_steps > 0 and abs(drift_pct) > TRAIN_LR_SCHED_DRIFT_ADJUST_THRESHOLD:
            remaining_iters = max(0, TRAIN_TOTAL_ITERATIONS - self.iteration)
            new_total = int(self.scheduler.t + remaining_iters * actual_steps)
            new_total = max(self.scheduler.t + 1, new_total)
            self.scheduler.set_total_steps(new_total)
            self.log.info(f"[LR ] adjust total_steps -> {self.scheduler.total} (iter1 measured {actual_steps} vs est {TRAIN_LR_SCHED_STEPS_PER_ITER_EST}, drift {drift_pct:+.1f}%)")

        stats.update({
            "policy_loss": pol_loss,
            "value_loss": val_loss,
            "learning_rate": float(current_lr),
            "buffer_size": buf_sz,
            "buffer_percent": buf_pct2,
            "training_time": tr_elapsed,
            "batches_per_sec": bps,
            "optimizer_steps": actual_steps,
            "lr_sched_t": self.scheduler.t,
            "lr_sched_total": self.scheduler.total,
        })

        recent_pct = int(round(100.0 * TRAIN_RECENT_SAMPLE_RATIO))
        old_pct = max(0, 100 - recent_pct)
        tr_plan_line = f"plan {steps} | tgt {TRAIN_TARGET_TRAIN_SAMPLES_PER_NEW:.1f}x | act {ratio:>3.1f}x | mix {recent_pct}/{old_pct}%" if steps > 0 else "plan 0 skip (buffer underfilled)"
        tr_line = (
            f"steps {len(losses):>3} | b/s {bps:>4.1f} | sps {self._format_si(sps, digits=1)} | t {self._format_time(tr_elapsed)} | "
            f"P {pol_loss:>6.4f} | V {val_loss:>6.4f} | LR {current_lr:.2e} | grad {avg_grad_norm:>5.3f} | ent {avg_entropy:>5.3f} | "
            f"entc {ent_coef_current:.2e} | clip {TRAIN_GRAD_CLIP_NORM:.1f} | buf {int(buf_pct2):>3}% ({self._format_si(buf_sz)})"
        )
        lr_sched_fragment = f"sched {TRAIN_LR_SCHED_STEPS_PER_ITER_EST}->{actual_steps} | drift {drift_pct:+.0f}% | pos {self.scheduler.t}/{self._format_si(self.scheduler.total)}"

        ev_short = f"req {self._format_si(eval_req)}(+{self._format_si(d_req)}) | hit {hit_rate:>4.1f}% (+{hit_rate_d:>4.1f}%) | batches {self._format_si(eval_batches)}(+{self._format_si(d_batches)}) | evalN {self._format_si(eval_evalN)}(+{self._format_si(d_evalN)}) | bmax {eval_bmax}"
        self.log.info("[SP] " + sp_line + " | [EV] " + ev_short)
        self.log.info("[TR] " + tr_plan_line + " | " + tr_line + " | " + lr_sched_fragment)
        self._prev_eval_m = eval_m

        return stats

    def _clone_from_ema(self) -> torch.nn.Module:
        m = self._clone_model()
        if self.ema is not None:
            self.ema.copy_to(m)
        return m

    def _arena_match(self, challenger: torch.nn.Module, incumbent: torch.nn.Module) -> tuple[float, int, int, int]:
        import chesscore as _ccore
        import numpy as _np

        from .model import BatchedEvaluator as _BatchedEval

        wins = draws = losses = 0
        openings = self._arena_openings
        if not openings:
            raise RuntimeError("Arena openings not loaded.")

        with _BatchedEval(self.device) as ce, _BatchedEval(self.device) as ie:
            ce.refresh_from(challenger)
            ie.refresh_from(incumbent)
            ce.cache_cap = ARENA_EVAL_CACHE_CAPACITY
            ie.cache_cap = ARENA_EVAL_CACHE_CAPACITY

            def play(e1: _BatchedEval, e2: _BatchedEval, start_fen: str) -> int:
                pos = _ccore.Position()
                pos.from_fen(start_fen)
                m1 = _ccore.MCTS(
                    MCTS_EVAL_SIMULATIONS,
                    MCTS_C_PUCT,
                    MCTS_DIRICHLET_ALPHA,
                    0.0 if ARENA_DETERMINISTIC else float(ARENA_DIRICHLET_WEIGHT),
                )
                m1.set_c_puct_params(MCTS_C_PUCT_BASE, MCTS_C_PUCT_INIT)
                m1.set_fpu_reduction(MCTS_FPU_REDUCTION)
                m2 = _ccore.MCTS(
                    MCTS_EVAL_SIMULATIONS,
                    MCTS_C_PUCT,
                    MCTS_DIRICHLET_ALPHA,
                    0.0 if ARENA_DETERMINISTIC else float(ARENA_DIRICHLET_WEIGHT),
                )
                m2.set_c_puct_params(MCTS_C_PUCT_BASE, MCTS_C_PUCT_INIT)
                m2.set_fpu_reduction(MCTS_FPU_REDUCTION)
                t = 0
                while pos.result() == _ccore.ONGOING and t < GAME_MAX_PLIES:
                    visits = m1.search_batched(pos, e1.infer_positions, EVAL_BATCH_SIZE_MAX) if t % 2 == 0 else m2.search_batched(pos, e2.infer_positions, EVAL_BATCH_SIZE_MAX)
                    if visits is None:
                        break
                    v = _np.asarray(visits, dtype=_np.float64)
                    if v.size == 0:
                        break
                    moves = pos.legal_moves()
                    if ARENA_DETERMINISTIC:
                        idx = int(_np.argmax(v))
                    else:
                        if t < ARENA_TEMP_MOVES:
                            temp = float(ARENA_TEMPERATURE)
                            if temp <= 0.0 + ARENA_OPENING_TEMPERATURE_EPS:
                                idx = int(_np.argmax(v))
                            else:
                                v_pos = _np.maximum(v, 0.0)
                                s0 = v_pos.sum()
                                if s0 <= 0:
                                    idx = int(_np.argmax(v))
                                else:
                                    probs = v_pos ** (1.0 / temp)
                                    s = probs.sum()
                                    idx = int(_np.argmax(v)) if s <= 0 else int(_np.random.choice(len(moves), p=probs / s))
                        else:
                            idx = int(_np.argmax(v))
                    pos.make_move(moves[idx])
                    t += 1
                r = pos.result()
                return 1 if r == _ccore.WHITE_WIN else (-1 if r == _ccore.BLACK_WIN else 0)

            pairs = max(1, ARENA_GAMES_PER_EVAL // ARENA_PAIRING_FACTOR)
            if ARENA_STRATIFY_OPENINGS:
                n_open = len(openings)
                if n_open == 0:
                    raise RuntimeError("Arena openings not loaded.")
                if pairs <= n_open:
                    idxs = _np.random.permutation(n_open)[:pairs]
                else:
                    reps = int(_np.ceil(pairs / n_open))
                    idxs = _np.tile(_np.random.permutation(n_open), reps)[:pairs]
            else:
                idxs = _np.random.randint(0, len(openings), size=pairs)
            for i in idxs:
                start_fen = openings[int(i)]
                r1 = play(ce, ie, start_fen)
                r2 = -play(ie, ce, start_fen)
                for r in (r1, r2):
                    if r > 0:
                        wins += 1
                    elif r < 0:
                        losses += 1
                    else:
                        draws += 1

        total_games = max(1, wins + draws + losses)
        score = (wins + ARENA_DRAW_SCORE * draws) / total_games
        return score, wins, draws, losses

    def train(self) -> None:
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        header_lines = [
            f"Device: {self.device} ({self.device_name}) | GPU {self.device_total_gb:.1f} GB | AMP {'on' if AMP_ENABLED else 'off'} | compiled {self._compiled}",
            f"Torch: {torch.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}",
            f"Threads: torch intra {torch.get_num_threads()} | inter {torch.get_num_interop_threads()} | CPU cores {os.cpu_count()} | selfplay workers {SELFPLAY_NUM_WORKERS}",
            f"Model: {total_params:.1f}M parameters | {MODEL_BLOCKS} blocks x {MODEL_CHANNELS} channels | channels_last {MODEL_CHANNELS_LAST}",
            f"Buffer: size {REPLAY_BUFFER_CAPACITY:,}",
            f"Selfplay: sims {MCTS_TRAIN_SIMULATIONS_BASE}→≥{MCTS_TRAIN_SIMULATIONS_MIN} | temp {SELFPLAY_TEMP_HIGH}->{SELFPLAY_TEMP_LOW}@{SELFPLAY_TEMP_MOVES} | resign {RESIGN_VALUE_THRESHOLD} x{RESIGN_CONSECUTIVE_PLIES} | dirichlet α {MCTS_DIRICHLET_ALPHA} w {MCTS_DIRICHLET_WEIGHT}",
            f"Training: {TRAIN_TOTAL_ITERATIONS} iterations | {SELFPLAY_GAMES_PER_ITER} games/iter | sched_est {TRAIN_LR_SCHED_STEPS_PER_ITER_EST} | batch {TRAIN_BATCH_SIZE}",
            f"LR: init {TRAIN_LR_INIT:.2e} | warmup {self.scheduler.warm} | final {TRAIN_LR_FINAL:.2e} | wd {TRAIN_WEIGHT_DECAY} | mom {TRAIN_MOMENTUM}",
            f"Augment: mirror {AUGMENT_MIRROR_PROB:.2f} | rot180 {AUGMENT_ROT180_PROB:.2f} | vflip_cs {AUGMENT_VFLIP_CS_PROB:.2f} | policy_smooth {LOSS_POLICY_LABEL_SMOOTH:.2f}",
            f"EMA: {'on' if EMA_ENABLED else 'off'} decay {EMA_DECAY if EMA_ENABLED else 0}",
            f"Eval: batch {EVAL_BATCH_SIZE_MAX}@{EVAL_BATCH_COALESCE_MS}ms | cache {EVAL_CACHE_CAPACITY}",
            f"Arena: games {ARENA_GAMES_PER_EVAL}/every {ARENA_EVAL_EVERY_ITERS} | rule LB>{100.0 * ARENA_GATE_BASELINE_P:.0f}% @Z {ARENA_GATE_Z_EARLY}->{ARENA_GATE_Z_LATE} (switch @{ARENA_GATE_Z_SWITCH_ITER})",
            f"Arena: det {'on' if ARENA_DETERMINISTIC else 'off'} | dir_w {0.0 if ARENA_DETERMINISTIC else ARENA_DIRICHLET_WEIGHT} | temp {(ARENA_TEMPERATURE if not ARENA_DETERMINISTIC else 0.0)}->0.0 @{ARENA_TEMP_MOVES}",
            f"Arena openings: {len(self._arena_openings):,} positions loaded",
            f"Expected: {TRAIN_TOTAL_ITERATIONS * SELFPLAY_GAMES_PER_ITER:,} total games",
        ]
        title = "Hybrid Chess AI Training"
        bar = "=" * max(60, len(title))
        self.log.info("\n" + bar + f"\n{title}\n" + bar + "\n" + "\n".join(header_lines) + "\n" + bar)

        for iteration in range(self.iteration + 1, TRAIN_TOTAL_ITERATIONS + 1):
            self.iteration = iteration
            iter_stats = self.training_iteration()

            do_eval = (iteration % ARENA_EVAL_EVERY_ITERS) == 0 if ARENA_EVAL_EVERY_ITERS > 0 else False
            arena_elapsed = 0.0
            if do_eval:
                if (not self._gate_active) or (self._gate_rounds >= ARENA_CANDIDATE_MAX_ROUNDS) or ((self._gate.w + self._gate.d + self._gate.losses) >= ARENA_CANDIDATE_MAX_GAMES):
                    if self._gate_active:
                        self.log.info("[AR ] reset: timeboxing stuck challenger")
                    self._pending_challenger = self._clone_from_ema()
                    self._gate.reset()
                    self._gate_active = True
                    self._gate_started_iter = iteration
                    self._gate_rounds = 0
                t_ar = time.time()
                assert self._pending_challenger is not None
                _, aw, ad, al = self._arena_match(self._pending_challenger, self.best_model)
                arena_elapsed = time.time() - t_ar

                self._gate.update(aw, ad, al)
                self._gate_rounds += 1
                decision, m = self._gate.decision()
                self.log.info(
                    "[AR ] "
                    f"W/D/L {aw}/{ad}/{al} | n {int(m.get('n', 0))} | p {100.0 * m.get('p', 0):>5.1f}% | "
                    f"elo {m.get('elo', 0):>6.1f} ±{m.get('se_elo', 0):.1f} | decision {decision.upper()} | "
                    f"time {self._format_time(arena_elapsed)} | age_iter {iteration - self._gate_started_iter} | rounds {self._gate_rounds}"
                )

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
                self.log.info(f"[AR ] skipped | games 0 | time {self._format_time(arena_elapsed)}")

            if (ARENA_EVAL_EVERY_ITERS == 0) and (self.iteration % CHECKPOINT_SAVE_EVERY_ITERS == 0):
                self._save_checkpoint()

            sp_time = float(iter_stats.get("selfplay_time", 0.0))
            tr_time = float(iter_stats.get("training_time", 0.0))
            full_iter_time = sp_time + tr_time + arena_elapsed
            next_ar = 0
            if ARENA_EVAL_EVERY_ITERS > 0:
                k = ARENA_EVAL_EVERY_ITERS
                r = self.iteration % k
                next_ar = (k - r) if r != 0 else k
            peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1024**3
            peak_res = torch.cuda.max_memory_reserved(self.device) / 1024**3
            mem2 = self._get_mem_info()
            self.log.info(
                f"[SUM] iter {self._format_time(full_iter_time)} | sp {self._format_time(sp_time)} | tr {self._format_time(tr_time)} | ar {self._format_time(arena_elapsed)} | "
                f"elapsed {self._format_time(time.time() - self.start_time)} | next_ar {next_ar} | games {self.total_games:,} | "
                f"peak GPU {self._format_gb(peak_alloc)}/{self._format_gb(peak_res)} | RSS {self._format_gb(mem2['rss_gb'])}"
            )

            self._prev_eval_m = self.evaluator.get_metrics()


class EloGater:
    def __init__(
        self,
        z: float = ARENA_GATE_Z_LATE,
        min_games: int = ARENA_GATE_MIN_GAMES,
        draw_w: float = ARENA_GATE_DRAW_WEIGHT,
        baseline_p: float = ARENA_GATE_BASELINE_P,
        decisive_secondary: bool = ARENA_GATE_DECISIVE_SECONDARY,
        min_decisive: int = ARENA_GATE_MIN_DECISIVES,
    ):
        self.z = float(z)
        self.min_games = int(min_games)
        self.draw_w = float(draw_w)
        self.baseline_p = float(baseline_p)
        self.decisive_secondary = bool(decisive_secondary)
        self.min_decisive = int(min_decisive)
        self.reset()

    def reset(self) -> None:
        self.w = 0
        self.d = 0
        self.losses = 0

    def update(self, w: int, d: int, losses: int) -> None:
        self.w += int(w)
        self.d += int(d)
        self.losses += int(losses)

    def decision(self) -> tuple[str, dict[str, float]]:
        import math

        n = self.w + self.d + self.losses
        if n < self.min_games:
            return "undecided", {"n": float(n)}
        p = (self.w + self.draw_w * self.d) / max(1, n)
        w_frac = self.w / n
        d_frac = self.d / n
        var = max(1e-9, w_frac + (self.draw_w**2) * d_frac - (p * p))
        se = math.sqrt(var / n)
        lb = p - self.z * se
        ub = p + self.z * se
        eps = ARENA_GATE_EPS
        pc = min(1.0 - eps, max(eps, p))
        elo = 400.0 * math.log10(pc / (1.0 - pc))
        denom = max(eps, pc * (1.0 - pc))
        se_elo = (400.0 / math.log(10.0)) * se / denom
        if lb > self.baseline_p:
            return "accept", {
                "n": float(n),
                "p": p,
                "lb": lb,
                "elo": elo,
                "se_elo": se_elo,
            }
        if ub < self.baseline_p:
            return "reject", {
                "n": float(n),
                "p": p,
                "ub": ub,
                "elo": elo,
                "se_elo": se_elo,
            }
        decisives = self.w + self.losses
        if self.decisive_secondary and decisives >= self.min_decisive:
            p_dec = self.w / max(1, decisives)
            se_dec = math.sqrt(max(1e-9, p_dec * (1.0 - p_dec) / max(1, decisives)))
            lb_dec = p_dec - self.z * se_dec
            if lb_dec > 0.5:
                pc_dec = min(1.0 - eps, max(eps, p_dec))
                elo_dec = 400.0 * math.log10(pc_dec / (1.0 - pc_dec))
                denom_dec = max(eps, pc_dec * (1.0 - pc_dec))
                se_elo_dec = (400.0 / math.log(10.0)) * se_dec / denom_dec
                return "accept", {
                    "n": float(n),
                    "p": p,
                    "lb": lb,
                    "elo": elo_dec,
                    "se_elo": se_elo_dec,
                }
        return "undecided", {
            "n": float(n),
            "p": p,
            "lb": lb,
            "ub": ub,
            "elo": elo,
            "se_elo": se_elo,
        }


if __name__ == "__main__":
    import sys

    root = logging.getLogger()
    lvl = getattr(logging, str(LOG_LEVEL).upper(), logging.INFO)
    root.setLevel(lvl)
    for h in list(root.handlers):
        root.removeHandler(h)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)
    if LOG_TO_FILE:
        fh = logging.FileHandler(LOG_FILE_PATH, mode="w", encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)
    torch.set_float32_matmul_precision(TORCH_MATMUL_FLOAT32_PRECISION)
    torch.backends.cuda.matmul.allow_tf32 = bool(TORCH_ALLOW_TF32)
    torch.backends.cudnn.allow_tf32 = bool(TORCH_ALLOW_TF32)
    torch.backends.cudnn.benchmark = TORCH_CUDNN_BENCHMARK
    torch.set_num_threads(TORCH_THREADS_INTRA)
    torch.set_num_interop_threads(TORCH_THREADS_INTER)
    import random as _py_random

    _py_random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    resume_flag = any(a in ("--resume", "resume") for a in sys.argv[1:])
    Trainer(resume=resume_flag).train()
