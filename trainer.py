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

from config import (
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
    ARENA_LOG_CSV_ENABLE,
    ARENA_LOG_CSV_PATH,
    METRICS_LOG_CSV_ENABLE,
    METRICS_LOG_CSV_PATH,
    ARENA_GATE_MIN_DECISIVES,
    ARENA_GATE_MIN_GAMES,
    ARENA_GATE_Z_EARLY,
    ARENA_GATE_Z_LATE,
    ARENA_GATE_Z_SWITCH_ITER,
    ARENA_OPENING_TEMPERATURE_EPS,
    ARENA_PAIRING_FACTOR,
    ARENA_TEMP_MOVES,
    ARENA_TEMPERATURE,
    AUGMENT_MIRROR_PROB,
    AUGMENT_ROT180_PROB,
    AUGMENT_VFLIP_CS_PROB,
    BEST_MODEL_FILE_PATH,
    CHECKPOINT_FILE_PATH,
    CHECKPOINT_SAVE_EVERY_ITERS,
    CUDA_EMPTY_CACHE_EVERY_ITERS,
    DYN_ARENA_EVAL_MAX,
    DYN_ARENA_EVAL_MIN,
    DYN_ARENA_EVAL_STEP,
    DYN_EVAL_MAX,
    DYN_EVAL_MIN,
    DYN_EVAL_STEP,
    DYN_RAM_HIGH_PCT,
    DYN_RAM_LOW_PCT,
    DYN_REPLAY_MAX,
    DYN_REPLAY_MIN,
    DYN_REPLAY_STEP,
    DYN_TUNE_COOLDOWN_ITERS,
    DYN_TUNE_RAM_ENABLED,
    EMA_DECAY,
    EMA_ENABLED,
    EVAL_BATCH_COALESCE_MS,
    EVAL_BATCH_SIZE_MAX,
    EVAL_CACHE_CAPACITY,
    EVAL_CACHE_USE_FP16,
    EVAL_MODEL_CHANNELS_LAST,
    EVAL_PIN_MEMORY,
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
    MODEL_VALUE_CONV_CHANNELS,
    MODEL_VALUE_HIDDEN_DIM,
    REPLAY_BUFFER_CAPACITY,
    RESIGN_CONSECUTIVE_MIN,
    RESIGN_CONSECUTIVE_PLIES,
    RESIGN_VALUE_THRESHOLD,
    RESIGN_DISABLE_UNTIL_ITERS,
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
    TRAIN_BATCH_SIZE_MAX,
    TRAIN_BATCH_SIZE_MIN,
    TRAIN_TARGET_TRAIN_SAMPLES_PER_NEW,
    TRAIN_TOTAL_ITERATIONS,
    TRAIN_UPDATE_STEPS_MAX,
    TRAIN_UPDATE_STEPS_MIN,
    TRAIN_WEIGHT_DECAY,
    U8_SCALE,
    VALUE_I8_SCALE,
)
from model import BatchedEvaluator, ChessNet
from selfplay import SelfPlayEngine


class Trainer:
    """Coordinates self-play generation, training updates, and model gating."""

    def __init__(self, device: str | None = None, resume: bool = False) -> None:
        self.log = logging.getLogger("hybridchess.trainer")
        device_obj = torch.device(device or "cuda")
        self.device = device_obj
        net_any: Any = ChessNet().to(self.device)
        if MODEL_CHANNELS_LAST:
            self.model = net_any.to(memory_format=torch.channels_last)
        else:
            self.model = net_any
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
            (
                nodecay
                if (n.endswith(".bias") or "bn" in n.lower() or "batchnorm" in n.lower())
                else decay
            ).append(p)

        self.optimizer = torch.optim.SGD(
            [
                {"params": decay, "weight_decay": TRAIN_WEIGHT_DECAY},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=TRAIN_LR_INIT,
            momentum=TRAIN_MOMENTUM,
            nesterov=True,
        )

        total_expected_train_steps = int(TRAIN_TOTAL_ITERATIONS * TRAIN_LR_SCHED_STEPS_PER_ITER_EST)
        warmup_steps_clamped = int(
            max(1, min(TRAIN_LR_WARMUP_STEPS, max(1, total_expected_train_steps - 1)))
        )

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
                    lr = self.final + (self.base - self.final) * 0.5 * (
                        1.0 + math.cos(math.pi * progress)
                    )
                for pg in self.opt.param_groups:
                    pg["lr"] = lr

            def peek_next_lr(self) -> float:
                t_next = self.t + 1
                if t_next <= self.warm:
                    lr = self.base * (t_next / self.warm)
                else:
                    import math

                    progress = min(1.0, (t_next - self.warm) / max(1, self.total - self.warm))
                    lr = self.final + (self.base - self.final) * 0.5 * (
                        1.0 + math.cos(math.pi * progress)
                    )
                return lr

            def set_total_steps(self, total_steps: int) -> None:
                self.total = max(self.t + 1, int(total_steps))
                if self.warm >= self.total:
                    self.warm = max(1, self.total - 1)

        self.scheduler = WarmupCosine(
            self.optimizer,
            TRAIN_LR_INIT,
            warmup_steps_clamped,
            TRAIN_LR_FINAL,
            total_expected_train_steps,
        )
        use_bf16 = AMP_PREFER_BFLOAT16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(AMP_ENABLED and not use_bf16))

        self.evaluator = BatchedEvaluator(self.device)
        self._eval_batch_cap = int(EVAL_BATCH_SIZE_MAX)
        self._eval_coalesce_ms = int(EVAL_BATCH_COALESCE_MS)
        try:
            self.evaluator.set_batching_params(self._eval_batch_cap, self._eval_coalesce_ms)
        except Exception:
            pass
        self.train_batch_size: int = int(TRAIN_BATCH_SIZE)
        self._tune_cooldown_counter: int = 0
        self._current_eval_cache_cap: int = int(EVAL_CACHE_CAPACITY)
        self._current_replay_cap: int = int(REPLAY_BUFFER_CAPACITY)
        self._arena_eval_cache_cap: int = int(ARENA_EVAL_CACHE_CAPACITY)
        self._oom_cooldown_iters: int = 0

        self._proc = psutil.Process(os.getpid())
        psutil.cpu_percent(0.0)
        self._proc.cpu_percent(0.0)

        class EMA:
            """Lightweight EMA of model parameters for evaluation and gating."""

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
        if DYN_TUNE_RAM_ENABLED:
            try:
                new_eval_cap = max(0, min(self._current_eval_cache_cap, int(DYN_EVAL_MIN)))
                if new_eval_cap != self._current_eval_cache_cap:
                    self.evaluator.set_cache_capacity(new_eval_cap)
                    self._current_eval_cache_cap = new_eval_cap
                new_replay_cap = max(1, min(self._current_replay_cap, int(DYN_REPLAY_MIN)))
                self.selfplay_engine.set_capacity(new_replay_cap)
                self._current_replay_cap = new_replay_cap
                self._arena_eval_cache_cap = max(
                    0, min(self._arena_eval_cache_cap, int(DYN_ARENA_EVAL_MIN))
                )
            except Exception:
                pass
        self.iteration = 0
        self.total_games = 0
        self.start_time = time.time()
        props = torch.cuda.get_device_properties(self.device)
        self.device_name = props.name
        self.device_total_gb = props.total_memory / 1024**3
        self._prev_eval_m: dict[str, float] = {}


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

    def _startup_summary(self) -> str:
        props = torch.cuda.get_device_properties(self.device)
        sm = f"{props.major}.{props.minor}"
        mp = int(getattr(props, "multi_processor_count", 0))
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        autocast_mode = (
            "bf16"
            if (AMP_ENABLED and AMP_PREFER_BFLOAT16 and bf16_supported)
            else ("fp16" if AMP_ENABLED else "off")
        )
        tf32_effective = bool(
            torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        )
        total_params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        env_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
        env_modload = os.environ.get("CUDA_MODULE_LOADING", "default")
        env_cache = os.environ.get("CUDA_CACHE_MAXSIZE", "default")

        sections: list[str] = []
        sections.append(
            f"[GPU     ] {self.device_name} (device {self.device}) | SM {sm} | VRAM {self.device_total_gb:.1f}G | MPs {mp}"
        )
        sections.append(
            f"[PyTorch ] {torch.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()} | compile={'on' if self._compiled else 'off'}(mode={TORCH_COMPILE_MODE}, full={TORCH_COMPILE_FULLGRAPH}, dyn={TORCH_COMPILE_DYNAMIC})"
        )
        sections.append(
            f"[Precision] autocast={autocast_mode} | TF32={'on' if tf32_effective else 'off'} | matmul={torch.get_float32_matmul_precision()} | GradScaler={'on' if (AMP_ENABLED and not AMP_PREFER_BFLOAT16) else 'off'}"
        )
        sections.append(
            f"[Memory  ] channels_last train={MODEL_CHANNELS_LAST} eval={EVAL_MODEL_CHANNELS_LAST} | pin_memory train={TRAIN_PIN_MEMORY} eval={EVAL_PIN_MEMORY}"
        )
        sections.append(
            f"[Allocator] {env_alloc} | cuda_module_loading={env_modload} | cuda_cache_max={env_cache}"
        )
        try:
            sp_workers = int(self.selfplay_engine.get_num_workers())
        except Exception:
            sp_workers = int(SELFPLAY_NUM_WORKERS)
        sections.append(
            f"[Threads ] torch intra={torch.get_num_threads()} inter={torch.get_num_interop_threads()} | CPU cores={os.cpu_count()} | selfplay workers={sp_workers}"
        )
        sections.append(
            f"[Model   ] params={total_params_m:.1f}M | blocks={MODEL_BLOCKS} | channels={MODEL_CHANNELS} | value_head=({MODEL_VALUE_CONV_CHANNELS}->{MODEL_VALUE_HIDDEN_DIM})"
        )
        sections.append(f"[Replay  ] capacity={REPLAY_BUFFER_CAPACITY:,}")
        sections.append(
            f"[SelfPlay] sims {MCTS_TRAIN_SIMULATIONS_BASE}→≥{MCTS_TRAIN_SIMULATIONS_MIN} | temp {SELFPLAY_TEMP_HIGH}->{SELFPLAY_TEMP_LOW}@{SELFPLAY_TEMP_MOVES} | resign {RESIGN_VALUE_THRESHOLD} x{RESIGN_CONSECUTIVE_PLIES} | dirichlet α={MCTS_DIRICHLET_ALPHA} w={MCTS_DIRICHLET_WEIGHT}"
        )
        sections.append(
            f"[Train   ] iters={TRAIN_TOTAL_ITERATIONS} | games/iter={SELFPLAY_GAMES_PER_ITER} | batch={self.train_batch_size} | sched_est={TRAIN_LR_SCHED_STEPS_PER_ITER_EST}"
        )
        sections.append(
            f"[LR      ] init={TRAIN_LR_INIT:.2e} | warmup={self.scheduler.warm} | final={TRAIN_LR_FINAL:.2e} | weight_decay={TRAIN_WEIGHT_DECAY} | momentum={TRAIN_MOMENTUM}"
        )
        sections.append(
            f"[Loss    ] policy={LOSS_POLICY_WEIGHT} | value={LOSS_VALUE_WEIGHT}->{LOSS_VALUE_WEIGHT_LATE}@{LOSS_VALUE_WEIGHT_SWITCH_ITER} | smooth={LOSS_POLICY_LABEL_SMOOTH:.2f} | entropy={LOSS_ENTROPY_COEF_INIT:.2e}→{LOSS_ENTROPY_COEF_MIN:.2e}@{LOSS_ENTROPY_ANNEAL_ITERS}"
        )
        sections.append(
            f"[Augment ] mirror={AUGMENT_MIRROR_PROB:.2f} | rot180={AUGMENT_ROT180_PROB:.2f} | vflip_cs={AUGMENT_VFLIP_CS_PROB:.2f}"
        )
        sections.append(
            f"[EMA     ] {'on' if EMA_ENABLED else 'off'} | decay={EMA_DECAY if EMA_ENABLED else 0}"
        )
        sections.append(
            f"[Eval    ] batch_max={EVAL_BATCH_SIZE_MAX} @ {EVAL_BATCH_COALESCE_MS}ms | cache={EVAL_CACHE_CAPACITY:,} | dtype={'fp16' if EVAL_CACHE_USE_FP16 else 'fp32'}"
        )
        sections.append(
            f"[Arena   ] every={ARENA_EVAL_EVERY_ITERS} | games={ARENA_GAMES_PER_EVAL} | baseline_p={100.0 * ARENA_GATE_BASELINE_P:.0f}% | Z {ARENA_GATE_Z_EARLY}->{ARENA_GATE_Z_LATE} (switch@{ARENA_GATE_Z_SWITCH_ITER}) | min_games={ARENA_GATE_MIN_GAMES} | det={'on' if ARENA_DETERMINISTIC else 'off'}"
        )
        sections.append(
            f"[Expect  ] total_games={TRAIN_TOTAL_ITERATIONS * SELFPLAY_GAMES_PER_ITER:,}"
        )
        title = "Hybrid Chess AI Training"
        bar = "=" * max(72, len(title))
        return "\n" + bar + f"\n{title}\n" + bar + "\n" + "\n".join(sections) + "\n" + bar

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
        proc = self._proc
        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1024**3,
            "total_gb": self.device_total_gb,
            "rss_gb": proc.memory_info().rss / 1024**3,
        }

    def _get_sys_info(self) -> dict[str, float]:
        vmem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        try:
            load1, load5, load15 = os.getloadavg()
        except Exception:
            load1 = load5 = load15 = 0.0
        return {
            "cpu_sys_pct": float(psutil.cpu_percent(0.0)),
            "cpu_proc_pct": float(self._proc.cpu_percent(0.0)),
            "ram_used_gb": float(vmem.used) / 1024**3,
            "ram_total_gb": float(vmem.total) / 1024**3,
            "ram_pct": float(vmem.percent),
            "swap_used_gb": float(swap.used) / 1024**3,
            "swap_total_gb": float(swap.total) / 1024**3,
            "swap_pct": float(swap.percent),
            "load1": float(load1),
            "load5": float(load5),
            "load15": float(load15),
        }

    def _maybe_autotune_ram(self, sys_info: dict[str, float]) -> None:
        """Adjust replay/eval cache sizes based on host RAM pressure."""
        if not DYN_TUNE_RAM_ENABLED:
            return
        try:
            used_pct = float(sys_info.get("ram_pct", 0.0))
            if self._tune_cooldown_counter > 0:
                self._tune_cooldown_counter -= 1
                return
            grow = used_pct < float(DYN_RAM_LOW_PCT)
            shrink = used_pct > float(DYN_RAM_HIGH_PCT)
            if not (grow or shrink):
                return
            new_replay = int(self._current_replay_cap)
            new_eval = int(self._current_eval_cache_cap)
            new_arena = int(self._arena_eval_cache_cap)
            if grow:
                new_replay = min(int(DYN_REPLAY_MAX), new_replay + int(DYN_REPLAY_STEP))
                new_eval = min(int(DYN_EVAL_MAX), new_eval + int(DYN_EVAL_STEP))
                new_arena = min(int(DYN_ARENA_EVAL_MAX), new_arena + int(DYN_ARENA_EVAL_STEP))
            else:
                new_replay = max(int(DYN_REPLAY_MIN), new_replay - int(DYN_REPLAY_STEP))
                new_eval = max(int(DYN_EVAL_MIN), new_eval - int(DYN_EVAL_STEP))
                new_arena = max(int(DYN_ARENA_EVAL_MIN), new_arena - int(DYN_ARENA_EVAL_STEP))
            changes: list[str] = []
            if new_replay != self._current_replay_cap:
                try:
                    self.selfplay_engine.set_capacity(new_replay)
                    changes.append(f"replay {self._current_replay_cap}->{new_replay}")
                    self._current_replay_cap = new_replay
                except Exception:
                    pass
            if new_eval != self._current_eval_cache_cap:
                try:
                    self.evaluator.set_cache_capacity(new_eval)
                    changes.append(f"eval_cache {self._current_eval_cache_cap}->{new_eval}")
                    self._current_eval_cache_cap = new_eval
                except Exception:
                    pass
            if new_arena != self._arena_eval_cache_cap:
                self._arena_eval_cache_cap = new_arena
                changes.append(f"arena_cache ->{new_arena}")
            if changes:
                self._tune_cooldown_counter = int(max(0, int(DYN_TUNE_COOLDOWN_ITERS)))
                self.log.info(
                    "[AUTO] mem_tune | " + " | ".join(changes) + f" | RAM {used_pct:.0f}%"
                )
        except Exception:
            pass

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
                "pending_challenger": (
                    None
                    if (not self._gate_active or self._pending_challenger is None)
                    else self._pending_challenger.state_dict()
                ),
                "rng": {
                    "py": random.getstate(),
                    "np": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all(),
                },
            }
            ckpt_dir = os.path.dirname(CHECKPOINT_FILE_PATH)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            tmp_path = f"{CHECKPOINT_FILE_PATH}.tmp"  # atomic replace
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, CHECKPOINT_FILE_PATH)
            self.log.info("[CKPT] saved checkpoint")
        except Exception as e:
            self.log.warning(f"Failed to save checkpoint: {e}")

    def _save_best_model(self) -> None:
        try:
            state_dict = self.best_model.state_dict()
            payload = {
                "iter": int(self.iteration),
                "total_games": int(self.total_games),
                "model": state_dict,
            }
            best_dir = os.path.dirname(BEST_MODEL_FILE_PATH)
            if best_dir:
                os.makedirs(best_dir, exist_ok=True)
            tmp_path = f"{BEST_MODEL_FILE_PATH}.tmp"  # atomic replace
            torch.save(payload, tmp_path)
            os.replace(tmp_path, BEST_MODEL_FILE_PATH)
            self.log.info("[BEST] saved best model")
        except Exception as e:
            self.log.warning(f"Failed to save best model: {e}")

    def _try_resume(self) -> None:
        path = CHECKPOINT_FILE_PATH
        fallback = os.path.basename(path)
        if not os.path.isfile(path) and fallback and os.path.isfile(fallback):
            path = fallback
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
                best_ckpt = torch.load(BEST_MODEL_FILE_PATH, map_location="cpu")
                state_dict = best_ckpt.get("model", best_ckpt)
                self.best_model.load_state_dict(state_dict, strict=True)
                self.best_model.eval()
                self.evaluator.refresh_from(self.best_model)
            elif "best_model" in ckpt:
                self.best_model.load_state_dict(ckpt["best_model"], strict=True)
                self.best_model.eval()
                self.evaluator.refresh_from(self.best_model)
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                sd = ckpt["scheduler"]
                self.scheduler.set_total_steps(int(sd.get("total", self.scheduler.total)))
                self.scheduler.t = int(sd.get("t", self.scheduler.t))
            if "scaler" in ckpt and isinstance(self.scaler, torch.amp.GradScaler):
                self.scaler.load_state_dict(ckpt["scaler"])
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
            if "py" in rng:
                random.setstate(rng["py"])
            if "np" in rng:
                np.random.set_state(rng["np"])
            if "torch_cpu" in rng:
                torch.set_rng_state(rng["torch_cpu"])
            if "torch_cuda" in rng:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            self._prev_eval_m = self.evaluator.get_metrics()
            if not getattr(self, "_prev_eval_m", None):
                self._prev_eval_m = {}
            self.log.info(f"[CKPT] resumed from {path} @ iter {self.iteration}")
        except Exception as e:
            self.log.warning(f"[CKPT] failed to resume: {e}")


    def train_step(
        self, batch_data: tuple[list[Any], list[np.ndarray], list[np.ndarray]]
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        states_u8_list, counts_u16_list, values_i8_list = batch_data
        x_u8_np = np.stack(states_u8_list).astype(np.uint8, copy=False)
        x_u8_t = torch.from_numpy(x_u8_np)  # CPU uint8 [B,C,H,W]
        if TRAIN_PIN_MEMORY:
            try:
                x_u8_t = x_u8_t.pin_memory()
            except Exception:
                pass
        x = x_u8_t.to(self.device, non_blocking=True)
        pi_u16_np = np.stack(counts_u16_list).astype(np.uint16, copy=False)
        pi_u16_t = torch.from_numpy(pi_u16_np)  # CPU uint16 [B,P]
        if TRAIN_PIN_MEMORY:
            try:
                pi_u16_t = pi_u16_t.pin_memory()
            except Exception:
                pass
        pi_counts = pi_u16_t.to(self.device, non_blocking=True).to(dtype=torch.float32)
        denom = pi_counts.sum(dim=1, keepdim=True)
        num_actions = int(pi_counts.shape[1])
        pi_target = torch.where(
            denom > 0.0,
            pi_counts / denom.clamp_min(1.0),
            torch.full_like(pi_counts, 1.0 / max(1, num_actions)),
        )
        v_i8_np = np.asarray(values_i8_list, dtype=np.int8)
        v_i8_t = torch.from_numpy(v_i8_np)
        if TRAIN_PIN_MEMORY:
            try:
                v_i8_t = v_i8_t.pin_memory()
            except Exception:
                pass
        v_target = v_i8_t.to(self.device, non_blocking=True).to(dtype=torch.float32) / float(
            VALUE_I8_SCALE
        )
        x = x.to(dtype=torch.float32) / float(U8_SCALE)
        x = x.contiguous(memory_format=torch.channels_last)
        if not hasattr(self, "_aug_mirror_idx"):
            from selfplay import Augment as _Aug

            mirror_idx = _Aug._policy_index_permutation("mirror")
            rot180_idx = _Aug._policy_index_permutation("rot180")
            vflip_idx = _Aug._policy_index_permutation("vflip_cs")
            plane_perm = _Aug._vflip_cs_plane_permutation(x.shape[1])
            feat_idx = _Aug._feature_plane_indices()
            self._turn_plane_idx = int(feat_idx.get("turn_plane", x.shape[1]))
            self._aug_mirror_idx = torch.tensor(mirror_idx, dtype=torch.long, device=self.device)
            self._aug_rot180_idx = torch.tensor(rot180_idx, dtype=torch.long, device=self.device)
            self._aug_vflip_idx = torch.tensor(vflip_idx, dtype=torch.long, device=self.device)
            self._aug_vflip_plane_perm = torch.tensor(
                plane_perm, dtype=torch.long, device=self.device
            )
        if np.random.rand() < AUGMENT_MIRROR_PROB:
            x = torch.flip(x, dims=[-1])
            pi_target = pi_target.index_select(1, self._aug_mirror_idx)
        if np.random.rand() < AUGMENT_ROT180_PROB:
            x = torch.flip(x, dims=[-1, -2])
            pi_target = pi_target.index_select(1, self._aug_rot180_idx)
        if np.random.rand() < AUGMENT_VFLIP_CS_PROB:
            x = torch.flip(x, dims=[-2])
            x = x.index_select(1, self._aug_vflip_plane_perm)
            if 0 <= getattr(self, "_turn_plane_idx", x.shape[1]) < x.shape[1]:
                x[:, self._turn_plane_idx] = 1.0 - x[:, self._turn_plane_idx]
            pi_target = pi_target.index_select(1, self._aug_vflip_idx)
            v_target = -v_target
        self.model.train()
        with torch.autocast(
            device_type="cuda",
            dtype=(torch.bfloat16 if AMP_PREFER_BFLOAT16 else torch.float16),
            enabled=AMP_ENABLED,
        ):
            policy_logits, value_pred = self.model(x)
            value_pred = value_pred.squeeze(-1)
            if LOSS_POLICY_LABEL_SMOOTH > 0.0:
                num_actions = pi_target.shape[1]
                pi_smooth = (1.0 - LOSS_POLICY_LABEL_SMOOTH) * pi_target + (
                    LOSS_POLICY_LABEL_SMOOTH / num_actions
                )
            else:
                pi_smooth = pi_target
            policy_loss = F.kl_div(
                F.log_softmax(policy_logits, dim=1), pi_smooth, reduction="batchmean"
            )
            value_loss = F.mse_loss(value_pred, v_target)
            entropy_coef = 0.0
            if LOSS_ENTROPY_COEF_INIT > 0 and self.iteration <= LOSS_ENTROPY_ANNEAL_ITERS:
                entropy_coef = LOSS_ENTROPY_COEF_INIT * (
                    1.0 - (self.iteration - 1) / max(1, LOSS_ENTROPY_ANNEAL_ITERS)
                )
            if LOSS_ENTROPY_COEF_MIN > 0:
                entropy_coef = max(float(entropy_coef), float(LOSS_ENTROPY_COEF_MIN))
            entropy = (
                -(F.softmax(policy_logits, dim=1) * F.log_softmax(policy_logits, dim=1)).sum(dim=1)
            ).mean()
            value_loss_weight = (
                LOSS_VALUE_WEIGHT_LATE
                if self.iteration >= LOSS_VALUE_WEIGHT_SWITCH_ITER
                else LOSS_VALUE_WEIGHT
            )
            total_loss = (
                LOSS_POLICY_WEIGHT * policy_loss
                + value_loss_weight * value_loss
                - entropy_coef * entropy
            )
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), TRAIN_GRAD_CLIP_NORM
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)
        return (
            policy_loss.detach(),
            value_loss.detach(),
            float(grad_total_norm.detach().cpu()),
            float(entropy.detach().cpu()),
        )

    def training_iteration(self) -> dict[str, int | float]:
        if self.iteration < int(RESIGN_DISABLE_UNTIL_ITERS):
            self.selfplay_engine.resign_consecutive = 0
        else:
            self.selfplay_engine.resign_consecutive = max(int(RESIGN_CONSECUTIVE_PLIES), int(RESIGN_CONSECUTIVE_MIN))
        if self.iteration == ARENA_GATE_Z_SWITCH_ITER:
            self._gate.z = float(ARENA_GATE_Z_LATE)
        stats: dict[str, int | float] = {}
        torch.cuda.reset_peak_memory_stats(self.device)
        header_lr = float(self.scheduler.peek_next_lr())
        mem_info = self._get_mem_info()
        sys_info = self._get_sys_info()
        buffer_len = len(self.selfplay_engine.buffer)
        try:
            buffer_cap = int(self.selfplay_engine.get_capacity())
        except Exception:
            buffer_cap = int(self.selfplay_engine.buffer.maxlen or 1)
        buffer_pct = (buffer_len / buffer_cap) * 100
        try:
            import psutil as _ps
            max_workers_cap = int(min(32, max(4, (_ps.cpu_count(logical=True) or 8) - 2)))
        except Exception:
            max_workers_cap = 16
        try:
            cur_workers = int(self.selfplay_engine.get_num_workers())
        except Exception:
            cur_workers = None
        if cur_workers is not None:
            cpu_pct = float(sys_info.get("cpu_sys_pct", 0.0))
            target_workers = int(cur_workers)
            if (cpu_pct < 55.0 and buffer_pct < 60.0 and cur_workers < max_workers_cap):
                target_workers = min(max_workers_cap, cur_workers + 2)
            elif cpu_pct > 85.0 and cur_workers > 4:
                target_workers = max(4, cur_workers - 2)
            if target_workers != cur_workers:
                try:
                    self.selfplay_engine.set_num_workers(target_workers)
                    self.log.info(f"[AUTO] selfplay_workers {cur_workers} -> {target_workers}")
                except Exception:
                    pass
        total_elapsed = time.time() - self.start_time
        pct_done = 100.0 * (self.iteration - 1) / max(1, TRAIN_TOTAL_ITERATIONS)
        self.log.info(
            f"[ITR {self.iteration:>3}/{TRAIN_TOTAL_ITERATIONS} {pct_done:>4.1f}%] "
            f"LRnext {header_lr:.2e} | t {self._format_time(total_elapsed)} | "
            f"buf {self._format_si(buffer_len)}/{self._format_si(buffer_cap)} ({int(buffer_pct)}%) | "
            f"GPU {self._format_gb(mem_info['allocated_gb'])}/{self._format_gb(mem_info['reserved_gb'])}/{self._format_gb(mem_info['total_gb'])} | "
            f"RSS {self._format_gb(mem_info['rss_gb'])} | CPU {sys_info['cpu_sys_pct']:.0f}/{sys_info['cpu_proc_pct']:.0f}% | "
            f"RAM {self._format_gb(sys_info['ram_used_gb'])}/{self._format_gb(sys_info['ram_total_gb'])} ({sys_info['ram_pct']:.0f}%) | "
            f"load {sys_info['load1']:.2f}"
        )
        self._maybe_autotune_ram(sys_info)

        t0 = time.time()
        game_stats = self.selfplay_engine.play_games(SELFPLAY_GAMES_PER_ITER)
        self.total_games += int(game_stats["games"])
        sp_elapsed = time.time() - t0

        games_per_min = game_stats["games"] / max(1e-9, sp_elapsed / 60)
        moves_per_sec = game_stats["moves"] / max(1e-9, sp_elapsed)
        games_count = int(game_stats["games"])
        white_wins = int(game_stats["white_wins"])
        black_wins = int(game_stats["black_wins"])
        draws_count = int(game_stats["draws"])
        white_win_pct = 100.0 * white_wins / max(1, games_count)
        draw_pct = 100.0 * draws_count / max(1, games_count)
        black_win_pct = 100.0 * black_wins / max(1, games_count)
        avg_game_length = game_stats["moves"] / max(1, games_count)

        eval_metrics = self.evaluator.get_metrics()
        requests_total = int(eval_metrics.get("requests_total", 0))
        cache_hits_total = int(eval_metrics.get("cache_hits_total", 0))
        batches_total = int(eval_metrics.get("batches_total", 0))
        eval_positions_total = int(eval_metrics.get("eval_positions_total", 0))
        max_batch_size = int(eval_metrics.get("batch_size_max", 0))
        prev_metrics = getattr(self, "_prev_eval_m", {}) or {}
        delta_requests = int(requests_total - int(prev_metrics.get("requests_total", 0)))
        delta_hits = int(cache_hits_total - int(prev_metrics.get("cache_hits_total", 0)))
        delta_batches = int(batches_total - int(prev_metrics.get("batches_total", 0)))
        delta_eval_positions = int(
            eval_positions_total - int(prev_metrics.get("eval_positions_total", 0))
        )
        hit_rate = (100.0 * cache_hits_total / max(1, requests_total)) if requests_total else 0.0
        hit_rate_d = (100.0 * delta_hits / max(1, delta_requests)) if delta_requests > 0 else 0.0

        try:
            if delta_batches > 0 and delta_eval_positions > 0:
                avg_batch = delta_eval_positions / max(1, delta_batches)
                cap = int(self._eval_batch_cap)
                max_cap_allowed = int(min(8192, max(1024, int(2048 * (self.device_total_gb / 16.0)))))
                mem_now = self._get_mem_info()
                reserved_frac = float(mem_now["reserved_gb"]) / max(1e-9, float(mem_now["total_gb"]))
                if avg_batch >= 0.90 * cap and cap < max_cap_allowed and reserved_frac < 0.90:
                    new_cap = int(min(max_cap_allowed, cap + 512))
                    if new_cap != cap:
                        self._eval_batch_cap = new_cap
                        self.evaluator.set_batching_params(batch_size_max=new_cap)
                        self.log.info(f"[AUTO] eval_batch_size_max {cap} -> {new_cap}")
                elif avg_batch <= 0.25 * cap and self._eval_coalesce_ms > 4:
                    old_ms = int(self._eval_coalesce_ms)
                    new_ms = int(max(4, int(old_ms * 0.8)))
                    if new_ms != old_ms:
                        self._eval_coalesce_ms = new_ms
                        self.evaluator.set_batching_params(coalesce_ms=new_ms)
                        self.log.info(f"[AUTO] eval_coalesce_ms {old_ms} -> {new_ms}")
                elif avg_batch >= 0.80 * cap and self._eval_coalesce_ms < 50:
                    old_ms = int(self._eval_coalesce_ms)
                    new_ms = int(min(50, int(old_ms * 1.2 + 1)))
                    if new_ms != old_ms:
                        self._eval_coalesce_ms = new_ms
                        self.evaluator.set_batching_params(coalesce_ms=new_ms)
                        self.log.info(f"[AUTO] eval_coalesce_ms {old_ms} -> {new_ms}")
        except Exception:
            pass

        sp_line = (
            f"games {games_count:,} | W/D/B {white_wins}/{draws_count}/{black_wins} "
            f"({white_win_pct:.0f}%/{draw_pct:.0f}%/{black_win_pct:.0f}%) | len {avg_game_length:>4.1f} | "
            f"gpm {games_per_min:>5.1f} | mps {moves_per_sec / 1000:>4.1f}k | t {self._format_time(sp_elapsed)} | "
            f"new {self._format_si(int(game_stats.get('moves', 0)))}"
        )

        ev_short = (
            f"req {self._format_si(requests_total)}(+{self._format_si(delta_requests)}) | hit {hit_rate:>4.1f}% (+{hit_rate_d:>4.1f}%) | "
            f"batches {self._format_si(batches_total)}(+{self._format_si(delta_batches)}) | "
            f"evalN {self._format_si(eval_positions_total)}(+{self._format_si(delta_eval_positions)}) | bmax {max_batch_size}"
        )
        self.log.info("[SP] " + sp_line + "\n" + " " * 6 + "[EV] " + ev_short)

        stats.update(game_stats)
        stats["selfplay_time"] = sp_elapsed
        stats["games_per_min"] = games_per_min
        stats["moves_per_sec"] = moves_per_sec

        t1 = time.time()
        if (CUDA_EMPTY_CACHE_EVERY_ITERS > 0) and (
            self.iteration % CUDA_EMPTY_CACHE_EVERY_ITERS == 0
        ):
            torch.cuda.empty_cache()

        losses: list[tuple[torch.Tensor, torch.Tensor]] = []
        snapshot = self.selfplay_engine.snapshot()
        min_batch_samples = max(1, self.train_batch_size // 2)
        new_examples = int(game_stats.get("moves", 0))
        target_train_samples = int(TRAIN_TARGET_TRAIN_SAMPLES_PER_NEW * max(1, new_examples))
        num_steps = int(np.ceil(target_train_samples / max(1, self.train_batch_size)))
        num_steps = max(TRAIN_UPDATE_STEPS_MIN, min(TRAIN_UPDATE_STEPS_MAX, num_steps))
        samples_ratio = 0.0
        if len(snapshot) < min_batch_samples:
            num_steps = 0
        else:
            samples_ratio = (num_steps * self.train_batch_size) / max(1, new_examples)

        grad_norm_sum: float = 0.0
        entropy_sum: float = 0.0
        for _step in range(num_steps):
            batch = self.selfplay_engine.sample_from_snapshot(
                snapshot, self.train_batch_size, recent_ratio=TRAIN_RECENT_SAMPLE_RATIO
            )
            if not batch:
                continue
            states, policies, values = batch
            try:
                policy_loss_t, value_loss_t, grad_norm_val, pred_entropy = self.train_step(
                    (states, policies, values)
                )
                grad_norm_sum += float(grad_norm_val)
                entropy_sum += float(pred_entropy)
                losses.append((policy_loss_t, value_loss_t))
            except torch.OutOfMemoryError:
                prev_bs_local = int(self.train_batch_size)
                new_bs_local = int(max(int(TRAIN_BATCH_SIZE_MIN), prev_bs_local - 1024))
                if new_bs_local < prev_bs_local:
                    self.train_batch_size = new_bs_local
                    self.log.info(
                        f"[AUTO] OOM encountered; reducing train_batch_size {prev_bs_local} -> {new_bs_local}"
                    )
                self._oom_cooldown_iters = max(int(self._oom_cooldown_iters), 3)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    torch.cuda.reset_peak_memory_stats(self.device)
                except Exception:
                    pass
                break

        train_elapsed_s = time.time() - t1
        actual_update_steps = len(losses)
        avg_policy_loss = (
            float(torch.stack([pair[0] for pair in losses]).mean().detach().cpu())
            if losses
            else float("nan")
        )
        avg_value_loss = (
            float(torch.stack([pair[1] for pair in losses]).mean().detach().cpu())
            if losses
            else float("nan")
        )
        buffer_size = len(self.selfplay_engine.buffer)
        try:
            buffer_capacity2 = int(self.selfplay_engine.get_capacity())
        except Exception:
            buffer_capacity2 = int(self.selfplay_engine.buffer.maxlen or 1)
        buffer_pct2 = (buffer_size / buffer_capacity2) * 100
        batches_per_sec = (len(losses) / max(1e-9, train_elapsed_s)) if losses else 0.0
        samples_per_sec = (
            ((len(losses) * self.train_batch_size) / max(1e-9, train_elapsed_s)) if losses else 0.0
        )
        learning_rate = self.optimizer.param_groups[0]["lr"]
        avg_grad_norm = (grad_norm_sum / max(1, len(losses))) if losses else 0.0
        avg_entropy = (entropy_sum / max(1, len(losses))) if losses else 0.0
        entropy_coef = 0.0
        if LOSS_ENTROPY_COEF_INIT > 0 and self.iteration <= LOSS_ENTROPY_ANNEAL_ITERS:
            entropy_coef = LOSS_ENTROPY_COEF_INIT * (
                1.0 - (self.iteration - 1) / max(1, LOSS_ENTROPY_ANNEAL_ITERS)
            )
        if LOSS_ENTROPY_COEF_MIN > 0:
            entropy_coef = max(float(entropy_coef), float(LOSS_ENTROPY_COEF_MIN))
        sched_drift_pct = 0.0
        if TRAIN_LR_SCHED_STEPS_PER_ITER_EST > 0:
            sched_drift_pct = (
                100.0
                * (actual_update_steps - TRAIN_LR_SCHED_STEPS_PER_ITER_EST)
                / TRAIN_LR_SCHED_STEPS_PER_ITER_EST
                if actual_update_steps > 0
                else 0.0
            )
        if (
            self.iteration == 1
            and actual_update_steps > 0
            and abs(sched_drift_pct) > TRAIN_LR_SCHED_DRIFT_ADJUST_THRESHOLD
        ):
            remaining_iters = max(0, TRAIN_TOTAL_ITERATIONS - self.iteration)
            new_total = int(self.scheduler.t + remaining_iters * actual_update_steps)
            new_total = max(self.scheduler.t + 1, new_total)
            self.scheduler.set_total_steps(new_total)
            self.log.info(
                f"[LR ] adjust total_steps -> {self.scheduler.total} (iter1 measured {actual_update_steps} vs est {TRAIN_LR_SCHED_STEPS_PER_ITER_EST}, drift {sched_drift_pct:+.1f}%)"
            )

        stats.update(
            {
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "learning_rate": float(learning_rate),
                "buffer_size": buffer_size,
                "buffer_percent": buffer_pct2,
                "training_time": train_elapsed_s,
                "batches_per_sec": batches_per_sec,
                "optimizer_steps": actual_update_steps,
                "lr_sched_t": self.scheduler.t,
                "lr_sched_total": self.scheduler.total,
            }
        )

        recent_pct = int(round(100.0 * TRAIN_RECENT_SAMPLE_RATIO))
        old_pct = max(0, 100 - recent_pct)
        tr_plan_line = (
            f"plan {num_steps} | tgt {TRAIN_TARGET_TRAIN_SAMPLES_PER_NEW:.1f}x | act {samples_ratio:>3.1f}x | mix {recent_pct}/{old_pct}%"
            if num_steps > 0
            else "plan 0 skip (buffer underfilled)"
        )
        tr_line = (
            f"steps {len(losses):>3} | b/s {batches_per_sec:>4.1f} | sps {self._format_si(samples_per_sec, digits=1)} | t {self._format_time(train_elapsed_s)} | "
            f"P {avg_policy_loss:>6.4f} | V {avg_value_loss:>6.4f} | LR {learning_rate:.2e} | grad {avg_grad_norm:>5.3f} | ent {avg_entropy:>5.3f} | "
            f"entc {entropy_coef:.2e} | clip {TRAIN_GRAD_CLIP_NORM:.1f} | buf {int(buffer_pct2):>3}% ({self._format_si(buffer_size)}) | batch {self.train_batch_size}"
        )
        lr_sched_fragment = f"sched {TRAIN_LR_SCHED_STEPS_PER_ITER_EST}->{actual_update_steps} | drift {sched_drift_pct:+.0f}% | pos {self.scheduler.t}/{self._format_si(self.scheduler.total)}"

        tr_line_parts = tr_line.split(" | ")
        tr_line_step = tr_line_parts[0] if tr_line_parts else ""
        tr_line_rest = " | ".join(tr_line_parts[1:]) if len(tr_line_parts) > 1 else ""
        tr_block = (
            "[TR] " + tr_plan_line + "\n"
            + " " * 6 + tr_line_step + "\n"
            + " " * 6 + tr_line_rest + "\n"
            + " " * 6 + lr_sched_fragment
        )
        self.log.info(tr_block)
        self._prev_eval_m = eval_metrics

        return stats

    def _clone_from_ema(self) -> torch.nn.Module:
        model_clone = self._clone_model()
        if self.ema is not None:
            self.ema.copy_to(model_clone)
        return model_clone

    def _arena_match(
        self, challenger: torch.nn.Module, incumbent: torch.nn.Module
    ) -> tuple[float, int, int, int]:
        """Play a stratified arena between challenger and incumbent; returns (score, W, D, L)."""
        import chesscore as _ccore
        import numpy as _np

        from model import BatchedEvaluator as _BatchedEval

        wins = draws = losses = 0

        with (
            _BatchedEval(self.device) as challenger_eval,
            _BatchedEval(self.device) as incumbent_eval,
        ):
            challenger_eval.refresh_from(challenger)
            incumbent_eval.refresh_from(incumbent)
            challenger_eval.cache_capacity = self._arena_eval_cache_cap
            incumbent_eval.cache_capacity = self._arena_eval_cache_cap

            def play(
                evaluator_white: _BatchedEval,
                evaluator_black: _BatchedEval,
                start_fen: str,
            ) -> tuple[int, list[str]]:
                position = _ccore.Position()
                position.from_fen(start_fen)
                mcts_white = _ccore.MCTS(
                    MCTS_EVAL_SIMULATIONS,
                    MCTS_C_PUCT,
                    MCTS_DIRICHLET_ALPHA,
                    0.0 if ARENA_DETERMINISTIC else float(ARENA_DIRICHLET_WEIGHT),
                )
                mcts_white.set_c_puct_params(MCTS_C_PUCT_BASE, MCTS_C_PUCT_INIT)
                mcts_white.set_fpu_reduction(MCTS_FPU_REDUCTION)
                mcts_black = _ccore.MCTS(
                    MCTS_EVAL_SIMULATIONS,
                    MCTS_C_PUCT,
                    MCTS_DIRICHLET_ALPHA,
                    0.0 if ARENA_DETERMINISTIC else float(ARENA_DIRICHLET_WEIGHT),
                )
                mcts_black.set_c_puct_params(MCTS_C_PUCT_BASE, MCTS_C_PUCT_INIT)
                mcts_black.set_fpu_reduction(MCTS_FPU_REDUCTION)
                ply = 0
                moves_uci: list[str] = []
                while position.result() == _ccore.ONGOING and ply < GAME_MAX_PLIES:
                    visits = (
                        mcts_white.search_batched(
                            position,
                            evaluator_white.infer_positions,
                            EVAL_BATCH_SIZE_MAX,
                        )
                        if ply % 2 == 0
                        else mcts_black.search_batched(
                            position,
                            evaluator_black.infer_positions,
                            EVAL_BATCH_SIZE_MAX,
                        )
                    )
                    if visits is None:
                        break
                    visit_counts = _np.asarray(visits, dtype=_np.float64)
                    if visit_counts.size == 0:
                        break
                    legal_moves = position.legal_moves()
                    if ARENA_DETERMINISTIC:
                        idx = int(_np.argmax(visit_counts))
                    elif ply < ARENA_TEMP_MOVES:
                        temp = float(ARENA_TEMPERATURE)
                        if temp <= 0.0 + ARENA_OPENING_TEMPERATURE_EPS:
                            idx = int(_np.argmax(visit_counts))
                        else:
                            v_pos = _np.maximum(visit_counts, 0.0)
                            s0 = v_pos.sum()
                            if s0 <= 0:
                                idx = int(_np.argmax(visit_counts))
                            else:
                                probs = v_pos ** (1.0 / temp)
                                s = probs.sum()
                                idx = (
                                    int(_np.argmax(visit_counts))
                                    if s <= 0
                                    else int(_np.random.choice(len(legal_moves), p=probs / s))
                                )
                    else:
                        idx = int(_np.argmax(visit_counts))
                    mv = legal_moves[idx]
                    try:
                        mv_str = str(mv)
                        if mv_str:
                            moves_uci.append(mv_str)
                    except Exception:
                        pass
                    position.make_move(mv)
                    ply += 1
                r = position.result()
                return (
                    1 if r == _ccore.WHITE_WIN else (-1 if r == _ccore.BLACK_WIN else 0),
                    moves_uci,
                )

            pair_count = max(1, ARENA_GAMES_PER_EVAL // ARENA_PAIRING_FACTOR)
            pgn_candidates: list[dict[str, object]] = []
            for _ in range(pair_count):
                start_fen = _ccore.Position().to_fen()
                r1, mv1 = play(challenger_eval, incumbent_eval, start_fen)
                r2_raw, mv2 = play(incumbent_eval, challenger_eval, start_fen)
                r2 = -r2_raw
                pgn_candidates.append({
                    "fen": start_fen,
                    "white": "Challenger-EMA",
                    "black": "Incumbent-Best",
                    "result": r1,
                    "moves": mv1,
                })
                pgn_candidates.append({
                    "fen": start_fen,
                    "white": "Incumbent-Best",
                    "black": "Challenger-EMA",
                    "result": r2_raw,
                    "moves": mv2,
                })
                for outcome in (r1, r2):
                    if outcome > 0:
                        wins += 1
                    elif outcome < 0:
                        losses += 1
                    else:
                        draws += 1

        total_games = max(1, wins + draws + losses)
        score = (wins + ARENA_DRAW_SCORE * draws) / total_games
        try:
            from config import (
                ARENA_SAVE_PGN_ENABLE,
                ARENA_SAVE_PGN_DIR,
                ARENA_SAVE_PGN_ON_PROMOTION,
                ARENA_SAVE_PGN_SAMPLES_PER_ROUND,
            )
            if ARENA_SAVE_PGN_ENABLE and pgn_candidates:
                import os as _os
                import datetime as _dt
                _os.makedirs(ARENA_SAVE_PGN_DIR, exist_ok=True)
                iso_date = _dt.datetime.now(_dt.UTC).strftime("%Y.%m.%d")
                round_tag = f"iter-{self.iteration}-r{self._gate_rounds + 1}"
                def res_str(r: int) -> str:
                    return "1-0" if r > 0 else ("0-1" if r < 0 else "1/2-1/2")
                def has_promo(moves: list[str]) -> bool:
                    return any((len(m) >= 5 and m[-1] in ("q", "r", "b", "n")) for m in moves)
                promo = [g for g in pgn_candidates if has_promo(g.get("moves", []) or [])] if ARENA_SAVE_PGN_ON_PROMOTION else []
                saved = 0
                def write_game(g: dict[str, object], name: str) -> None:
                    nonlocal saved
                    if saved >= int(ARENA_SAVE_PGN_SAMPLES_PER_ROUND):
                        return
                    path = _os.path.join(ARENA_SAVE_PGN_DIR, f"{round_tag}_{name}_{saved+1}.pgn")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(f"[Event \"HybridChess Arena\"]\n")
                        f.write(f"[Site \"local\"]\n")
                        f.write(f"[Date \"{iso_date}\"]\n")
                        f.write(f"[Round \"{round_tag}\"]\n")
                        f.write(f"[White \"{g['white']}\"]\n")
                        f.write(f"[Black \"{g['black']}\"]\n")
                        f.write(f"[Result \"{res_str(int(g['result']))}\"]\n")
                        f.write(f"[FEN \"{g['fen']}\"]\n[SetUp \"1\"]\n")
                        mv = list(g.get("moves", []) or [])
                        out = []
                        move_no = 1
                        for i, mvs in enumerate(mv):
                            if i % 2 == 0:
                                out.append(f"{move_no}. {mvs}")
                                move_no += 1
                            else:
                                out.append(mvs)
                        f.write(" ".join(out) + f" {res_str(int(g['result']))}\n")
                    saved += 1
                for g in promo:
                    write_game(g, "promo")
                if saved < int(ARENA_SAVE_PGN_SAMPLES_PER_ROUND):
                    for g in pgn_candidates:
                        if ARENA_SAVE_PGN_ON_PROMOTION and g in promo:
                            continue
                        write_game(g, "sample")
        except Exception:
            pass
        return score, wins, draws, losses

    def train(self) -> None:
        self.log.info(self._startup_summary())

        for iteration in range(self.iteration + 1, TRAIN_TOTAL_ITERATIONS + 1):
            self.iteration = iteration
            iter_stats = self.training_iteration()

            do_eval = (
                (iteration % ARENA_EVAL_EVERY_ITERS) == 0 if ARENA_EVAL_EVERY_ITERS > 0 else False
            )
            arena_elapsed = 0.0
            arena_w = arena_d = arena_l = 0
            arena_decision = "skipped"
            arena_metrics: dict[str, float] = {}
            if do_eval:
                if (
                    (not self._gate_active)
                    or (self._gate_rounds >= ARENA_CANDIDATE_MAX_ROUNDS)
                    or (
                        (self._gate.w + self._gate.d + self._gate.losses)
                        >= ARENA_CANDIDATE_MAX_GAMES
                    )
                ):
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
                    f"[AR ] W/D/L {aw}/{ad}/{al} | n {int(m.get('n', 0))} | p {100.0 * m.get('p', 0):>5.1f}% | elo {m.get('elo', 0):>6.1f} ±{m.get('se_elo', 0):.1f} | decision {decision.upper()} | time {self._format_time(arena_elapsed)} | age_iter {iteration - self._gate_started_iter} | rounds {self._gate_rounds}"
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

                if ARENA_LOG_CSV_ENABLE:
                    try:
                        import csv
                        ar_dir = os.path.dirname(ARENA_LOG_CSV_PATH)
                        if ar_dir:
                            os.makedirs(ar_dir, exist_ok=True)
                        write_header = not os.path.isfile(ARENA_LOG_CSV_PATH)
                        fields = {
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
                            "draw_w": float(ARENA_GATE_DRAW_WEIGHT),
                            "baseline_p": float(ARENA_GATE_BASELINE_P),
                            "deterministic": bool(ARENA_DETERMINISTIC),
                            "mcts_sims": int(MCTS_EVAL_SIMULATIONS),
                        }
                        with open(ARENA_LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                            w = csv.DictWriter(f, fieldnames=list(fields.keys()))
                            if write_header:
                                w.writeheader()
                            w.writerow(fields)
                    except Exception:
                        pass

                if decision == "accept":
                    assert self._pending_challenger is not None
                    self.best_model.load_state_dict(
                        self._pending_challenger.state_dict(), strict=True
                    )
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

            if self.iteration % CHECKPOINT_SAVE_EVERY_ITERS == 0:
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
            mem_info_summary = self._get_mem_info()
            try:
                target_lo = 0.60 * self.device_total_gb
                prev_bs = int(self.train_batch_size)
                if (not TORCH_COMPILE) or TORCH_COMPILE_DYNAMIC:
                    headroom_gb = max(0.0, float(self.device_total_gb) - float(peak_res))
                    if (
                        peak_res < target_lo
                        and headroom_gb >= 8.0
                        and prev_bs < int(TRAIN_BATCH_SIZE_MAX)
                        and int(self._oom_cooldown_iters) == 0
                    ):
                        self.train_batch_size = int(min(int(TRAIN_BATCH_SIZE_MAX), prev_bs + 512))
                    elif peak_res > 0.92 * self.device_total_gb and prev_bs > int(TRAIN_BATCH_SIZE_MIN):
                        self.train_batch_size = int(max(int(TRAIN_BATCH_SIZE_MIN), prev_bs - 1024))
                    if self.train_batch_size != prev_bs:
                        self.log.info(
                            f"[AUTO] train_batch_size {prev_bs} -> {self.train_batch_size} (peak_res {self._format_gb(peak_res)})"
                        )
                if int(self._oom_cooldown_iters) > 0:
                    self._oom_cooldown_iters = int(self._oom_cooldown_iters) - 1
                else:
                    pass
            except Exception:
                pass
            try:
                import ctypes
                import gc

                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

            self.log.info(
                f"[SUM] iter {self._format_time(full_iter_time)} | sp {self._format_time(sp_time)} | tr {self._format_time(tr_time)} | ar {self._format_time(arena_elapsed)} | "
                f"elapsed {self._format_time(time.time() - self.start_time)} | next_ar {next_ar} | games {self.total_games:,} | "
                f"peak GPU {self._format_gb(peak_alloc)}/{self._format_gb(peak_res)} | RSS {self._format_gb(mem_info_summary['rss_gb'])} | batch {self.train_batch_size}"
            )

            self._prev_eval_m = self.evaluator.get_metrics()

            if METRICS_LOG_CSV_ENABLE:
                try:
                    import csv
                    eval_metrics_now = self.evaluator.get_metrics()
                    try:
                        buf_cap = int(self.selfplay_engine.get_capacity())
                    except Exception:
                        buf_cap = int(self.selfplay_engine.buffer.maxlen or 1)
                    sys_info = self._get_sys_info()
                    lr = float(iter_stats.get("learning_rate", 0.0))
                    opt_steps = int(iter_stats.get("optimizer_steps", 0))
                    train_bs = int(self.train_batch_size)
                    train_time = float(iter_stats.get("training_time", 0.0))
                    samples_per_sec = (
                        (opt_steps * train_bs) / train_time if train_time > 0 else 0.0
                    )
                    mt_dir = os.path.dirname(METRICS_LOG_CSV_PATH)
                    if mt_dir:
                        os.makedirs(mt_dir, exist_ok=True)
                    write_header = not os.path.isfile(METRICS_LOG_CSV_PATH)
                    fields = [
                        "iter",
                        "elapsed_s",
                        "next_ar",
                        # training
                        "train_batch_size",
                        "optimizer_steps",
                        "learning_rate",
                        "policy_loss",
                        "value_loss",
                        "batches_per_sec",
                        "samples_per_sec",
                        "lr_sched_t",
                        "lr_sched_total",
                        # selfplay
                        "sp_games",
                        "sp_white_wins",
                        "sp_draws",
                        "sp_black_wins",
                        "sp_gpm",
                        "sp_mps_k",
                        "sp_avg_len",
                        "sp_new_moves",
                        "selfplay_time",
                        # replay buffer
                        "buffer_size",
                        "buffer_capacity",
                        "buffer_percent",
                        # evaluator
                        "eval_requests_total",
                        "eval_cache_hits_total",
                        "eval_hit_rate",
                        "eval_batches_total",
                        "eval_positions_total",
                        "eval_batch_size_max",
                        "eval_batch_cap",
                        "eval_coalesce_ms",
                        # arena
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
                        # memory/system
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
                        # training
                        "train_batch_size": train_bs,
                        "optimizer_steps": opt_steps,
                        "learning_rate": lr,
                        "policy_loss": float(iter_stats.get("policy_loss", 0.0)),
                        "value_loss": float(iter_stats.get("value_loss", 0.0)),
                        "batches_per_sec": float(iter_stats.get("batches_per_sec", 0.0)),
                        "samples_per_sec": float(samples_per_sec),
                        "lr_sched_t": int(iter_stats.get("lr_sched_t", 0)),
                        "lr_sched_total": int(iter_stats.get("lr_sched_total", 0)),
                        # selfplay
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
                        # replay buffer
                        "buffer_size": int(iter_stats.get("buffer_size", len(self.selfplay_engine.buffer))),
                        "buffer_capacity": int(buf_cap),
                        "buffer_percent": float(iter_stats.get("buffer_percent", 100.0 * len(self.selfplay_engine.buffer) / max(1, buf_cap))),
                        # evaluator
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
                        # arena
                        "arena_ran": int(1 if do_eval else 0),
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
                        # memory/system
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
                    with open(METRICS_LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        if write_header:
                            w.writeheader()
                        w.writerow(row)
                except Exception:
                    pass


class EloGater:
    """Statistical gate to accept/reject new models based on match score."""

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

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256"
    )
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(2 * 1024 * 1024 * 1024))  # 2 GiB
    if SEED != 0:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")

    root = logging.getLogger()
    log_level = getattr(logging, str(LOG_LEVEL).upper(), logging.INFO)
    root.setLevel(log_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(stdout_handler)
    if LOG_TO_FILE:
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE_PATH, mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        root.addHandler(file_handler)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this training pipeline (A100 expected).")
    torch.set_float32_matmul_precision(TORCH_MATMUL_FLOAT32_PRECISION)
    torch.backends.cuda.matmul.allow_tf32 = bool(TORCH_ALLOW_TF32 and (SEED == 0))
    torch.backends.cudnn.allow_tf32 = bool(TORCH_ALLOW_TF32 and (SEED == 0))
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = bool(TORCH_CUDNN_BENCHMARK and (SEED == 0))
    if SEED != 0:
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(TORCH_THREADS_INTRA)
    torch.set_num_interop_threads(TORCH_THREADS_INTER)
    import random as _py_random

    if SEED != 0:
        _py_random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    resume_flag = any(a in ("--resume", "resume") for a in sys.argv[1:])
    Trainer(resume=resume_flag).train()
