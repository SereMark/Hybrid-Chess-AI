from __future__ import annotations

import csv
import ctypes
import gc
import logging
import math
import os
import random
import time
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import config as C
from model import BatchedEvaluator, ChessNet
from selfplay import SelfPlayEngine


class Trainer:
    """Coordinates self-play generation, training updates, and model gating."""

    def __init__(self, device: str | None = None, resume: bool = False) -> None:
        self.log = logging.getLogger("hybridchess.trainer")
        device_obj = torch.device(device or "cuda")
        self.device = device_obj
        net_any: Any = ChessNet().to(self.device)
        if C.TORCH.MODEL_CHANNELS_LAST:
            self.model = net_any.to(memory_format=torch.channels_last)
        else:
            self.model = net_any
        self._compiled = False
        try:
            if C.TORCH.COMPILE:
                self.model = torch.compile(
                    self.model,
                    mode=C.TORCH.COMPILE_MODE,
                    fullgraph=C.TORCH.COMPILE_FULLGRAPH,
                    dynamic=C.TORCH.COMPILE_DYNAMIC,
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
                {"params": decay, "weight_decay": C.TRAIN.WEIGHT_DECAY},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=C.TRAIN.LR_INIT,
            momentum=C.TRAIN.MOMENTUM,
            nesterov=True,
        )

        total_expected_train_steps = int(C.TRAIN.TOTAL_ITERATIONS * C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST)
        warmup_steps_clamped = int(max(1, min(C.TRAIN.LR_WARMUP_STEPS, max(1, total_expected_train_steps - 1))))

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
            C.TRAIN.LR_INIT,
            warmup_steps_clamped,
            C.TRAIN.LR_FINAL,
            total_expected_train_steps,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=C.TORCH.AMP_ENABLED)

        self.evaluator = BatchedEvaluator(self.device)
        self._eval_batch_cap = int(C.EVAL.BATCH_SIZE_MAX)
        self._eval_coalesce_ms = int(C.EVAL.COALESCE_MS)
        try:
            self.evaluator.set_batching_params(self._eval_batch_cap, self._eval_coalesce_ms)
        except Exception:
            pass
        self.train_batch_size: int = int(C.TRAIN.BATCH_SIZE)
        self._current_eval_cache_cap: int = int(C.EVAL.CACHE_CAPACITY)
        self._current_replay_cap: int = int(C.REPLAY.BUFFER_CAPACITY)
        self._arena_eval_cache_cap: int = int(C.EVAL.ARENA_CACHE_CAPACITY)
        self._oom_cooldown_iters: int = 0

        self._proc = psutil.Process(os.getpid())
        psutil.cpu_percent(0.0)
        self._proc.cpu_percent(0.0)

        class EMA:
            """Lightweight EMA of model parameters for evaluation and gating."""

            def __init__(self, model: torch.nn.Module, decay: float | None = None) -> None:
                self.decay = float(C.TRAIN.EMA_DECAY if decay is None else decay)
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
            self._try_resume()
        else:
            try:
                if C.LOG.METRICS_LOG_CSV_ENABLE and os.path.isfile(C.LOG.METRICS_LOG_CSV_PATH):
                    os.remove(C.LOG.METRICS_LOG_CSV_PATH)
            except Exception:
                pass

    def _startup_summary(self) -> str:
        props = torch.cuda.get_device_properties(self.device)
        sm = f"{props.major}.{props.minor}"
        mp = int(getattr(props, "multi_processor_count", 0))
        autocast_mode = "fp16" if C.TORCH.AMP_ENABLED else "off"
        tf32_effective = bool(torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32)
        total_params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        env_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
        env_modload = os.environ.get("CUDA_MODULE_LOADING", "default")
        env_cache = os.environ.get("CUDA_CACHE_MAXSIZE", "default")

        sections: list[str] = []
        sections.append(
            f"[GPU     ] {self.device_name} (device {self.device}) | SM {sm} | VRAM {self.device_total_gb:.1f}G | MPs {mp}"
        )
        sections.append(
            f"[PyTorch ] {torch.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()} | compile={'on' if self._compiled else 'off'}(mode={C.TORCH.COMPILE_MODE}, full={C.TORCH.COMPILE_FULLGRAPH}, dyn={C.TORCH.COMPILE_DYNAMIC})"
        )
        sections.append(
            f"[Precision] autocast={autocast_mode} | TF32={'on' if tf32_effective else 'off'} | matmul={torch.get_float32_matmul_precision()} | GradScaler={'on' if C.TORCH.AMP_ENABLED else 'off'}"
        )
        sections.append(
            f"[Memory  ] channels_last train={C.TORCH.MODEL_CHANNELS_LAST} eval={C.TORCH.EVAL_MODEL_CHANNELS_LAST} | pin_memory train={C.TORCH.TRAIN_PIN_MEMORY} eval={C.TORCH.EVAL_PIN_MEMORY}"
        )
        sections.append(f"[Allocator] {env_alloc} | cuda_module_loading={env_modload} | cuda_cache_max={env_cache}")
        try:
            sp_workers = int(self.selfplay_engine.get_num_workers())
        except Exception:
            sp_workers = int(C.SELFPLAY.NUM_WORKERS)
        sections.append(
            f"[Threads ] torch intra={torch.get_num_threads()} inter={torch.get_num_interop_threads()} | CPU cores={os.cpu_count()} | selfplay workers={sp_workers}"
        )
        sections.append(
            f"[Model   ] params={total_params_m:.1f}M | blocks={C.MODEL.BLOCKS} | channels={C.MODEL.CHANNELS} | value_head=({C.MODEL.VALUE_CONV_CHANNELS}->{C.MODEL.VALUE_HIDDEN_DIM})"
        )
        sections.append(f"[Replay  ] capacity={C.REPLAY.BUFFER_CAPACITY:,}")
        sections.append(
            f"[SelfPlay] sims {C.MCTS.TRAIN_SIMULATIONS_BASE}→≥{C.MCTS.TRAIN_SIMULATIONS_MIN} | temp {C.SELFPLAY.TEMP_HIGH}->{C.SELFPLAY.TEMP_LOW}@{C.SELFPLAY.TEMP_MOVES} | resign {C.RESIGN.VALUE_THRESHOLD} x{C.RESIGN.CONSECUTIVE_PLIES} | dirichlet alpha={C.MCTS.DIRICHLET_ALPHA} w={C.MCTS.DIRICHLET_WEIGHT}"
        )
        sections.append(
            f"[Train   ] iters={C.TRAIN.TOTAL_ITERATIONS} | games/iter={C.TRAIN.GAMES_PER_ITER} | batch={self.train_batch_size} | sched_est={C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST}"
        )
        sections.append(
            f"[LR      ] init={C.TRAIN.LR_INIT:.2e} | warmup={self.scheduler.warm} | final={C.TRAIN.LR_FINAL:.2e} | weight_decay={C.TRAIN.WEIGHT_DECAY} | momentum={C.TRAIN.MOMENTUM}"
        )
        sections.append(
            f"[Loss    ] policy={C.TRAIN.LOSS_POLICY_WEIGHT} | value={C.TRAIN.LOSS_VALUE_WEIGHT}->{C.TRAIN.LOSS_VALUE_WEIGHT_LATE}@{C.TRAIN.LOSS_VALUE_WEIGHT_SWITCH_ITER} | smooth={C.TRAIN.LOSS_POLICY_LABEL_SMOOTH:.2f} | entropy={C.TRAIN.LOSS_ENTROPY_COEF_INIT:.2e}→{C.TRAIN.LOSS_ENTROPY_COEF_MIN:.2e}@{C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS}"
        )
        sections.append(
            f"[Augment ] mirror={C.AUGMENT.MIRROR_PROB:.2f} | rot180={C.AUGMENT.ROT180_PROB:.2f} | vflip_cs={C.AUGMENT.VFLIP_CS_PROB:.2f}"
        )
        sections.append(
            f"[EMA     ] {'on' if C.TRAIN.EMA_ENABLED else 'off'} | decay={C.TRAIN.EMA_DECAY if C.TRAIN.EMA_ENABLED else 0}"
        )
        sections.append(
            f"[Eval    ] batch_max={C.EVAL.BATCH_SIZE_MAX} @ {C.EVAL.COALESCE_MS}ms | cache={C.EVAL.CACHE_CAPACITY:,} | dtype={'fp16' if C.EVAL.CACHE_USE_FP16 else 'fp32'}"
        )
        sections.append(
            f"[Arena   ] every={C.ARENA.EVAL_EVERY_ITERS} | games={C.ARENA.GAMES_PER_EVAL} | baseline_p={100.0 * C.ARENA.GATE_BASELINE_P:.0f}% | Z {C.ARENA.GATE_Z_EARLY}->{C.ARENA.GATE_Z_LATE} (switch@{C.ARENA.GATE_Z_SWITCH_ITER}) | min_games={C.ARENA.GATE_MIN_GAMES} | det={'on' if C.ARENA.DETERMINISTIC else 'off'}"
        )
        sections.append(f"[Expect  ] total_games={C.TRAIN.TOTAL_ITERATIONS * C.TRAIN.GAMES_PER_ITER:,}")
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
            tmp_path = f"{C.LOG.CHECKPOINT_FILE_PATH}.tmp"
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, C.LOG.CHECKPOINT_FILE_PATH)
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
            tmp_path = f"{C.LOG.BEST_MODEL_FILE_PATH}.tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, C.LOG.BEST_MODEL_FILE_PATH)
            self.log.info("[BEST] saved best model")
        except Exception as e:
            self.log.warning(f"Failed to save best model: {e}")

    def _try_resume(self) -> None:
        path = C.LOG.CHECKPOINT_FILE_PATH
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
            if os.path.isfile(C.LOG.BEST_MODEL_FILE_PATH):
                best_ckpt = torch.load(C.LOG.BEST_MODEL_FILE_PATH, map_location="cpu")
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
        x_u8_t = torch.from_numpy(x_u8_np)
        if C.TORCH.TRAIN_PIN_MEMORY:
            try:
                x_u8_t = x_u8_t.pin_memory()
            except Exception:
                pass
        x = x_u8_t.to(self.device, non_blocking=True)
        pi_u16_np = np.stack(counts_u16_list).astype(np.uint16, copy=False)
        pi_u16_t = torch.from_numpy(pi_u16_np)
        if C.TORCH.TRAIN_PIN_MEMORY:
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
        if C.TORCH.TRAIN_PIN_MEMORY:
            try:
                v_i8_t = v_i8_t.pin_memory()
            except Exception:
                pass
        v_target = v_i8_t.to(self.device, non_blocking=True).to(dtype=torch.float32) / float(C.DATA.VALUE_I8_SCALE)
        x = x.to(dtype=torch.float32) / float(C.DATA.U8_SCALE)
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
            self._aug_vflip_plane_perm = torch.tensor(plane_perm, dtype=torch.long, device=self.device)
        if np.random.rand() < C.AUGMENT.MIRROR_PROB:
            x = torch.flip(x, dims=[-1])
            pi_target = pi_target.index_select(1, self._aug_mirror_idx)
        if np.random.rand() < C.AUGMENT.ROT180_PROB:
            x = torch.flip(x, dims=[-1, -2])
            pi_target = pi_target.index_select(1, self._aug_rot180_idx)
        if np.random.rand() < C.AUGMENT.VFLIP_CS_PROB:
            x = torch.flip(x, dims=[-2])
            x = x.index_select(1, self._aug_vflip_plane_perm)
            if 0 <= getattr(self, "_turn_plane_idx", x.shape[1]) < x.shape[1]:
                x[:, self._turn_plane_idx] = 1.0 - x[:, self._turn_plane_idx]
            pi_target = pi_target.index_select(1, self._aug_vflip_idx)
            v_target = -v_target
        self.model.train()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=C.TORCH.AMP_ENABLED,
        ):
            policy_logits, value_pred = self.model(x)
            value_pred = value_pred.squeeze(-1)
            if C.TRAIN.LOSS_POLICY_LABEL_SMOOTH > 0.0:
                num_actions = pi_target.shape[1]
                pi_smooth = (1.0 - C.TRAIN.LOSS_POLICY_LABEL_SMOOTH) * pi_target + (
                    C.TRAIN.LOSS_POLICY_LABEL_SMOOTH / num_actions
                )
            else:
                pi_smooth = pi_target
            policy_loss = F.kl_div(F.log_softmax(policy_logits, dim=1), pi_smooth, reduction="batchmean")
            value_loss = F.mse_loss(value_pred, v_target)
            entropy_coef = 0.0
            if C.TRAIN.LOSS_ENTROPY_COEF_INIT > 0 and self.iteration <= C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS:
                entropy_coef = C.TRAIN.LOSS_ENTROPY_COEF_INIT * (
                    1.0 - (self.iteration - 1) / max(1, C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS)
                )
            if C.TRAIN.LOSS_ENTROPY_COEF_MIN > 0:
                entropy_coef = max(float(entropy_coef), float(C.TRAIN.LOSS_ENTROPY_COEF_MIN))
            entropy = (-(F.softmax(policy_logits, dim=1) * F.log_softmax(policy_logits, dim=1)).sum(dim=1)).mean()
            value_loss_weight = (
                C.TRAIN.LOSS_VALUE_WEIGHT_LATE
                if self.iteration >= C.TRAIN.LOSS_VALUE_WEIGHT_SWITCH_ITER
                else C.TRAIN.LOSS_VALUE_WEIGHT
            )
            total_loss = (
                C.TRAIN.LOSS_POLICY_WEIGHT * policy_loss + value_loss_weight * value_loss - entropy_coef * entropy
            )
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), C.TRAIN.GRAD_CLIP_NORM)
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
        if self.iteration < int(C.RESIGN.DISABLE_UNTIL_ITERS):
            self.selfplay_engine.resign_consecutive = 0
        else:
            self.selfplay_engine.resign_consecutive = max(
                int(C.RESIGN.CONSECUTIVE_PLIES), int(C.RESIGN.CONSECUTIVE_MIN)
            )
        if self.iteration == C.ARENA.GATE_Z_SWITCH_ITER:
            self._gate.z = float(C.ARENA.GATE_Z_LATE)
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
            if cpu_pct < 55.0 and buffer_pct < 60.0 and cur_workers < max_workers_cap:
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
        pct_done = 100.0 * (self.iteration - 1) / max(1, C.TRAIN.TOTAL_ITERATIONS)
        self.log.info(
            f"[ITR {self.iteration:>3}/{C.TRAIN.TOTAL_ITERATIONS} {pct_done:>4.1f}%] "
            f"LRnext {header_lr:.2e} | t {self._format_time(total_elapsed)} | "
            f"buf {self._format_si(buffer_len)}/{self._format_si(buffer_cap)} ({int(buffer_pct)}%) | "
            f"GPU {self._format_gb(mem_info['allocated_gb'])}/{self._format_gb(mem_info['reserved_gb'])}/{self._format_gb(mem_info['total_gb'])} | "
            f"RSS {self._format_gb(mem_info['rss_gb'])} | CPU {sys_info['cpu_sys_pct']:.0f}/{sys_info['cpu_proc_pct']:.0f}% | "
            f"RAM {self._format_gb(sys_info['ram_used_gb'])}/{self._format_gb(sys_info['ram_total_gb'])} ({sys_info['ram_pct']:.0f}%) | "
            f"load {sys_info['load1']:.2f}"
        )

        t0 = time.time()
        game_stats = self.selfplay_engine.play_games(C.TRAIN.GAMES_PER_ITER)
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
        delta_eval_positions = int(eval_positions_total - int(prev_metrics.get("eval_positions_total", 0)))
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
        if (C.LOG.CUDA_EMPTY_CACHE_EVERY_ITERS > 0) and (self.iteration % C.LOG.CUDA_EMPTY_CACHE_EVERY_ITERS == 0):
            torch.cuda.empty_cache()

        losses: list[tuple[torch.Tensor, torch.Tensor]] = []
        snapshot = self.selfplay_engine.snapshot()
        min_batch_samples = max(1, self.train_batch_size // 2)
        new_examples = int(game_stats.get("moves", 0))
        target_train_samples = int(C.TRAIN.TARGET_TRAIN_SAMPLES_PER_NEW * max(1, new_examples))
        num_steps = int(np.ceil(target_train_samples / max(1, self.train_batch_size)))
        num_steps = max(C.TRAIN.UPDATE_STEPS_MIN, min(C.TRAIN.UPDATE_STEPS_MAX, num_steps))
        samples_ratio = 0.0
        if len(snapshot) < min_batch_samples:
            num_steps = 0
        else:
            samples_ratio = (num_steps * self.train_batch_size) / max(1, new_examples)

        grad_norm_sum: float = 0.0
        entropy_sum: float = 0.0
        for _step in range(num_steps):
            batch = self.selfplay_engine.sample_from_snapshot(
                snapshot, self.train_batch_size, recent_ratio=C.SAMPLING.TRAIN_RECENT_SAMPLE_RATIO
            )
            if not batch:
                continue
            states, policies, values = batch
            try:
                policy_loss_t, value_loss_t, grad_norm_val, pred_entropy = self.train_step((states, policies, values))
                grad_norm_sum += float(grad_norm_val)
                entropy_sum += float(pred_entropy)
                losses.append((policy_loss_t, value_loss_t))
            except torch.OutOfMemoryError:
                prev_bs_local = int(self.train_batch_size)
                new_bs_local = int(max(int(C.TRAIN.BATCH_SIZE_MIN), prev_bs_local - 1024))
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
            float(torch.stack([pair[0] for pair in losses]).mean().detach().cpu()) if losses else float("nan")
        )
        avg_value_loss = (
            float(torch.stack([pair[1] for pair in losses]).mean().detach().cpu()) if losses else float("nan")
        )
        buffer_size = len(self.selfplay_engine.buffer)
        try:
            buffer_capacity2 = int(self.selfplay_engine.get_capacity())
        except Exception:
            buffer_capacity2 = int(self.selfplay_engine.buffer.maxlen or 1)
        buffer_pct2 = (buffer_size / buffer_capacity2) * 100
        batches_per_sec = (len(losses) / max(1e-9, train_elapsed_s)) if losses else 0.0
        samples_per_sec = ((len(losses) * self.train_batch_size) / max(1e-9, train_elapsed_s)) if losses else 0.0
        learning_rate = self.optimizer.param_groups[0]["lr"]
        avg_grad_norm = (grad_norm_sum / max(1, len(losses))) if losses else 0.0
        avg_entropy = (entropy_sum / max(1, len(losses))) if losses else 0.0
        entropy_coef = 0.0
        if C.TRAIN.LOSS_ENTROPY_COEF_INIT > 0 and self.iteration <= C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS:
            entropy_coef = C.TRAIN.LOSS_ENTROPY_COEF_INIT * (
                1.0 - (self.iteration - 1) / max(1, C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS)
            )
        if C.TRAIN.LOSS_ENTROPY_COEF_MIN > 0:
            entropy_coef = max(float(entropy_coef), float(C.TRAIN.LOSS_ENTROPY_COEF_MIN))
        sched_drift_pct = 0.0
        if C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST > 0:
            sched_drift_pct = (
                100.0
                * (actual_update_steps - C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST)
                / C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST
                if actual_update_steps > 0
                else 0.0
            )
        if (
            self.iteration == 1
            and actual_update_steps > 0
            and abs(sched_drift_pct) > C.TRAIN.LR_SCHED_DRIFT_ADJUST_THRESHOLD
        ):
            remaining_iters = max(0, C.TRAIN.TOTAL_ITERATIONS - self.iteration)
            new_total = int(self.scheduler.t + remaining_iters * actual_update_steps)
            new_total = max(self.scheduler.t + 1, new_total)
            self.scheduler.set_total_steps(new_total)
            self.log.info(
                f"[LR ] adjust total_steps -> {self.scheduler.total} (iter1 measured {actual_update_steps} vs est {C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST}, drift {sched_drift_pct:+.1f}%)"
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

        recent_pct = round(100.0 * C.SAMPLING.TRAIN_RECENT_SAMPLE_RATIO)
        old_pct = max(0, 100 - recent_pct)
        tr_plan_line = (
            f"plan {num_steps} | tgt {C.TRAIN.TARGET_TRAIN_SAMPLES_PER_NEW:.1f}x | act {samples_ratio:>3.1f}x | mix {recent_pct}/{old_pct}%"
            if num_steps > 0
            else "plan 0 skip (buffer underfilled)"
        )
        tr_line = (
            f"steps {len(losses):>3} | b/s {batches_per_sec:>4.1f} | sps {self._format_si(samples_per_sec, digits=1)} | t {self._format_time(train_elapsed_s)} | "
            f"P {avg_policy_loss:>6.4f} | V {avg_value_loss:>6.4f} | LR {learning_rate:.2e} | grad {avg_grad_norm:>5.3f} | ent {avg_entropy:>5.3f} | "
            f"entc {entropy_coef:.2e} | clip {C.TRAIN.GRAD_CLIP_NORM:.1f} | buf {int(buffer_pct2):>3}% ({self._format_si(buffer_size)}) | batch {self.train_batch_size}"
        )
        lr_sched_fragment = f"sched {C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST}->{actual_update_steps} | drift {sched_drift_pct:+.0f}% | pos {self.scheduler.t}/{self._format_si(self.scheduler.total)}"

        tr_line_parts = tr_line.split(" | ")
        tr_line_step = tr_line_parts[0] if tr_line_parts else ""
        tr_line_rest = " | ".join(tr_line_parts[1:]) if len(tr_line_parts) > 1 else ""
        tr_block = (
            "[TR] "
            + tr_plan_line
            + "\n"
            + " " * 6
            + tr_line_step
            + "\n"
            + " " * 6
            + tr_line_rest
            + "\n"
            + " " * 6
            + lr_sched_fragment
        )
        self.log.info(tr_block)
        self._prev_eval_m = eval_metrics

        return stats

    def _clone_from_ema(self) -> torch.nn.Module:
        model_clone = self._clone_model()
        if self.ema is not None:
            self.ema.copy_to(model_clone)
        return model_clone

    def _arena_match(self, challenger: torch.nn.Module, incumbent: torch.nn.Module) -> tuple[float, int, int, int]:
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
                    C.ARENA.MCTS_EVAL_SIMULATIONS,
                    C.MCTS.C_PUCT,
                    C.MCTS.DIRICHLET_ALPHA,
                    0.0 if C.ARENA.DETERMINISTIC else float(C.ARENA.DIRICHLET_WEIGHT),
                )
                mcts_white.set_c_puct_params(C.MCTS.C_PUCT_BASE, C.MCTS.C_PUCT_INIT)
                mcts_white.set_fpu_reduction(C.MCTS.FPU_REDUCTION)
                mcts_black = _ccore.MCTS(
                    C.ARENA.MCTS_EVAL_SIMULATIONS,
                    C.MCTS.C_PUCT,
                    C.MCTS.DIRICHLET_ALPHA,
                    0.0 if C.ARENA.DETERMINISTIC else float(C.ARENA.DIRICHLET_WEIGHT),
                )
                mcts_black.set_c_puct_params(C.MCTS.C_PUCT_BASE, C.MCTS.C_PUCT_INIT)
                mcts_black.set_fpu_reduction(C.MCTS.FPU_REDUCTION)
                ply = 0
                moves_uci: list[str] = []
                while position.result() == _ccore.ONGOING and ply < C.SELFPLAY.GAME_MAX_PLIES:
                    visits = (
                        mcts_white.search_batched(
                            position,
                            evaluator_white.infer_positions,
                            C.EVAL.BATCH_SIZE_MAX,
                        )
                        if ply % 2 == 0
                        else mcts_black.search_batched(
                            position,
                            evaluator_black.infer_positions,
                            C.EVAL.BATCH_SIZE_MAX,
                        )
                    )
                    if visits is None:
                        break
                    visit_counts = _np.asarray(visits, dtype=_np.float64)
                    if visit_counts.size == 0:
                        break
                    legal_moves = position.legal_moves()
                    if C.ARENA.DETERMINISTIC:
                        idx = int(_np.argmax(visit_counts))
                    elif ply < C.ARENA.TEMP_MOVES:
                        temp = float(C.ARENA.TEMPERATURE)
                        if temp <= 0.0 + C.ARENA.OPENING_TEMPERATURE_EPS:
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

                    try:
                        mcts_white.advance_root(position, mv)
                        mcts_black.advance_root(position, mv)
                    except Exception:
                        pass
                    ply += 1
                r = position.result()
                return (
                    1 if r == _ccore.WHITE_WIN else (-1 if r == _ccore.BLACK_WIN else 0),
                    moves_uci,
                )

            pair_count = max(1, C.ARENA.GAMES_PER_EVAL // C.ARENA.PAIRING_FACTOR)
            pgn_candidates: list[dict[str, object]] = []
            for _ in range(pair_count):
                start_fen = _ccore.Position().to_fen()
                r1, mv1 = play(challenger_eval, incumbent_eval, start_fen)
                r2_raw, mv2 = play(incumbent_eval, challenger_eval, start_fen)
                r2 = -r2_raw
                pgn_candidates.append(
                    {
                        "fen": start_fen,
                        "white": "Challenger-EMA",
                        "black": "Incumbent-Best",
                        "result": r1,
                        "moves": mv1,
                    }
                )
                pgn_candidates.append(
                    {
                        "fen": start_fen,
                        "white": "Incumbent-Best",
                        "black": "Challenger-EMA",
                        "result": r2_raw,
                        "moves": mv2,
                    }
                )
                for outcome in (r1, r2):
                    if outcome > 0:
                        wins += 1
                    elif outcome < 0:
                        losses += 1
                    else:
                        draws += 1

        total_games = max(1, wins + draws + losses)
        score = (wins + C.ARENA.DRAW_SCORE * draws) / total_games
        try:
            if C.LOG.ARENA_SAVE_PGN_ENABLE and pgn_candidates:
                import datetime as _dt
                import os as _os

                iso_date = _dt.datetime.now(_dt.UTC).strftime("%Y.%m.%d")
                round_tag = f"iter-{self.iteration}-r{self._gate_rounds + 1}"

                from typing import Any, TypedDict, cast

                def _as_int(x: object) -> int:
                    try:
                        return int(cast(Any, x))
                    except Exception:
                        return 0

                def _as_str_list(x: object) -> list[str]:
                    try:
                        if isinstance(x, list):
                            return [str(i) for i in x]
                        return []
                    except Exception:
                        return []

                class _PGNCandidate(TypedDict):
                    fen: str
                    white: str
                    black: str
                    result: int
                    moves: list[str]

                pgn_candidates_typed: list[_PGNCandidate] = []
                for _g in pgn_candidates:
                    try:
                        pgn_candidates_typed.append(
                            _PGNCandidate(
                                fen=str(_g.get("fen", "")),
                                white=str(_g.get("white", "")),
                                black=str(_g.get("black", "")),
                                result=_as_int(_g.get("result", 0)),
                                moves=_as_str_list(_g.get("moves", []) or []),
                            )
                        )
                    except Exception:
                        continue

                def res_str(r: int) -> str:
                    return "1-0" if r > 0 else ("0-1" if r < 0 else "1/2-1/2")

                def has_promo(moves: list[str]) -> bool:
                    return any((len(m) >= 5 and m[-1] in ("q", "r", "b", "n")) for m in moves)

                promo: list[_PGNCandidate] = (
                    [g for g in pgn_candidates_typed if has_promo(g["moves"])]
                    if C.LOG.ARENA_SAVE_PGN_ON_PROMOTION
                    else []
                )
                saved = 0

                def write_game(g: _PGNCandidate, name: str) -> None:
                    nonlocal saved
                    if saved >= int(C.LOG.ARENA_SAVE_PGN_SAMPLES_PER_ROUND):
                        return
                    path = f"{round_tag}_{name}_{saved+1}.pgn"
                    with open(path, "w", encoding="utf-8") as f:
                        f.write('[Event "HybridChess Arena"]\n')
                        f.write('[Site "local"]\n')
                        f.write(f'[Date "{iso_date}"]\n')
                        f.write(f'[Round "{round_tag}"]\n')
                        f.write(f"[White \"{g['white']}\"]\n")
                        f.write(f"[Black \"{g['black']}\"]\n")
                        f.write(f"[Result \"{res_str(int(g['result']))}\"]\n")
                        f.write(f"[FEN \"{g['fen']}\"]\n[SetUp \"1\"]\n")
                        mv = list(g["moves"])
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
                if saved < int(C.LOG.ARENA_SAVE_PGN_SAMPLES_PER_ROUND):
                    for g in pgn_candidates_typed:
                        if C.LOG.ARENA_SAVE_PGN_ON_PROMOTION and g in promo:
                            continue
                        write_game(g, "sample")
        except Exception:
            pass
        return score, wins, draws, losses

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
                self.log.info(f"[AR ] skipped | games 0 | time {self._format_time(arena_elapsed)}")

            if self.iteration % C.LOG.CHECKPOINT_SAVE_EVERY_ITERS == 0:
                self._save_checkpoint()

            sp_time = float(iter_stats.get("selfplay_time", 0.0))
            tr_time = float(iter_stats.get("training_time", 0.0))
            full_iter_time = sp_time + tr_time + arena_elapsed
            next_ar = 0
            if C.ARENA.EVAL_EVERY_ITERS > 0:
                k = C.ARENA.EVAL_EVERY_ITERS
                r = self.iteration % k
                next_ar = (k - r) if r != 0 else k
            peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1024**3
            peak_res = torch.cuda.max_memory_reserved(self.device) / 1024**3
            mem_info_summary = self._get_mem_info()
            try:
                target_lo = 0.60 * self.device_total_gb
                prev_bs = int(self.train_batch_size)
                if (not C.TORCH.COMPILE) or C.TORCH.COMPILE_DYNAMIC:
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
                            f"[AUTO] train_batch_size {prev_bs} -> {self.train_batch_size} (peak_res {self._format_gb(peak_res)})"
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

            self.log.info(
                f"[SUM] iter {self._format_time(full_iter_time)} | sp {self._format_time(sp_time)} | tr {self._format_time(tr_time)} | ar {self._format_time(arena_elapsed)} | "
                f"elapsed {self._format_time(time.time() - self.start_time)} | next_ar {next_ar} | games {self.total_games:,} | "
                f"peak GPU {self._format_gb(peak_alloc)}/{self._format_gb(peak_res)} | RSS {self._format_gb(mem_info_summary['rss_gb'])} | batch {self.train_batch_size}"
            )

            self._prev_eval_m = self.evaluator.get_metrics()

            if C.LOG.METRICS_LOG_CSV_ENABLE:
                try:

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
                        "buffer_size": int(iter_stats.get("buffer_size", len(self.selfplay_engine.buffer))),
                        "buffer_capacity": int(buf_cap),
                        "buffer_percent": float(
                            iter_stats.get("buffer_percent", 100.0 * len(self.selfplay_engine.buffer) / max(1, buf_cap))
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


class EloGater:
    """Statistical gate to accept/reject new models based on match score."""

    def __init__(
        self,
        z: float = C.ARENA.GATE_Z_LATE,
        min_games: int = C.ARENA.GATE_MIN_GAMES,
        draw_w: float = C.ARENA.DRAW_SCORE,
        baseline_p: float = C.ARENA.GATE_BASELINE_P,
        decisive_secondary: bool = C.ARENA.GATE_DECISIVE_SECONDARY,
        min_decisive: int = C.ARENA.GATE_MIN_DECISIVES,
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
        eps = C.ARENA.GATE_EPS
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
