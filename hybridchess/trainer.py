from __future__ import annotations

import os
import time
import csv
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from .model import (
    BLOCKS,
    CHANNELS,
    EVAL_BATCH_TIMEOUT_MS,
    EVAL_CACHE_SIZE,
    EVAL_MAX_BATCH,
    BatchedEvaluator,
    ChessNet,
)
from .selfplay import (
    BUFFER_SIZE,
    C_PUCT,
    C_PUCT_BASE,
    C_PUCT_INIT,
    DIRICHLET_ALPHA,
    DIRICHLET_WEIGHT,
    MAX_GAME_MOVES,
    MCTS_MIN_SIMS,
    RESIGN_CONSECUTIVE,
    RESIGN_THRESHOLD,
    SELFPLAY_WORKERS,
    SIMULATIONS_TRAIN,
    TEMP_HIGH,
    TEMP_LOW,
    TEMP_MOVES,
    Augment,
    SelfPlayEngine,
)

BATCH_SIZE = 1792
LR_INIT = 1.0e-2
LR_WARMUP_STEPS = 3_000
LR_FINAL = 3.0e-4
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
GRAD_CLIP = 1.0
POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 0.5
VALUE_WEIGHT_LATE = 1.0
VALUE_WEIGHT_SWITCH_ITER = 60
ITERATIONS = 600
GAMES_PER_ITER = 220
LR_SCHED_STEPS_PER_ITER_EST = 100
RATIO_TARGET_TRAIN_PER_NEW = 6.0
RATIO_UPDATE_STEPS_MIN = 48
RATIO_UPDATE_STEPS_MAX = 224
AUGMENT_MIRROR_PROB = 0.5
AUGMENT_ROT180_PROB = 0.25
AUGMENT_VFLIP_CS_PROB = 0.25
SIMULATIONS_EVAL = 64
ARENA_EVAL_EVERY = 20
ARENA_EVAL_CACHE_CAP = 8192
ARENA_GAMES = 160
ARENA_TEMPERATURE = 0.0
ARENA_TEMP_MOVES = 0
ARENA_DIRICHLET_WEIGHT = 0.0
ARENA_OPENING_TEMPERATURE_EPS = 1e-6
ARENA_DRAW_SCORE = 0.5
POLICY_LABEL_SMOOTH = 0.05
ENTROPY_COEF_INIT = 5.0e-4
ENTROPY_ANNEAL_ITERS = 60
EMA_ENABLED = True
EMA_DECAY = 0.9995
OUTPUT_DIR = "out"


class Trainer:
    def __init__(self, device: str | torch.device | None = None) -> None:
        device = torch.device(device or "cuda")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA device required (accepts 'cuda' or 'cuda:N')")
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        m_any: Any = ChessNet().to(self.device)
        self.model = m_any.to(memory_format=torch.channels_last)
        self._compiled = True
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True, dynamic=False)
        except Exception as e:
            self._compiled = False
            print(f"Warning: torch.compile failed or unavailable: {e}")

        decay: list[torch.nn.Parameter] = []
        nodecay: list[torch.nn.Parameter] = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (nodecay if (n.endswith(".bias") or "bn" in n.lower() or "batchnorm" in n.lower()) else decay).append(p)

        self.optimizer = torch.optim.SGD(
            [
                {"params": decay, "weight_decay": WEIGHT_DECAY},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=LR_INIT,
            momentum=MOMENTUM,
            nesterov=True,
        )

        TOTAL_EXPECTED_TRAIN_STEPS = int(ITERATIONS * LR_SCHED_STEPS_PER_ITER_EST)
        WARMUP_STEPS_CLAMPED = int(max(1, min(LR_WARMUP_STEPS, max(1, TOTAL_EXPECTED_TRAIN_STEPS - 1))))

        class WarmupCosine:
            def __init__(self, optimizer, base_lr: float, warmup_steps: int, final_lr: float, total_steps: int):
                self.opt = optimizer
                self.base = base_lr
                self.warm = max(1, int(warmup_steps))
                self.final = final_lr
                self.total = max(1, int(total_steps))
                self.t = 0

            def step(self):
                self.t += 1
                if self.t <= self.warm:
                    lr = self.base * (self.t / self.warm)
                else:
                    import math
                    progress = min(1.0, (self.t - self.warm) / max(1, self.total - self.warm))
                    lr = self.final + (self.base - self.final) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for pg in self.opt.param_groups:
                    pg["lr"] = lr

            def peek_next_lr(self):
                t_next = self.t + 1
                if t_next <= self.warm:
                    lr = self.base * (t_next / self.warm)
                else:
                    import math
                    progress = min(1.0, (t_next - self.warm) / max(1, self.total - self.warm))
                    lr = self.final + (self.base - self.final) * 0.5 * (1.0 + math.cos(math.pi * progress))
                return lr

            def set_total_steps(self, total_steps: int):
                self.total = max(self.t + 1, int(total_steps))
                if self.warm >= self.total:
                    self.warm = max(1, self.total - 1)

        self.scheduler = WarmupCosine(self.optimizer, LR_INIT, WARMUP_STEPS_CLAMPED, LR_FINAL, TOTAL_EXPECTED_TRAIN_STEPS)
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)

        self.evaluator = BatchedEvaluator(self.device)
        self.evaluator.refresh_from(self.model)

        self._proc = psutil.Process(os.getpid())
        psutil.cpu_percent(0.0)
        self._proc.cpu_percent(0.0)

        class EMA:
            def __init__(self, model: torch.nn.Module, decay: float = 0.999):
                self.decay = decay
                base = getattr(model, "_orig_mod", model)
                if hasattr(base, "module"):
                    base = base.module
                self.shadow = {k: v.detach().clone() for k, v in base.state_dict().items()}

            @torch.no_grad()
            def update(self, model: torch.nn.Module):
                base = getattr(model, "_orig_mod", model)
                if hasattr(base, "module"):
                    base = base.module
                for k, v in base.state_dict().items():
                    if not torch.is_floating_point(v):
                        self.shadow[k] = v.detach().clone()
                        continue
                    if self.shadow[k].dtype != v.dtype:
                        self.shadow[k] = self.shadow[k].to(dtype=v.dtype)
                    self.shadow[k].mul_(self.decay).add_((v.detach()), alpha=1.0 - self.decay)

            def copy_to(self, model: torch.nn.Module):
                base = getattr(model, "_orig_mod", model)
                if hasattr(base, "module"):
                    base = base.module
                base.load_state_dict(self.shadow, strict=True)

        self.ema = EMA(self.model, EMA_DECAY) if EMA_ENABLED else None
        self.best_model = self._clone_model()

        self.selfplay_engine = SelfPlayEngine(self.evaluator)
        self.iteration = 0
        self.total_games = 0
        self.start_time = time.time()
        props = torch.cuda.get_device_properties(self.device)
        self.device_name = props.name
        self.device_total_gb = props.total_memory / 1024**3
        self._prev_eval_m: dict[str, float] = {}

        self._arena_openings = self._load_openings()
        if not self._arena_openings:
            raise RuntimeError(
                "No arena openings found."
            )

        self._gate = EloGater(z=1.96, min_games=400, draw_w=0.5)
        self._gate_active = False
        self._pending_challenger: torch.nn.Module | None = None

    @staticmethod
    def _format_time(s: float) -> str:
        return f"{s:.1f}s" if s < 60 else (f"{s/60:.1f}m" if s < 3600 else f"{s/3600:.1f}h")

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

        if not os.path.isfile("openings.txt"):
            return []

        with open("openings.txt", "r", encoding="utf-8") as f:
            for line in f:
                _add_fenish(line)

        return sorted(seen)

    def train_step(self, batch_data: tuple[list[Any], list[np.ndarray], list[float]]) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        states, policies, values = batch_data
        x = (
            torch.from_numpy(np.stack(states).astype(np.float32, copy=False))
            .pin_memory()
            .to(self.device, non_blocking=True)
            .contiguous(memory_format=torch.channels_last)
        )
        pi_target = torch.from_numpy(np.stack(policies).astype(np.float32)).pin_memory().to(self.device, non_blocking=True)
        v_target = torch.tensor(values, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
        self.model.train()
        with torch.autocast(device_type="cuda", enabled=True):
            pi_pred, v_pred = self.model(x)
            if POLICY_LABEL_SMOOTH > 0.0:
                A = pi_target.shape[1]
                pi_smooth = (1.0 - POLICY_LABEL_SMOOTH) * pi_target + (POLICY_LABEL_SMOOTH / A)
            else:
                pi_smooth = pi_target
            policy_loss = F.kl_div(F.log_softmax(pi_pred, dim=1), pi_smooth, reduction="batchmean")
            value_loss = F.mse_loss(v_pred, v_target)
            ent_coef = 0.0
            if ENTROPY_COEF_INIT > 0 and self.iteration <= ENTROPY_ANNEAL_ITERS:
                ent_coef = ENTROPY_COEF_INIT * (1.0 - (self.iteration - 1) / max(1, ENTROPY_ANNEAL_ITERS))
            entropy = (-(F.softmax(pi_pred, dim=1) * F.log_softmax(pi_pred, dim=1)).sum(dim=1)).mean()
            value_weight_now = VALUE_WEIGHT_LATE if self.iteration >= VALUE_WEIGHT_SWITCH_ITER else VALUE_WEIGHT
            total_loss = POLICY_WEIGHT * policy_loss + value_weight_now * value_loss - ent_coef * entropy
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_total_norm_t = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)
        return policy_loss.detach(), value_loss.detach(), float(grad_total_norm_t.detach().cpu()), float(entropy.detach().cpu())

    def training_iteration(self) -> dict[str, int | float]:
        if self.iteration >= 200 and self.selfplay_engine.resign_consecutive < 2:
            self.selfplay_engine.resign_consecutive = 2

        stats: dict[str, int | float] = {}
        torch.cuda.reset_peak_memory_stats(self.device)
        header_lr = float(self.scheduler.peek_next_lr())
        mem = self._get_mem_info()
        sysi = self._get_sys_info()
        buf_len = len(self.selfplay_engine.buffer)
        buf_pct = (buf_len / self.selfplay_engine.buffer.maxlen) * 100
        total_elapsed = time.time() - self.start_time
        pct_done = 100.0 * (self.iteration - 1) / max(1, ITERATIONS)
        print(
            f"\n[Iter {self.iteration:>3}/{ITERATIONS} | {pct_done:>4.1f}%] LRnext {header_lr:.2e} | elapsed {self._format_time(total_elapsed)} | "
            f"GPU {mem['allocated_gb']:.1f}/{mem['reserved_gb']:.1f}/{mem['total_gb']:.1f} GB | RSS {mem['rss_gb']:.1f} GB | "
            f"CPU {sysi['cpu_sys_pct']:>4.0f}%/{sysi['cpu_proc_pct']:>4.0f}% | RAM {sysi['ram_used_gb']:.1f}/{sysi['ram_total_gb']:.1f} GB ({sysi['ram_pct']:>3.0f}%) | "
            f"load {sysi['load1']:.2f} | buf {buf_len:,}/{self.selfplay_engine.buffer.maxlen:,} ({int(buf_pct):>3}%)"
        )

        t0 = time.time()
        game_stats = self.selfplay_engine.play_games(GAMES_PER_ITER)
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
        spm = game_stats.get("sp_metrics", {}) if isinstance(game_stats, dict) else {}
        spi = game_stats.get("sp_metrics_iter", {}) if isinstance(game_stats, dict) else {}

        eval_m = self.evaluator.get_metrics()
        eval_req = int(eval_m.get("requests_total", 0))
        eval_hit = int(eval_m.get("cache_hits_total", 0))
        eval_miss = int(eval_m.get("cache_misses_total", 0))
        eval_batches = int(eval_m.get("batches_total", 0))
        eval_evalN = int(eval_m.get("eval_positions_total", 0))
        eval_bmax = int(eval_m.get("batch_size_max", 0))
        eval_qmax = int(eval_m.get("pending_queue_max", 0))
        eval_uniq = int(eval_m.get("queued_unique_total", 0))
        enc_s = float(eval_m.get("encode_time_s_total", 0.0))
        fwd_s = float(eval_m.get("forward_time_s_total", 0.0))
        wait_s = float(eval_m.get("wait_time_s_total", 0.0))
        prev = getattr(self, "_prev_eval_m", {}) or {}
        d_req = int(eval_req - int(prev.get("requests_total", 0)))
        d_hit = int(eval_hit - int(prev.get("cache_hits_total", 0)))
        d_miss = int(eval_miss - int(prev.get("cache_misses_total", 0)))
        d_batches = int(eval_batches - int(prev.get("batches_total", 0)))
        d_evalN = int(eval_evalN - int(prev.get("eval_positions_total", 0)))
        hit_rate = (100.0 * eval_hit / max(1, eval_req)) if eval_req else 0.0
        hit_rate_d = (100.0 * d_hit / max(1, d_req)) if d_req > 0 else 0.0
        enc_ms_per = (1000.0 * enc_s / max(1, eval_evalN)) if eval_evalN else 0.0
        fwd_ms_per = (1000.0 * fwd_s / max(1, eval_evalN)) if eval_evalN else 0.0
        wait_ms_per = (1000.0 * wait_s / max(1, eval_evalN)) if eval_evalN else 0.0
        req_per_s = d_req / max(1e-9, sp_elapsed)
        pos_per_s = d_evalN / max(1e-9, sp_elapsed)

        sp_line = (
            f"games {gc:,} | avg_len {avg_len:>5.1f} | "
            f"W/D/B {ww}/{dd}/{bb} ({wpct:>3.0f}%/{dpct:>3.0f}%/{bpct:>3.0f}%) | "
            f"gpm {gpm:>6.1f} | mps {mps/1000:>5.1f}K | "
            f"time {self._format_time(sp_elapsed)} | new {int(game_stats.get('moves', 0)):,}"
        )
        sp_plus_line = (
            (
                f"sims/calls {int(spm.get('mcts_sims_total', 0)):,}"
                f"(+{int(spi.get('mcts_sims_total', 0)):,})/"
                f"{int(spm.get('mcts_calls_total', 0)):,}"
                f"(+{int(spi.get('mcts_calls_total', 0)):,}) | "
                f"temp H/L {int(spm.get('temp_moves_high_total', 0)):,}"
                f"(+{int(spi.get('temp_moves_high_total', 0)):,})/"
                f"{int(spm.get('temp_moves_low_total', 0)):,}"
                f"(+{int(spi.get('temp_moves_low_total', 0)):,}) | "
                f"mcts_batch_max {int(spm.get('mcts_batch_max', 0))} | "
                f"resigns {int(spm.get('resigns_total', 0)):,}"
                f"(+{int(spi.get('resigns_total', 0)):,}) | "
                f"forced {int(spm.get('forced_results_total', 0)):,}"
                f"(+{int(spi.get('forced_results_total', 0)):,})"
            )
            if spm
            else ""
        )
        ev_line = (
            f"req {eval_req:,}(+{d_req:,}) | uniq {eval_uniq:,} | "
            f"hits {eval_hit:,} ({hit_rate:>4.1f}%) (+{d_hit:,} {hit_rate_d:>4.1f}%) | "
            f"miss {eval_miss:,}(+{d_miss:,}) | batches {eval_batches:,}(+{d_batches:,}) | "
            f"evalN {eval_evalN:,}(+{d_evalN:,}) | r/s {req_per_s:>6.1f} | n/s {int(pos_per_s):,} | "
            f"bmax {eval_bmax} | qmax {eval_qmax} | enc {enc_s:.2f}s ({enc_ms_per:.2f} ms/pos) | "
            f"fwd {fwd_s:.2f}s ({fwd_ms_per:.2f} ms/pos) | wait {wait_s:.2f}s ({wait_ms_per:.2f} ms/pos)"
        )
        stats.update(game_stats)
        stats["selfplay_time"] = sp_elapsed
        stats["games_per_min"] = gpm
        stats["moves_per_sec"] = mps

        t1 = time.time()
        losses: list[tuple[torch.Tensor, torch.Tensor]] = []
        snap = self.selfplay_engine.snapshot()
        min_samples = max(1, BATCH_SIZE // 2)
        new_examples = int(game_stats.get("moves", 0))
        desired_train_samples = int(RATIO_TARGET_TRAIN_PER_NEW * max(1, new_examples))
        steps = int(np.ceil(desired_train_samples / BATCH_SIZE))
        steps = max(RATIO_UPDATE_STEPS_MIN, min(RATIO_UPDATE_STEPS_MAX, steps))
        ratio = 0.0
        if len(snap) < min_samples:
            steps = 0
        else:
            ratio = (steps * BATCH_SIZE) / max(1, new_examples)

        grad_norm_running: float = 0.0
        ent_running: float = 0.0
        for _i_step in range(steps):
            batch = self.selfplay_engine.sample_from_snapshot(snap, BATCH_SIZE, recent_ratio=0.6)
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
        buf_pct2 = (buf_sz / self.selfplay_engine.buffer.maxlen) * 100
        bps = (len(losses) / max(1e-9, tr_elapsed)) if losses else 0.0
        sps = ((len(losses) * BATCH_SIZE) / max(1e-9, tr_elapsed)) if losses else 0.0
        current_lr = self.optimizer.param_groups[0]["lr"]
        avg_grad_norm = (grad_norm_running / max(1, len(losses))) if losses else 0.0
        avg_entropy = (ent_running / max(1, len(losses))) if losses else 0.0
        ent_coef_current = 0.0
        if ENTROPY_COEF_INIT > 0 and self.iteration <= ENTROPY_ANNEAL_ITERS:
            ent_coef_current = ENTROPY_COEF_INIT * (1.0 - (self.iteration - 1) / max(1, ENTROPY_ANNEAL_ITERS))
        drift_pct = 0.0
        if LR_SCHED_STEPS_PER_ITER_EST > 0:
            drift_pct = 100.0 * (actual_steps - LR_SCHED_STEPS_PER_ITER_EST) / LR_SCHED_STEPS_PER_ITER_EST if actual_steps > 0 else 0.0
        at_edge = steps in (RATIO_UPDATE_STEPS_MIN, RATIO_UPDATE_STEPS_MAX)
        if self.iteration == 1 and actual_steps > 0 and abs(drift_pct) > 20.0 and not at_edge:
            remaining_iters = max(0, ITERATIONS - self.iteration)
            new_total = int(self.scheduler.t + remaining_iters * actual_steps)
            new_total = max(self.scheduler.t + 1, new_total)
            self.scheduler.set_total_steps(new_total)
            print(f"LR   adjusted total_steps -> {self.scheduler.total} (iter1 measured {actual_steps} vs est {LR_SCHED_STEPS_PER_ITER_EST}, drift {drift_pct:+.1f}%)")

        stats.update(
            {
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
            }
        )

        tr_plan_line = (
            f"plan {steps} | target {RATIO_TARGET_TRAIN_PER_NEW:.1f}x | actual {ratio:>4.1f}x | mix recent 60% (last 20%), old 40%"
            if steps > 0
            else "plan 0 skip (buffer underfilled)"
        )
        tr_line = (
            f"steps {len(losses):>3} | batch/s {bps:>5.1f} | samp/s {int(sps):>6,} | time {self._format_time(tr_elapsed)} | "
            f"P {pol_loss:>7.4f} | V {val_loss:>7.4f} | LR {current_lr:.2e} | grad {avg_grad_norm:>6.3f} | entropy {avg_entropy:>6.3f} | "
            f"ent_coef {ent_coef_current:.2e} | clip {GRAD_CLIP:.1f} | buf {int(buf_pct2):>3}% ({buf_sz:,})"
        )
        lr_sched_fragment = f"sched est/iter {LR_SCHED_STEPS_PER_ITER_EST} | actual {actual_steps} | drift {drift_pct:+.1f}% | pos {self.scheduler.t}/{self.scheduler.total}"

        print("SP " + sp_line + (" | " + sp_plus_line if sp_plus_line else ""))
        print("EV " + ev_line)
        print("TR " + tr_plan_line + " | " + tr_line + " | " + lr_sched_fragment)
        self._prev_eval_m = eval_m

        self.evaluator.refresh_from(self.best_model)

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
            ce.cache_cap = ARENA_EVAL_CACHE_CAP
            ie.cache_cap = ARENA_EVAL_CACHE_CAP

            def play(e1: _BatchedEval, e2: _BatchedEval, start_fen: str) -> int:
                pos = _ccore.Position()
                pos.from_fen(start_fen)
                m1 = _ccore.MCTS(SIMULATIONS_EVAL, C_PUCT, DIRICHLET_ALPHA, float(ARENA_DIRICHLET_WEIGHT))
                m1.set_c_puct_params(C_PUCT_BASE, C_PUCT_INIT)
                m1.set_fpu_reduction(0.10)
                m2 = _ccore.MCTS(SIMULATIONS_EVAL, C_PUCT, DIRICHLET_ALPHA, float(ARENA_DIRICHLET_WEIGHT))
                m2.set_c_puct_params(C_PUCT_BASE, C_PUCT_INIT)
                m2.set_fpu_reduction(0.10)
                t = 0
                while pos.result() == _ccore.ONGOING and t < MAX_GAME_MOVES:
                    visits = m1.search_batched(pos, e1.infer_positions, EVAL_MAX_BATCH) if t % 2 == 0 else m2.search_batched(pos, e2.infer_positions, EVAL_MAX_BATCH)
                    if not visits:
                        break
                    moves = pos.legal_moves()
                    if t < ARENA_TEMP_MOVES:
                        v = _np.maximum(_np.asarray(visits, dtype=_np.float64), 0)
                        if v.sum() <= 0:
                            idx = int(_np.argmax(visits))
                        else:
                            temp = max(ARENA_OPENING_TEMPERATURE_EPS, float(ARENA_TEMPERATURE))
                            probs = v ** (1.0 / temp)
                            s = probs.sum()
                            idx = int(_np.argmax(visits)) if s <= 0 else int(_np.random.choice(len(moves), p=probs / s))
                    else:
                        idx = int(_np.argmax(visits))
                    pos.make_move(moves[idx])
                    t += 1
                r = pos.result()
                return 1 if r == _ccore.WHITE_WIN else (-1 if r == _ccore.BLACK_WIN else 0)

            pairs = max(1, ARENA_GAMES // 2)
            for _ in range(pairs):
                start_fen = openings[_np.random.randint(0, len(openings))]
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
        header = [
            "",
            "Starting training...",
            "",
            f"Device: {self.device} ({self.device_name}) | GPU {self.device_total_gb:.1f} GB | AMP on | compiled {self._compiled}",
            f"Torch: {torch.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()} | TF32 matmul {torch.backends.cuda.matmul.allow_tf32} | cuDNN TF32 {torch.backends.cudnn.allow_tf32}",
            f"Threads: torch intra {torch.get_num_threads()} | inter {torch.get_num_interop_threads()} | CPU cores {os.cpu_count()} | selfplay workers {SELFPLAY_WORKERS}",
            f"Model: {total_params:.1f}M parameters | {BLOCKS} blocks x {CHANNELS} channels | channels_last True",
            f"Buffer: size {BUFFER_SIZE:,}",
            f"Selfplay: sims {SIMULATIONS_TRAIN}→≥{MCTS_MIN_SIMS} | temp {TEMP_HIGH}->{TEMP_LOW}@{TEMP_MOVES} | resign {RESIGN_THRESHOLD} x{RESIGN_CONSECUTIVE} | dirichlet α {DIRICHLET_ALPHA} w {DIRICHLET_WEIGHT}",
            f"Training: {ITERATIONS} iterations | {GAMES_PER_ITER} games/iter | sched_est {LR_SCHED_STEPS_PER_ITER_EST} | batch {BATCH_SIZE}",
            f"LR: init {LR_INIT:.2e} | warmup {self.scheduler.warm} | final {LR_FINAL:.2e} | wd {WEIGHT_DECAY} | mom {MOMENTUM}",
            f"Augment: mirror {AUGMENT_MIRROR_PROB:.2f} | rot180 {AUGMENT_ROT180_PROB:.2f} | vflip_cs {AUGMENT_VFLIP_CS_PROB:.2f} | policy_smooth {POLICY_LABEL_SMOOTH:.2f}",
            f"EMA: {'on' if EMA_ENABLED else 'off'} decay {EMA_DECAY if EMA_ENABLED else 0}",
            f"Eval: batch {EVAL_MAX_BATCH}@{EVAL_BATCH_TIMEOUT_MS}ms | cache {EVAL_CACHE_SIZE}",
            f"Arena: games {ARENA_GAMES}/every {ARENA_EVAL_EVERY} | rule per-candidate LB>50% @Z=1.96",
            f"Arena openings: {len(self._arena_openings):,} positions loaded",
            f"Expected: {ITERATIONS * GAMES_PER_ITER:,} total games | output {OUTPUT_DIR}",
        ]
        print("\n".join(header))

        for iteration in range(1, ITERATIONS + 1):
            self.iteration = iteration
            iter_stats = self.training_iteration()

            do_eval = (iteration % ARENA_EVAL_EVERY) == 0
            arena_elapsed = 0.0
            if do_eval:
                if not self._gate_active:
                    self._pending_challenger = self._clone_from_ema()
                    self._gate.reset()
                    self._gate_active = True
                t_ar = time.time()
                score, aw, ad, al = self._arena_match(self._pending_challenger, self.best_model)
                arena_elapsed = time.time() - t_ar

                self._gate.update(aw, ad, al)
                decision, m = self._gate.decision()
                print(
                    "AR   "
                    f"n {int(m.get('n',0))} | p {100.0*m.get('p',0):>5.1f}% | "
                    f"elo {m.get('elo',0):>6.1f} ±{m.get('se_elo',0):.1f} | decision {decision.upper()} | "
                    f"W/D/L {aw}/{ad}/{al} | time {self._format_time(arena_elapsed)}"
                )

                if decision == "accept":
                    self.best_model.load_state_dict(self._pending_challenger.state_dict(), strict=True)
                    self.best_model.eval()
                    self.evaluator.refresh_from(self.best_model)
                    try:
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        tmp = os.path.join(OUTPUT_DIR, "best_model.pt.tmp")
                        dst = os.path.join(OUTPUT_DIR, "best_model.pt")
                        torch.save(self.best_model.state_dict(), tmp)
                        os.replace(tmp, dst)
                        print(f"Saved best model to {dst}")
                    except Exception as e:
                        print(f"Warning: failed to save best model: {e}")
                    self._gate_active = False
                    self._pending_challenger = None
                elif decision == "reject":
                    self._gate_active = False
                    self._pending_challenger = None
            else:
                print(f"AR   skipped | games 0 | time {self._format_time(arena_elapsed)}")

            sp_time = float(iter_stats.get("selfplay_time", 0.0))
            tr_time = float(iter_stats.get("training_time", 0.0))
            full_iter_time = sp_time + tr_time + arena_elapsed
            next_ar = 0
            if ARENA_EVAL_EVERY > 0:
                k = ARENA_EVAL_EVERY
                r = self.iteration % k
                next_ar = (k - r) if r != 0 else k
            peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1024**3
            peak_res = torch.cuda.max_memory_reserved(self.device) / 1024**3
            mem2 = self._get_mem_info()
            print(
                f"SUM  iter {self._format_time(full_iter_time)} | sp {self._format_time(sp_time)} | tr {self._format_time(tr_time)} | ar {self._format_time(arena_elapsed)} | "
                f"elapsed {self._format_time(time.time() - self.start_time)} | next_ar {next_ar} | games {self.total_games:,} | peak GPU {peak_alloc:.1f}/{peak_res:.1f} GB | RSS {mem2['rss_gb']:.1f} GB"
            )

            if (iteration % 50) == 0:
                try:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    torch.save(
                        {
                            "model": getattr(self.model, "_orig_mod", self.model).state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "sched_t": self.scheduler.t,
                            "iter": self.iteration,
                        },
                        os.path.join(OUTPUT_DIR, f"ckpt_{self.iteration:04d}.pt"),
                    )
                    print("Saved checkpoint")
                except Exception as e:
                    print(f"Warning: failed to save checkpoint: {e}")

            self._prev_eval_m = self.evaluator.get_metrics()


class EloGater:
    def __init__(self, z: float = 1.96, min_games: int = 400, draw_w: float = 0.5):
        self.z = float(z)
        self.min_games = int(min_games)
        self.draw_w = float(draw_w)
        self.reset()

    def reset(self):
        self.w = 0
        self.d = 0
        self.l = 0

    def update(self, w: int, d: int, l: int):
        self.w += int(w)
        self.d += int(d)
        self.l += int(l)

    def decision(self) -> tuple[str, dict[str, float]]:
        n = self.w + self.d + self.l
        if n < self.min_games:
            return "undecided", {"n": float(n)}
        p = (self.w + self.draw_w * self.d) / max(1, n)
        import math
        se = math.sqrt(max(1e-9, p * (1.0 - p) / max(1, n)))
        lb = p - self.z * se
        ub = p + self.z * se
        eps = 1e-9
        pc = min(1.0 - eps, max(eps, p))
        elo = 400.0 * math.log10(pc / (1.0 - pc))
        denom = max(eps, pc * (1.0 - pc))
        se_elo = (400.0 / math.log(10.0)) * se / denom
        if lb > 0.5:
            return "accept", {"n": float(n), "p": p, "lb": lb, "elo": elo, "se_elo": se_elo}
        if ub < 0.5:
            return "reject", {"n": float(n), "p": p, "ub": ub, "elo": elo, "se_elo": se_elo}
        return "undecided", {"n": float(n), "p": p, "lb": lb, "ub": ub, "elo": elo, "se_elo": se_elo}


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    Trainer().train()
