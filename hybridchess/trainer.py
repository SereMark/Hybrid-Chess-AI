from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
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
    C_PUCT,
    C_PUCT_BASE,
    C_PUCT_INIT,
    DIRICHLET_ALPHA,
    MAX_GAME_MOVES,
    Augment,
    SelfPlayEngine,
)

BATCH_SIZE = 1024
LR_INIT = 0.02
LR_SCHEDULE = ((200, 0.01), (450, 0.003), (550, 0.001))
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
GRAD_CLIP = 1.0
POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 1.0
ITERATIONS = 600
GAMES_PER_ITER = 180
TRAIN_STEPS_PER_ITER = 1024
CHECKPOINT_FREQ = 25
ITER_EMA_ALPHA = 0.3
AUGMENT_MIRROR_PROB = 0.5
AUGMENT_ROT180_PROB = 0.25
AUGMENT_VFLIP_CS_PROB = 0.25
SIMULATIONS_EVAL = 400
ARENA_EVAL_EVERY = 10
ARENA_GAMES = 100
ARENA_OPENINGS_PATH = ""
ARENA_TEMPERATURE = 0.25
ARENA_TEMP_MOVES = 8
ARENA_DIRICHLET_WEIGHT = 0.03
ARENA_OPENING_TEMPERATURE_EPS = 1e-6
ARENA_DRAW_SCORE = 0.5
ARENA_CONFIDENCE_Z = 1.0
ARENA_THRESHOLD_BASE = 0.5
ARENA_WIN_RATE = 0.52
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
        try:
            self.model = torch.compile(self.model, mode="default")
        except Exception as e:
            print(f"Warning: torch.compile failed or unavailable: {e}")
        decay: list[torch.nn.Parameter] = []
        nodecay: list[torch.nn.Parameter] = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (
                nodecay
                if (
                    n.endswith(".bias") or "bn" in n.lower() or "batchnorm" in n.lower()
                )
                else decay
            ).append(p)
        self.optimizer = torch.optim.SGD(
            [
                {"params": decay, "weight_decay": WEIGHT_DECAY},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=LR_INIT,
            momentum=MOMENTUM,
            nesterov=True,
        )
        sched_map = {m: lr for m, lr in LR_SCHEDULE}
        self._sched_map = sched_map

        def lr_lambda(epoch: int) -> float:
            t = LR_INIT
            for m in sorted(sched_map):
                if epoch >= m:
                    t = sched_map[m]
            return t / LR_INIT

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        self.evaluator = BatchedEvaluator(self.device)
        self.evaluator.refresh_from(self.model)
        self.best_model = self._clone_model()
        self.selfplay_engine = SelfPlayEngine(self.evaluator)
        self.iteration = 0
        self.total_games = 0
        self.start_time = time.time()
        self.iter_ema_time: float | None = None
        self.ema_sp_time: float | None = None
        self.ema_tr_time: float | None = None
        self.ema_ar_time: float | None = None
        self.ema_ck_time: float | None = None
        props = torch.cuda.get_device_properties(self.device)
        self.device_name = props.name
        self.device_total_gb = props.total_memory / 1024**3

    @staticmethod
    def _format_time(s: float) -> str:
        return (
            f"{s:.1f}s"
            if s < 60
            else (f"{s/60:.1f}m" if s < 3600 else f"{s/3600:.1f}h")
        )

    def _get_mem_info(self) -> dict[str, float]:
        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1024**3,
            "total_gb": self.device_total_gb,
        }

    def _clone_model(self) -> torch.nn.Module:
        clone = ChessNet().to(self.device)
        src = getattr(self.model, "_orig_mod", self.model)
        if hasattr(src, "module"):
            src = src.module
        clone.load_state_dict(src.state_dict(), strict=True)
        clone.eval()
        return clone

    def train_step(
        self, batch_data: tuple[list[Any], list[np.ndarray], list[float]]
    ) -> tuple[float, float]:
        states, policies, values = batch_data
        x = (
            torch.from_numpy(np.stack(states).astype(np.float32, copy=False))
            .pin_memory()
            .to(self.device, non_blocking=True)
            .contiguous(memory_format=torch.channels_last)
        )
        pi_target = (
            torch.from_numpy(np.stack(policies).astype(np.float32))
            .pin_memory()
            .to(self.device, non_blocking=True)
        )
        v_target = (
            torch.tensor(values, dtype=torch.float32)
            .pin_memory()
            .to(self.device, non_blocking=True)
        )
        self.model.train()
        with torch.autocast(device_type="cuda", enabled=True):
            pi_pred, v_pred = self.model(x)
            policy_loss = F.kl_div(
                F.log_softmax(pi_pred, dim=1), pi_target, reduction="batchmean"
            )
            value_loss = F.mse_loss(v_pred, v_target)
            total_loss = POLICY_WEIGHT * policy_loss + VALUE_WEIGHT * value_loss
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(policy_loss.item()), float(value_loss.item())

    def training_iteration(self) -> dict[str, int | float]:
        stats: dict[str, int | float] = {}
        torch.cuda.reset_peak_memory_stats(self.device)
        current_lr = self.optimizer.param_groups[0]["lr"]
        mem = self._get_mem_info()
        buf_len = len(self.selfplay_engine.buffer)
        buf_pct = (buf_len / BUFFER_SIZE) * 100
        total_elapsed = time.time() - self.start_time
        fut = [m for m in sorted(self._sched_map) if m > self.iteration]
        next_lr_iter = fut[0] if fut else None
        next_lr_val = self._sched_map[fut[0]] if fut else None
        pct_done = 100.0 * (self.iteration - 1) / max(1, ITERATIONS)
        nxt = (
            f" -> {next_lr_iter}:{next_lr_val:.2e}" if next_lr_iter is not None else ""
        )
        print(
            (
                f"\n[Iter {self.iteration}/{ITERATIONS} | {pct_done:.1f}%] | "
                f"LR {current_lr:.2e}{nxt} | "
                f"Elapsed {self._format_time(total_elapsed)}"
            )
        )
        print(
            (
                f"GPU {mem['allocated_gb']:.1f}/"
                f"{mem['reserved_gb']:.1f}/"
                f"{mem['total_gb']:.1f} GB | "
                f"Buffer {buf_len:,}/{BUFFER_SIZE:,} "
                f"({int(buf_pct)}%)"
            )
        )
        t0 = time.time()
        game_stats = self.selfplay_engine.play_games(GAMES_PER_ITER)
        self.total_games += int(game_stats["games"])
        sp_elapsed = time.time() - t0
        gpm = (game_stats["games"] / (sp_elapsed / 60)) if sp_elapsed > 0 else 0
        mps = (game_stats["moves"] / sp_elapsed) if sp_elapsed > 0 else 0
        gc = int(game_stats["games"])
        ww = int(game_stats["white_wins"])
        bb = int(game_stats["black_wins"])
        dd = int(game_stats["draws"])
        wpct = 100.0 * ww / gc if gc > 0 else 0.0
        dpct = 100.0 * dd / gc if gc > 0 else 0.0
        bpct = 100.0 * bb / gc if gc > 0 else 0.0
        avg_len = (game_stats["moves"] / max(1, gc)) if sp_elapsed > 0 else 0
        print(
            (
                f"SP   games {gc} | gpm {gpm:.1f} | "
                f"mps {mps/1000:.1f}K | avg_len {avg_len:.1f} | "
                f"W/D/B {ww}/{dd}/{bb} ("
                f"{wpct:.0f}%/{dpct:.0f}%/{bpct:.0f}%) | "
                f"time {self._format_time(sp_elapsed)}"
            )
        )
        stats.update(game_stats)
        stats["selfplay_time"] = sp_elapsed
        stats["games_per_min"] = gpm
        stats["moves_per_sec"] = mps
        t1 = time.time()
        losses: list[tuple[float, float]] = []
        snap = self.selfplay_engine.snapshot()
        min_samples = max(1, BATCH_SIZE // 2)
        steps = TRAIN_STEPS_PER_ITER
        if len(snap) < min_samples:
            print(
                (
                    "TR   warming up: buffer underfilled, "
                    "skipping training this iteration"
                )
            )
            steps = 0
        for _ in range(steps):
            batch = self.selfplay_engine.sample_from_snapshot(snap, BATCH_SIZE)
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
            losses.append(self.train_step((s, p, v)))
        tr_elapsed = time.time() - t1
        if losses:
            pol_loss = float(np.mean([pair[0] for pair in losses]))
            val_loss = float(np.mean([pair[1] for pair in losses]))
            buf_sz = len(self.selfplay_engine.buffer)
            buf_pct2 = (buf_sz / BUFFER_SIZE) * 100
            bps = len(losses) / tr_elapsed if tr_elapsed > 0 else 0
            sps = (len(losses) * BATCH_SIZE) / tr_elapsed if tr_elapsed > 0 else 0
            print(
                (
                    f"TR   steps {len(losses)} | batch/s {bps:.1f} | "
                    f"samp/s {int(sps)} | P {pol_loss:.4f} | "
                    f"V {val_loss:.4f} | LR {current_lr:.2e} | "
                    f"buf {int(buf_pct2)}% ({buf_sz:,}) | time "
                    f"{self._format_time(tr_elapsed)}"
                )
            )
            stats.update(
                {
                    "policy_loss": pol_loss,
                    "value_loss": val_loss,
                    "learning_rate": current_lr,
                    "buffer_size": buf_sz,
                    "buffer_percent": buf_pct2,
                    "training_time": tr_elapsed,
                    "batches_per_sec": bps,
                }
            )
        if losses:
            self.scheduler.step()
        self.evaluator.refresh_from(self.model)
        return stats

    def _arena_match(
        self, challenger: torch.nn.Module, incumbent: torch.nn.Module
    ) -> tuple[float, int, int, int]:
        import chesscore as _ccore
        import numpy as np

        from .model import BatchedEvaluator as _BatchedEval

        wins = draws = losses = 0
        ce = _BatchedEval(self.device)
        ce.eval_model.load_state_dict(challenger.state_dict(), strict=True)
        ce.eval_model.eval()
        ie = _BatchedEval(self.device)
        ie.eval_model.load_state_dict(incumbent.state_dict(), strict=True)
        ie.eval_model.eval()
        openings: list[str] = []
        if ARENA_OPENINGS_PATH:
            try:
                with open(ARENA_OPENINGS_PATH, "r", encoding="utf-8") as f:
                    openings = [line.strip() for line in f if line.strip()]
            except Exception:
                openings = []
        if not openings:
            openings = [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
                "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",
                "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
                ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/" "RNBQKBNR w KQkq c6 0 2"),
            ]

        def play(e1: _BatchedEval, e2: _BatchedEval, start_fen: str | None) -> int:
            pos = _ccore.Position()
            if start_fen:
                pos.from_fen(start_fen)
            m1 = _ccore.MCTS(
                SIMULATIONS_EVAL,
                C_PUCT,
                DIRICHLET_ALPHA,
                float(ARENA_DIRICHLET_WEIGHT),
            )
            m1.set_c_puct_params(C_PUCT_BASE, C_PUCT_INIT)
            m2 = _ccore.MCTS(
                SIMULATIONS_EVAL,
                C_PUCT,
                DIRICHLET_ALPHA,
                float(ARENA_DIRICHLET_WEIGHT),
            )
            m2.set_c_puct_params(C_PUCT_BASE, C_PUCT_INIT)
            t = 0
            while pos.result() == _ccore.ONGOING and t < MAX_GAME_MOVES:
                visits = (
                    m1.search_batched(pos, e1.infer_positions, EVAL_MAX_BATCH)
                    if t % 2 == 0
                    else m2.search_batched(pos, e2.infer_positions, EVAL_MAX_BATCH)
                )
                if not visits:
                    break
                moves = pos.legal_moves()
                if t < ARENA_TEMP_MOVES:
                    v = np.maximum(np.asarray(visits, dtype=np.float64), 0)
                    if v.sum() <= 0:
                        idx = int(np.argmax(visits))
                    else:
                        temp = max(
                            ARENA_OPENING_TEMPERATURE_EPS,
                            float(ARENA_TEMPERATURE),
                        )
                        probs = v ** (1.0 / temp)
                        s = probs.sum()
                        idx = (
                            int(np.argmax(visits))
                            if s <= 0
                            else int(np.random.choice(len(moves), p=probs / s))
                        )
                else:
                    idx = int(np.argmax(visits))
                pos.make_move(moves[idx])
                t += 1
            r = pos.result()
            return 1 if r == _ccore.WHITE_WIN else (-1 if r == _ccore.BLACK_WIN else 0)

        for g in range(ARENA_GAMES):
            start_fen = (
                openings[np.random.randint(0, len(openings))] if openings else None
            )
            r = play(ce, ie, start_fen) if g % 2 == 0 else -play(ie, ce, start_fen)
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1
        score = (wins + ARENA_DRAW_SCORE * draws) / max(1, ARENA_GAMES)
        return score, wins, draws, losses

    def train(self) -> None:
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        sched_pairs = ", ".join([f"{m} -> {lr:.1e}" for m, lr in LR_SCHEDULE])
        header = [
            "",
            "Starting training...",
            "",
            (
                f"Device: {self.device} ({self.device_name}) | GPU "
                f"{self.device_total_gb:.1f} GB | AMP on"
            ),
            (
                f"Model: {total_params:.1f}M parameters | {BLOCKS} blocks x "
                f"{CHANNELS} channels"
            ),
            (f"Training: {ITERATIONS} iterations | {GAMES_PER_ITER} " f"games/iter"),
            f"LR: init {LR_INIT:.2e} | schedule: {sched_pairs}",
            f"Expected: {ITERATIONS * GAMES_PER_ITER:,} total games",
            (
                "Config: eval batch "
                f"{EVAL_MAX_BATCH}@{EVAL_BATCH_TIMEOUT_MS}ms | "
                "cache "
                f"{EVAL_CACHE_SIZE} | arena {ARENA_GAMES}/every "
                f"{ARENA_EVAL_EVERY} | gate z {ARENA_CONFIDENCE_Z:.2f} "
                "thr "
                f"{max(ARENA_THRESHOLD_BASE, ARENA_WIN_RATE):.2f} | "
                "channels_last True"
            ),
        ]
        print("\n".join(header))
        for iteration in range(1, ITERATIONS + 1):
            self.iteration = iteration
            iter_stats = self.training_iteration()
            if iteration % CHECKPOINT_FREQ == 0:
                ck_start = time.time()
                src = getattr(self.model, "_orig_mod", self.model)
                if hasattr(src, "module"):
                    src = src.module
                ckpt = {
                    "iteration": self.iteration,
                    "total_games": self.total_games,
                    "model_state_dict": src.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "training_time": time.time() - self.start_time,
                }
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                path = os.path.join(OUTPUT_DIR, f"checkpoint_{self.iteration}.pt")
                tmp = path + ".tmp"
                torch.save(ckpt, tmp)
                os.replace(tmp, path)
                size_mb = os.path.getsize(path) / 1024**2
                ck_elapsed = time.time() - ck_start
                if self.ema_ck_time is None:
                    self.ema_ck_time = ck_elapsed
                else:
                    self.ema_ck_time = (
                        ITER_EMA_ALPHA * ck_elapsed
                        + (1 - ITER_EMA_ALPHA) * self.ema_ck_time
                    )
                print(
                    "Checkpoint: %s | %.1fMB | Games %s | Time %s"
                    % (
                        path,
                        size_mb,
                        f"{self.total_games:,}",
                        self._format_time(time.time() - self.start_time),
                    )
                )
            do_eval = (iteration % ARENA_EVAL_EVERY) == 0
            arena_elapsed = 0.0
            if do_eval:
                challenger = self._clone_model()
                t_ar = time.time()
                score, aw, ad, al = self._arena_match(challenger, self.best_model)
                arena_elapsed = time.time() - t_ar
                self._arena_acc_w = getattr(self, "_arena_acc_w", 0) + aw
                self._arena_acc_d = getattr(self, "_arena_acc_d", 0) + ad
                self._arena_acc_l = getattr(self, "_arena_acc_l", 0) + al
                gw, gd, gl = (
                    self._arena_acc_w,
                    self._arena_acc_d,
                    self._arena_acc_l,
                )
                gn = max(1, gw + gd + gl)
                acc_mu = (gw + ARENA_DRAW_SCORE * gd) / gn
                threshold = max(ARENA_THRESHOLD_BASE, ARENA_WIN_RATE)
                e_x2 = (self._arena_acc_w + 0.25 * self._arena_acc_d) / max(1, gn)
                var = max(0.0, e_x2 - acc_mu * acc_mu)
                se = (var / max(1, gn)) ** 0.5
                lb = acc_mu - ARENA_CONFIDENCE_Z * se
                promote = lb >= threshold
                cur_n = ARENA_GAMES
                cur_mu = (aw + ARENA_DRAW_SCORE * ad) / max(1, cur_n)
                cur_e_x2 = (aw + 0.25 * ad) / max(1, cur_n)
                cur_var = max(0.0, cur_e_x2 - cur_mu * cur_mu)
                cur_se = (cur_var / max(1, cur_n)) ** 0.5
                cur_lb = cur_mu - ARENA_CONFIDENCE_Z * cur_se
                acc_e_x2 = (self._arena_acc_w + 0.25 * self._arena_acc_d) / max(1, gn)
                acc_var = max(0.0, acc_e_x2 - acc_mu * acc_mu)
                acc_se = (acc_var / max(1, gn)) ** 0.5
                acc_lb = acc_mu - ARENA_CONFIDENCE_Z * acc_se
                pure_wr = aw / max(1, ARENA_GAMES)
                print(
                    (
                        f"AR   score {100.0 * score:.1f}% | "
                        f"win {100.0 * pure_wr:.1f}% | "
                        f"cur LB {100.0 * cur_lb:.1f}% | "
                        f"acc LB {100.0 * acc_lb:.1f}% ({gn}) | "
                        f"W/D/L {aw}/{ad}/{al} | games {ARENA_GAMES} | "
                        f"time {self._format_time(arena_elapsed)}"
                    )
                )
                if promote:
                    self.best_model.load_state_dict(
                        challenger.state_dict(), strict=True
                    )
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
                    if hasattr(self, "_arena_acc_w"):
                        self._arena_acc_w = 0
                        self._arena_acc_d = 0
                        self._arena_acc_l = 0
            else:
                print(
                    (
                        f"AR   skipped | games 0 | time "
                        f"{self._format_time(arena_elapsed)}"
                    )
                )
            sp_time = float(iter_stats.get("selfplay_time", 0.0))
            tr_time = float(iter_stats.get("training_time", 0.0))
            full_iter_time = sp_time + tr_time + arena_elapsed
            self.iter_ema_time = (
                full_iter_time
                if self.iter_ema_time is None
                else (
                    ITER_EMA_ALPHA * full_iter_time
                    + (1 - ITER_EMA_ALPHA) * self.iter_ema_time
                )
            )
            if self.ema_sp_time is None:
                self.ema_sp_time = sp_time
            else:
                self.ema_sp_time = (
                    ITER_EMA_ALPHA * sp_time + (1 - ITER_EMA_ALPHA) * self.ema_sp_time
                )
            if self.ema_tr_time is None:
                self.ema_tr_time = tr_time
            else:
                self.ema_tr_time = (
                    ITER_EMA_ALPHA * tr_time + (1 - ITER_EMA_ALPHA) * self.ema_tr_time
                )
            if arena_elapsed > 0.0:
                if self.ema_ar_time is None:
                    self.ema_ar_time = arena_elapsed
                else:
                    self.ema_ar_time = (
                        ITER_EMA_ALPHA * arena_elapsed
                        + (1 - ITER_EMA_ALPHA) * self.ema_ar_time
                    )
            avg_iter = (
                self.iter_ema_time if self.iter_ema_time is not None else full_iter_time
            )
            remaining = max(0, ITERATIONS - self.iteration)
            eta_sec = avg_iter * remaining
            if ARENA_EVAL_EVERY > 0:
                k = ARENA_EVAL_EVERY
                rem_arena = ((self.iteration + remaining) // k) - (self.iteration // k)
                sp_est_raw = (
                    self.ema_sp_time if self.ema_sp_time is not None else sp_time
                )
                tr_est_raw = (
                    self.ema_tr_time if self.ema_tr_time is not None else tr_time
                )
                sp_est = max(sp_est_raw, sp_time)
                tr_est = max(tr_est_raw, tr_time)
                if self.ema_ar_time is not None:
                    ar_est = self.ema_ar_time
                else:
                    ar_est = max(arena_elapsed, sp_est + tr_est)
                eta_sec = remaining * (sp_est + tr_est) + max(0, rem_arena) * max(
                    0.0, ar_est
                )
            if CHECKPOINT_FREQ > 0:
                k = CHECKPOINT_FREQ
                rem_ck = ((self.iteration + remaining) // k) - (self.iteration // k)
                ck_est = self.ema_ck_time if self.ema_ck_time is not None else 0.0
                eta_sec += max(0, rem_ck) * max(0.0, ck_est)
            next_ar = 0
            if ARENA_EVAL_EVERY > 0:
                k = ARENA_EVAL_EVERY
                r = self.iteration % k
                next_ar = (k - r) if r != 0 else k
            next_ck = 0
            if CHECKPOINT_FREQ > 0:
                k = CHECKPOINT_FREQ
                r = self.iteration % k
                next_ck = (k - r) if r != 0 else k
            peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1024**3
            peak_res = torch.cuda.max_memory_reserved(self.device) / 1024**3
            print(
                (
                    f"SUM  iter {self._format_time(full_iter_time)} | (sp "
                    f"{self._format_time(sp_time)} + tr "
                    f"{self._format_time(tr_time)} + ar "
                    f"{self._format_time(arena_elapsed)})"
                )
            )
            sp_ema = self._format_time(self.ema_sp_time) if self.ema_sp_time else "-"
            tr_ema = self._format_time(self.ema_tr_time) if self.ema_tr_time else "-"
            ar_ema = self._format_time(self.ema_ar_time) if self.ema_ar_time else "-"
            print(
                (
                    f"     avg {self._format_time(avg_iter)} | elapsed "
                    f"{self._format_time(time.time() - self.start_time)} | "
                    f"ETA {self._format_time(eta_sec)} | next_ar {next_ar} | "
                    f"next_ck {next_ck} | EMA sp/tr/ar {sp_ema} / {tr_ema} / "
                    f"{ar_ema}"
                )
            )
            print(
                (
                    f"     games {self.total_games:,} | peak GPU "
                    f"{peak_alloc:.1f}/{peak_res:.1f} GB"
                )
            )


if __name__ == "__main__":
    from .selfplay import BUFFER_SIZE

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    Trainer().train()
