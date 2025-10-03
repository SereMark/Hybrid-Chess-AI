from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from reporting import get_mem_info


def _entropy_coefficient(iteration: int) -> float:
    coef = 0.0
    if C.TRAIN.LOSS_ENTROPY_COEF_INIT > 0 and iteration <= C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS:
        coef = C.TRAIN.LOSS_ENTROPY_COEF_INIT * (
            1.0 - (iteration - 1) / max(1, C.TRAIN.LOSS_ENTROPY_ANNEAL_ITERS)
        )
    if C.TRAIN.LOSS_ENTROPY_COEF_MIN > 0:
        coef = max(float(coef), float(C.TRAIN.LOSS_ENTROPY_COEF_MIN))
    return float(coef)


def _value_loss_weight(iteration: int) -> float:
    if iteration >= C.TRAIN.LOSS_VALUE_WEIGHT_SWITCH_ITER:
        return float(C.TRAIN.LOSS_VALUE_WEIGHT_LATE)
    return float(C.TRAIN.LOSS_VALUE_WEIGHT)


def run_training_iteration(trainer: Any) -> dict[str, int | float | str]:
    stats: dict[str, int | float | str] = {}
    if hasattr(trainer, "_grad_skip_count"):
        trainer._grad_skip_count = 0
    buffer_reset = 0
    resign_status = "disabled:warmup"
    guard_min_len = float(getattr(C.RESIGN, "GUARD_MIN_AVG_LEN", 0.0))
    prev_cooldown = int(getattr(trainer, "_resign_cooldown", 0))
    trainer._resign_cooldown = max(0, prev_cooldown - 1)
    if not C.RESIGN.ENABLED:
        trainer.selfplay_engine.resign_consecutive = 0
        resign_status = "disabled:config"
    else:
        allow_resign = True
        if trainer.iteration < int(getattr(C.RESIGN, "DISABLE_UNTIL_ITERS", 0)):
            allow_resign = False
            resign_status = "disabled:warmup"
        elif getattr(trainer, "_resign_cooldown", 0) > 0:
            allow_resign = False
            resign_status = "disabled:cooldown"
        else:
            resign_status = "enabled"

        if not allow_resign:
            trainer.selfplay_engine.resign_consecutive = 0
        else:
            ramp_start = int(getattr(C.RESIGN, "VALUE_THRESHOLD_RAMP_START", 0))
            ramp_end = int(max(ramp_start + 1, getattr(C.RESIGN, "VALUE_THRESHOLD_RAMP_END", ramp_start + 1)))
            init_thr = float(getattr(C.RESIGN, "VALUE_THRESHOLD_INIT", C.RESIGN.VALUE_THRESHOLD))
            final_thr = float(getattr(C.RESIGN, "VALUE_THRESHOLD", init_thr))
            thr_alpha = 0.0 if ramp_end <= ramp_start else float(
                np.clip((trainer.iteration - ramp_start) / (ramp_end - ramp_start), 0.0, 1.0)
            )
            threshold = init_thr + thr_alpha * (final_thr - init_thr)

            window = getattr(trainer, "_resign_eval_losses", None)
            target_pct = float(getattr(trainer, "_resign_target_pct", getattr(C.RESIGN, "TARGET_LOSS_VALUE_PCTL", 0.25)))
            if window and len(window) >= 8:
                sorted_losses = sorted(window)
                idx = int(max(0, min(len(sorted_losses) - 1, math.floor(target_pct * (len(sorted_losses) - 1)))))
                dynamic_thr = float(sorted_losses[idx])
                threshold = float(min(threshold, dynamic_thr))

            mp_start = int(max(0, getattr(C.RESIGN, "MIN_PLIES_INIT", 0)))
            mp_final = int(max(0, getattr(C.RESIGN, "MIN_PLIES_FINAL", mp_start)))
            mp_end = int(max(ramp_start + 1, getattr(C.RESIGN, "MIN_PLIES_RAMP_END", ramp_end)))
            mp_alpha = 0.0 if mp_end <= ramp_start else float(
                np.clip((trainer.iteration - ramp_start) / (mp_end - ramp_start), 0.0, 1.0)
            )
            min_plies = round(mp_start + mp_alpha * (mp_final - mp_start))

            trainer._resign_threshold = threshold
            trainer._resign_min_plies = int(min_plies)
            trainer.selfplay_engine.set_resign_params(threshold, int(min_plies))
            trainer.selfplay_engine.resign_consecutive = max(
                int(C.RESIGN.CONSECUTIVE_PLIES), int(C.RESIGN.CONSECUTIVE_MIN)
            )

    if trainer.iteration == C.ARENA.GATE_Z_SWITCH_ITER:
        trainer._gate.z = float(C.ARENA.GATE_Z_LATE)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    try:
        buffer_len = int(trainer.selfplay_engine.size())
        buffer_cap = int(trainer.selfplay_engine.get_capacity())
    except Exception:
        buffer_len = 0
        buffer_cap = 1
    buffer_fill = buffer_len / max(1, buffer_cap)
    ratio_max = float(getattr(C.SAMPLING, "TRAIN_RECENT_SAMPLE_RATIO_MAX", C.SAMPLING.TRAIN_RECENT_SAMPLE_RATIO))
    ratio_min = float(getattr(C.SAMPLING, "TRAIN_RECENT_SAMPLE_RATIO_MIN", ratio_max))
    decay_start = float(getattr(C.SAMPLING, "TRAIN_RECENT_SAMPLE_BUFFER_DECAY_START", 0.6))
    decay_end = float(getattr(C.SAMPLING, "TRAIN_RECENT_SAMPLE_BUFFER_DECAY_END", 1.0))
    decay_start = float(np.clip(decay_start, 0.0, 1.0))
    decay_end = float(np.clip(decay_end, decay_start + 1e-6, 1.0))
    if buffer_fill <= decay_start:
        trainer._recent_sample_ratio = float(np.clip(ratio_max, ratio_min, ratio_max))
    elif buffer_fill >= decay_end:
        trainer._recent_sample_ratio = float(np.clip(ratio_min, ratio_min, ratio_max))
    else:
        span = decay_end - decay_start
        alpha = (buffer_fill - decay_start) / max(1e-9, span)
        target_ratio = ratio_max - alpha * (ratio_max - ratio_min)
        trainer._recent_sample_ratio = float(np.clip(target_ratio, ratio_min, ratio_max))


    t0 = time.time()
    try:
        trainer.selfplay_engine.update_adjudication(trainer.iteration)
    except Exception as exc:
        trainer.log.debug("Self-play adjudication tick failed: %s", exc, exc_info=True)
    game_stats = trainer.selfplay_engine.play_games(C.TRAIN.GAMES_PER_ITER)
    trainer.total_games += int(game_stats["games"])
    sp_elapsed = time.time() - t0

    loss_samples_priv = game_stats.pop("_loss_eval_samples", None)
    loss_window = getattr(trainer, "_resign_eval_losses", None)
    if loss_window is not None and loss_samples_priv:
        for sample in loss_samples_priv:
            try:
                val = float(sample)
            except (TypeError, ValueError):
                continue
            if val <= 0.0:
                loss_window.append(val)

    games_per_min = game_stats["games"] / max(1e-9, sp_elapsed / 60)
    moves_per_sec = game_stats["moves"] / max(1e-9, sp_elapsed)
    games_count = int(game_stats["games"])
    white_wins = int(game_stats["white_wins"])
    black_wins = int(game_stats["black_wins"])
    draws_true = int(game_stats.get("draws_true", 0))
    draws_cap = int(game_stats.get("draws_cap", 0))
    draws_count = int(draws_true + draws_cap)
    white_win_pct = 100.0 * white_wins / max(1, games_count)
    draw_pct = 100.0 * draws_count / max(1, games_count)
    black_win_pct = 100.0 * black_wins / max(1, games_count)
    draw_true_pct = 100.0 * draws_true / max(1, games_count)
    draw_cap_pct = 100.0 * draws_cap / max(1, games_count)
    avg_game_length = game_stats["moves"] / max(1, games_count)
    starts_white = int(game_stats.get("starts_white", 0))
    starts_black = int(game_stats.get("starts_black", 0))
    total_starts = max(1, starts_white + starts_black)
    white_start_pct = 100.0 * starts_white / total_starts
    black_start_pct = 100.0 * starts_black / total_starts
    unique_positions_avg = float(game_stats.get("unique_positions_avg", 0.0))
    unique_ratio_avg = float(game_stats.get("unique_ratio_avg", 0.0))
    unique_ratio_pct = 100.0 * unique_ratio_avg
    halfmove_avg = float(game_stats.get("halfmove_avg", 0.0))
    halfmove_max = int(game_stats.get("halfmove_max", 0))
    threefold_pct = float(game_stats.get("threefold_pct", 0.0))
    threefold_games = int(game_stats.get("threefold_games", 0))
    threefold_draws = int(game_stats.get("threefold_draws", 0))
    curriculum_games = int(game_stats.get("curriculum_games", 0))
    curriculum_win_pct = float(game_stats.get("curriculum_win_pct", 0.0))
    curriculum_draw_pct = float(game_stats.get("curriculum_draw_pct", 0.0))
    term_nat = int(game_stats.get("term_natural", 0))
    term_exh = int(game_stats.get("term_exhausted", 0))
    term_resign = int(game_stats.get("term_resign", 0))
    term_threefold = int(game_stats.get("term_threefold", 0))
    term_fifty = int(game_stats.get("term_fifty", 0))
    term_adjudicated = int(game_stats.get("term_adjudicated", 0))
    term_loop = int(game_stats.get("term_loop", 0))
    adjudicated_games = int(game_stats.get("adjudicated_games", 0))
    adjudicated_white = int(game_stats.get("adjudicated_white", 0))
    adjudicated_black = int(game_stats.get("adjudicated_black", 0))
    adjudicated_draw = int(game_stats.get("adjudicated_draw", 0))
    adjudicated_balance_total = float(game_stats.get("adjudicated_balance_total", 0.0))
    adjudicated_balance_abs_total = float(game_stats.get("adjudicated_balance_abs_total", 0.0))
    color_delta = abs(white_start_pct - 50.0)
    color_tol = float(getattr(C.SELFPLAY, "COLOR_BALANCE_TOLERANCE_PCT", 5.0))
    color_window = getattr(trainer, "_color_window", None)
    window_logged = False
    window_white_pct = None
    window_black_pct = None
    if color_window:
        color_window.append((starts_white, starts_black))
        window_games = sum(w + b for w, b in color_window)
        if window_games > 0 and len(color_window) >= color_window.maxlen:
            window_white = sum(w for w, _ in color_window)
            window_white_pct = 100.0 * window_white / window_games
            window_black_pct = 100.0 - window_white_pct
            window_delta = abs(window_white_pct - 50.0)
            if window_delta > color_tol:
                trainer.log.info(
                    "[SelfPlay] color-start imbalance (%d-game window) W/B %.1f/%.1f%% (tol=%.1f%%)",
                    window_games,
                    window_white_pct,
                    window_black_pct,
                    color_tol,
                )
                window_logged = True
    if not window_logged and color_delta > color_tol:
        trainer.log.info(
            "[SelfPlay] color-start imbalance W/B %.1f/%.1f%% (tol=%.1f%%)",
            white_start_pct,
            black_start_pct,
            color_tol,
        )

    imbalance_source_pct = window_white_pct if window_white_pct is not None else white_start_pct
    imbalance = (imbalance_source_pct - 50.0) / 100.0
    tol_ratio = color_tol / 100.0
    if abs(imbalance) <= tol_ratio:
        target_prob_black = 0.5
    else:
        adjust = float(np.clip(imbalance * 1.5, -0.3, 0.3))
        target_prob_black = float(np.clip(0.5 + adjust, 0.2, 0.8))
    try:
        trainer.selfplay_engine.set_color_bias(target_prob_black)
    except Exception as exc:
        trainer.log.debug("Failed to adjust color bias to %.3f: %s", target_prob_black, exc, exc_info=True)
    bias_actual = float(game_stats.get("color_bias_prob_black", trainer.selfplay_engine.get_color_bias()))

    eval_metrics = trainer.evaluator.get_metrics()
    requests_total = int(eval_metrics.get("requests_total", 0))
    cache_hits_total = int(eval_metrics.get("cache_hits_total", 0))
    batches_total = int(eval_metrics.get("batches_total", 0))
    eval_positions_total = int(eval_metrics.get("eval_positions_total", 0))
    max_batch_size = int(eval_metrics.get("batch_size_max", 0))
    prev_metrics = getattr(trainer, "_prev_eval_m", {}) or {}
    delta_requests = int(requests_total - int(prev_metrics.get("requests_total", 0)))
    delta_hits = int(cache_hits_total - int(prev_metrics.get("cache_hits_total", 0)))
    delta_batches = int(batches_total - int(prev_metrics.get("batches_total", 0)))
    delta_eval_positions = int(eval_positions_total - int(prev_metrics.get("eval_positions_total", 0)))
    hit_rate = (100.0 * cache_hits_total / max(1, requests_total)) if requests_total else 0.0
    hit_rate_d = (100.0 * delta_hits / max(1, delta_requests)) if delta_requests > 0 else 0.0
    avg_batch_delta = 0.0

    try:
        if delta_batches > 0 and delta_eval_positions > 0:
            avg_batch_delta = delta_eval_positions / max(1, delta_batches)
            cap = int(trainer._eval_batch_cap)
            max_cap_allowed = int(
                min(512, max(256, int(1024 * (trainer.device_total_gb / 12.0))))
            )
            mem_now = get_mem_info(trainer._proc, trainer.device, trainer.device_total_gb)
            reserved_frac = float(mem_now["reserved_gb"]) / max(1e-9, float(mem_now["total_gb"]))
            if avg_batch_delta >= 0.90 * cap and cap < max_cap_allowed and reserved_frac < 0.90:
                new_cap = int(min(max_cap_allowed, cap + 256))
                if new_cap != cap:
                    trainer._eval_batch_cap = new_cap
                    trainer.evaluator.set_batching_params(batch_size_max=new_cap)
                    trainer.log.info(f"[AUTO    ] eval_batch_size_max {cap} -> {new_cap}")
            elif avg_batch_delta <= 0.25 * cap and trainer._eval_coalesce_ms > 4:
                old_ms = int(trainer._eval_coalesce_ms)
                new_ms = int(max(4, int(old_ms * 0.8)))
                if new_ms != old_ms:
                    trainer._eval_coalesce_ms = new_ms
                    trainer.evaluator.set_batching_params(coalesce_ms=new_ms)
                    trainer.log.info(f"[AUTO    ] eval_coalesce_ms {old_ms} -> {new_ms}")
            elif avg_batch_delta >= 0.80 * cap and trainer._eval_coalesce_ms < 36:
                old_ms = int(trainer._eval_coalesce_ms)
                new_ms = int(min(36, int(old_ms * 1.15 + 1)))
                if new_ms != old_ms:
                    trainer._eval_coalesce_ms = new_ms
                    trainer.evaluator.set_batching_params(coalesce_ms=new_ms)
                    trainer.log.info(f"[AUTO    ] eval_coalesce_ms {old_ms} -> {new_ms}")
    except Exception:
        pass

    stats.update(game_stats)
    stats["selfplay_time"] = sp_elapsed
    stats["games_per_min"] = games_per_min
    stats["moves_per_sec"] = moves_per_sec
    stats["sp_terminal_eval_mean"] = float(game_stats.get("terminal_eval_mean", 0.0))
    stats["sp_terminal_eval_abs_mean"] = float(game_stats.get("terminal_eval_abs_mean", 0.0))
    stats["sp_tempo_bonus_avg"] = float(game_stats.get("tempo_bonus_avg", 0.0))
    stats["sp_sims_avg"] = float(game_stats.get("sims_avg", 0.0))
    stats["sp_value_trend_hit_pct"] = float(game_stats.get("value_trend_hit_pct", 0.0))
    stats["sp_loss_eval_mean"] = float(game_stats.get("loss_eval_mean", 0.0))
    stats["sp_loss_eval_p05"] = float(game_stats.get("loss_eval_p05", 0.0))
    stats["sp_loss_eval_p25"] = float(game_stats.get("loss_eval_p25", 0.0))
    stats["sp_white_win_pct"] = white_win_pct
    stats["sp_draw_pct"] = draw_pct
    stats["sp_draw_true_pct"] = draw_true_pct
    stats["sp_draw_cap_pct"] = draw_cap_pct
    stats["sp_black_win_pct"] = black_win_pct
    stats["sp_avg_len"] = avg_game_length
    stats["sp_new_moves"] = int(game_stats.get("moves", 0))
    stats["sp_white_starts"] = starts_white
    stats["sp_black_starts"] = starts_black
    stats["sp_white_start_pct"] = white_start_pct
    stats["sp_black_start_pct"] = black_start_pct
    stats["sp_color_bias_black_pct"] = 100.0 * bias_actual
    stats["sp_color_bias_target_pct"] = 100.0 * target_prob_black
    stats["sp_unique_positions_avg"] = unique_positions_avg
    stats["sp_unique_ratio_pct"] = unique_ratio_pct
    stats["sp_halfmove_avg"] = halfmove_avg
    stats["sp_halfmove_max"] = halfmove_max
    stats["sp_threefold_pct"] = threefold_pct
    stats["sp_threefold_games"] = threefold_games
    stats["sp_threefold_draws"] = threefold_draws
    loop_games = int(game_stats.get("loop_alert_games", 0))
    loop_alerts = int(game_stats.get("loop_alerts_total", 0))
    loop_flag_games = int(game_stats.get("loop_flag_games", 0))
    loop_flag_pct = float(game_stats.get("loop_flag_pct", 0.0))
    loop_unique_ratio_pct = float(game_stats.get("loop_unique_ratio_pct", 0.0))
    loop_bias_avg = float(game_stats.get("loop_bias_avg", 0.0))
    stats["sp_loop_games"] = loop_games
    stats["sp_loop_alerts"] = loop_alerts
    stats["sp_loop_pct"] = 100.0 * loop_games / max(1, games_count)
    stats["sp_loop_flag_games"] = loop_flag_games
    stats["sp_loop_flag_pct"] = loop_flag_pct
    stats["sp_loop_unique_ratio_pct"] = loop_unique_ratio_pct
    stats["sp_loop_bias_avg"] = loop_bias_avg

    loop_reset_threshold = float(getattr(C.SELFPLAY, "LOOP_AUTO_RESET_THRESHOLD", 0.0))
    loop_cooldown_iters = int(max(0, getattr(C.SELFPLAY, "LOOP_AUTO_RESET_COOLDOWN", 0)))
    current_loop_cooldown = max(0, getattr(trainer, "_loop_cooldown", 0))
    if loop_reset_threshold > 0.0 and loop_flag_pct >= loop_reset_threshold:
        if current_loop_cooldown == 0:
            trainer.log.warning(
                "[SelfPlay] loop rate %.1f%% >= %.1f%%; refreshing replay buffer",
                loop_flag_pct,
                loop_reset_threshold,
            )
            try:
                trainer.selfplay_engine.clear_buffer()
                buffer_reset = 1
            except Exception as exc:
                trainer.log.debug("Replay buffer clear failed during loop reset: %s", exc, exc_info=True)
            trainer._recent_sample_ratio = float(
                getattr(C.SAMPLING, "TRAIN_RECENT_SAMPLE_RATIO", trainer._recent_sample_ratio)
            )
            trainer._loop_cooldown = loop_cooldown_iters
        else:
            trainer._loop_cooldown = current_loop_cooldown - 1
    else:
        trainer._loop_cooldown = max(current_loop_cooldown - 1, 0)

    stats["sp_curriculum_games"] = curriculum_games
    stats["sp_curriculum_pct"] = 100.0 * curriculum_games / max(1, games_count)
    stats["sp_curriculum_win_pct"] = curriculum_win_pct
    stats["sp_curriculum_draw_pct"] = curriculum_draw_pct
    stats["sp_adjudicate_margin"] = float(game_stats.get("adjudicate_margin", 0.0))
    stats["sp_adjudicate_min_plies"] = int(game_stats.get("adjudicate_min_plies", 0))
    stats["sp_term_natural"] = term_nat
    stats["sp_term_exhausted"] = term_exh
    stats["sp_term_resign"] = term_resign
    stats["sp_term_threefold"] = term_threefold
    stats["sp_term_fifty"] = term_fifty
    stats["sp_term_adjudicated"] = term_adjudicated
    stats["sp_term_loop"] = term_loop
    stats["sp_adjudicated_games"] = adjudicated_games
    stats["sp_adjudicated_white"] = adjudicated_white
    stats["sp_adjudicated_black"] = adjudicated_black
    stats["sp_adjudicated_draw"] = adjudicated_draw
    term_total = max(1, term_nat + term_exh + term_resign + term_threefold + term_fifty + term_adjudicated + term_loop)
    adjudicated_pct = 100.0 * adjudicated_games / max(1, games_count)
    adjudicated_white_pct = 100.0 * adjudicated_white / max(1, adjudicated_games) if adjudicated_games else 0.0
    adjudicated_black_pct = 100.0 * adjudicated_black / max(1, adjudicated_games) if adjudicated_games else 0.0
    adjudicated_draw_pct = 100.0 * adjudicated_draw / max(1, adjudicated_games) if adjudicated_games else 0.0
    adjudicated_balance_mean = (
        adjudicated_balance_total / max(1, adjudicated_games)
        if adjudicated_games
        else 0.0
    )
    adjudicated_balance_abs_mean = (
        adjudicated_balance_abs_total / max(1, adjudicated_games)
        if adjudicated_games
        else 0.0
    )

    stats["sp_term_natural_pct"] = 100.0 * term_nat / term_total
    stats["sp_term_exhausted_pct"] = 100.0 * term_exh / term_total
    stats["sp_term_resign_pct"] = 100.0 * term_resign / term_total
    stats["sp_term_threefold_pct"] = 100.0 * term_threefold / term_total
    stats["sp_term_fifty_pct"] = 100.0 * term_fifty / term_total
    stats["sp_term_adjudicated_pct"] = 100.0 * term_adjudicated / term_total
    stats["sp_term_loop_pct"] = 100.0 * term_loop / term_total
    stats["sp_adjudicated_pct"] = adjudicated_pct
    stats["sp_adjudicated_white_pct"] = adjudicated_white_pct
    stats["sp_adjudicated_black_pct"] = adjudicated_black_pct
    stats["sp_adjudicated_draw_pct"] = adjudicated_draw_pct
    stats["sp_adjudicated_balance_mean"] = adjudicated_balance_mean
    stats["sp_adjudicated_balance_abs_mean"] = adjudicated_balance_abs_mean
    stats["sp_adjudicated_balance_total"] = adjudicated_balance_total
    stats["sp_adjudicated_balance_abs_total"] = adjudicated_balance_abs_total

    if hasattr(trainer, "_recent_len_window"):
        trainer._recent_len_window.append(float(avg_game_length))
        if len(trainer._recent_len_window) == trainer._recent_len_window.maxlen:
            window_avg = sum(trainer._recent_len_window) / max(1, len(trainer._recent_len_window))
            if C.RESIGN.ENABLED and window_avg < guard_min_len:
                guard_cd = int(max(0, getattr(C.RESIGN, "GUARD_COOLDOWN_ITERS", 0)))
                if guard_cd > 0 and trainer._resign_cooldown < guard_cd:
                    trainer._resign_cooldown = guard_cd
                    resign_status = "disabled:guard"
                    trainer.selfplay_engine.resign_consecutive = 0
                    try:
                        trainer.selfplay_engine.clear_buffer()
                        buffer_reset = 1
                    except Exception as exc:
                        trainer.log.debug("Replay buffer clear failed during resign guard: %s", exc, exc_info=True)
                    try:
                        trainer._recent_len_window.clear()
                        trainer._recent_sample_ratio = float(
                            getattr(C.SAMPLING, "TRAIN_RECENT_SAMPLE_RATIO", 0.8)
                        )
                    except Exception as exc:
                        trainer.log.debug("Unable to reset recent length window during resign guard: %s", exc, exc_info=True)
                    trainer.log.warning(
                        "[Resign  ] guard triggered; avg_len window %.2f < %.2f. Disabling resigns for %d iters",
                        window_avg,
                        guard_min_len,
                        guard_cd,
                    )

    stats["resign_status"] = resign_status
    stats["resign_threshold"] = float(getattr(trainer, "_resign_threshold", C.RESIGN.VALUE_THRESHOLD))
    stats["resign_min_plies"] = int(getattr(trainer, "_resign_min_plies", 0))
    stats["resign_cooldown_iters"] = int(getattr(trainer, "_resign_cooldown", 0))
    stats["replay_reset"] = buffer_reset

    t1 = time.time()
    if (
        trainer.device.type == "cuda"
        and C.LOG.CUDA_EMPTY_CACHE_EVERY_ITERS > 0
        and (trainer.iteration % C.LOG.CUDA_EMPTY_CACHE_EVERY_ITERS == 0)
    ):
        torch.cuda.empty_cache()

    losses: list[tuple[torch.Tensor, torch.Tensor]] = []
    min_batch_samples = max(1, trainer.train_batch_size // 2)
    new_examples = int(game_stats.get("moves", 0))
    target_train_samples = int(C.TRAIN.TARGET_TRAIN_SAMPLES_PER_NEW * max(1, new_examples))
    num_steps = int(np.ceil(target_train_samples / max(1, trainer.train_batch_size)))
    num_steps = max(C.TRAIN.UPDATE_STEPS_MIN, min(C.TRAIN.UPDATE_STEPS_MAX, num_steps))
    samples_ratio = 0.0
    try:
        buffer_size_now = int(trainer.selfplay_engine.size())
    except Exception:
        buffer_size_now = 0
    if buffer_size_now < min_batch_samples:
        num_steps = 0
    else:
        samples_ratio = (num_steps * trainer.train_batch_size) / max(1, new_examples)

    grad_norm_sum: float = 0.0
    entropy_sum: float = 0.0
    for _step in range(num_steps):
        batch = trainer.selfplay_engine.sample_batch(
            trainer.train_batch_size, recent_ratio=trainer._recent_sample_ratio
        )
        if not batch:
            continue
        states, indices_sparse, counts_sparse, values = batch
        try:
            step_result = trainer.train_step((states, indices_sparse, counts_sparse, values))
            if step_result is None:
                continue
            policy_loss_t, value_loss_t, grad_norm_val, pred_entropy = step_result
            if math.isfinite(float(grad_norm_val)):
                grad_norm_sum += float(grad_norm_val)
            entropy_sum += float(pred_entropy)
            losses.append((policy_loss_t, value_loss_t))
        except torch.OutOfMemoryError:
            prev_bs_local = int(trainer.train_batch_size)
            new_bs_local = int(max(int(C.TRAIN.BATCH_SIZE_MIN), prev_bs_local - 1024))
            if new_bs_local < prev_bs_local:
                trainer.train_batch_size = new_bs_local
                trainer.log.info(
                    f"[AUTO    ] OOM encountered; reducing train_batch_size {prev_bs_local} -> {new_bs_local}"
                )
            trainer._oom_cooldown_iters = max(int(trainer._oom_cooldown_iters), 3)
            if trainer.device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception as exc:
                    trainer.log.debug("torch.cuda.empty_cache failed: %s", exc, exc_info=True)
                try:
                    torch.cuda.reset_peak_memory_stats(trainer.device)
                except Exception as exc:
                    trainer.log.debug("reset_peak_memory_stats failed: %s", exc, exc_info=True)
            break

    train_elapsed_s = time.time() - t1
    actual_update_steps = len(losses)
    avg_policy_loss = float(torch.stack([pair[0] for pair in losses]).mean().detach().cpu()) if losses else float("nan")
    avg_value_loss = float(torch.stack([pair[1] for pair in losses]).mean().detach().cpu()) if losses else float("nan")
    try:
        buffer_size = int(trainer.selfplay_engine.size())
        buffer_capacity2 = int(trainer.selfplay_engine.get_capacity())
    except Exception:
        buffer_size = 0
        buffer_capacity2 = 1
    buffer_pct2 = (buffer_size / buffer_capacity2) * 100
    batches_per_sec = (len(losses) / max(1e-9, train_elapsed_s)) if losses else 0.0
    samples_per_sec = ((len(losses) * trainer.train_batch_size) / max(1e-9, train_elapsed_s)) if losses else 0.0
    learning_rate = trainer.optimizer.param_groups[0]["lr"]
    avg_grad_norm = (grad_norm_sum / max(1, len(losses))) if losses else 0.0
    avg_entropy = (entropy_sum / max(1, len(losses))) if losses else 0.0
    entropy_coef = _entropy_coefficient(trainer.iteration)
    sched_drift_pct = 0.0
    if C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST > 0:
        sched_drift_pct = (
            100.0 * (actual_update_steps - C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST) / C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST
            if actual_update_steps > 0
            else 0.0
        )
    if (
        trainer.iteration == 1
        and actual_update_steps > 0
        and abs(sched_drift_pct) > C.TRAIN.LR_SCHED_DRIFT_ADJUST_THRESHOLD
    ):
        remaining_iters = max(0, C.TRAIN.TOTAL_ITERATIONS - trainer.iteration)
        new_total = int(trainer.scheduler.t + remaining_iters * actual_update_steps)
        new_total = max(trainer.scheduler.t + 1, new_total)
        trainer.scheduler.set_total_steps(new_total)
        trainer.log.info(
            f"[LR      ] adjust total_steps -> {trainer.scheduler.total} (iter1 measured {actual_update_steps} vs est {C.TRAIN.LR_SCHED_STEPS_PER_ITER_EST}, drift {sched_drift_pct:+.1f}%)"
        )

    recent_pct = round(100.0 * trainer._recent_sample_ratio)

    stats.update(
        {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "learning_rate": float(learning_rate),
            "buffer_size": buffer_size,
            "buffer_percent": buffer_pct2,
            "train_time_s": train_elapsed_s,
            "batches_per_sec": batches_per_sec,
            "samples_per_sec": samples_per_sec,
            "optimizer_steps": actual_update_steps,
            "lr_sched_t": trainer.scheduler.t,
            "lr_sched_total": trainer.scheduler.total,
            "train_steps_planned": num_steps,
            "train_steps_actual": actual_update_steps,
            "samples_ratio": samples_ratio,
            "avg_grad_norm": avg_grad_norm,
            "avg_entropy": avg_entropy,
            "entropy_coef": entropy_coef,
            "train_samples": len(losses) * trainer.train_batch_size,
            "train_recent_pct": recent_pct,
            "lr_sched_drift_pct": sched_drift_pct,
            "grad_skip_count": int(getattr(trainer, "_grad_skip_count", 0)),
        }
    )
    stats.update(
        {
            "eval_requests_total": requests_total,
            "eval_cache_hits_total": cache_hits_total,
            "eval_hit_rate": hit_rate,
            "eval_batches_total": batches_total,
            "eval_positions_total": eval_positions_total,
            "eval_batch_size_max": max_batch_size,
            "eval_batch_cap": trainer._eval_batch_cap,
            "eval_coalesce_ms": trainer._eval_coalesce_ms,
            "eval_requests_delta": delta_requests,
            "eval_cache_hits_delta": delta_hits,
            "eval_batches_delta": delta_batches,
            "eval_positions_delta": delta_eval_positions,
            "eval_hit_rate_delta": hit_rate_d,
            "eval_avg_batch": avg_batch_delta,
        }
    )
    trainer._prev_eval_m = eval_metrics

    return stats


def train_step(
    trainer: Any, batch_data: tuple[list[Any], list[np.ndarray], list[np.ndarray], list[np.ndarray]]
) -> tuple[torch.Tensor, torch.Tensor, float, float] | None:
    states_u8_list, indices_i32_list, counts_u16_list, values_i8_list = batch_data
    x_u8_np = np.stack(states_u8_list).astype(np.uint8, copy=False)
    x = torch.from_numpy(x_u8_np).to(trainer.device, non_blocking=True)
    v_i8_np = np.asarray(values_i8_list, dtype=np.int8)
    v_target = (
        torch.from_numpy(v_i8_np)
        .to(trainer.device, non_blocking=True)
        .to(dtype=torch.float32)
        / float(C.DATA.VALUE_I8_SCALE)
    )
    x = x.to(dtype=torch.float32) / float(C.DATA.U8_SCALE)
    x = x.contiguous(memory_format=torch.channels_last)
    if not hasattr(trainer, "_aug_mirror_idx"):
        from augmentation import Augment as _Aug

        mirror_idx = _Aug._policy_index_permutation("mirror")
        rot180_idx = _Aug._policy_index_permutation("rot180")
        vflip_idx = _Aug._policy_index_permutation("vflip_cs")
        plane_perm = _Aug._vflip_cs_plane_permutation(x.shape[1])
        feat_idx = _Aug._feature_plane_indices()
        _tp = feat_idx.get("turn_plane")
        trainer._turn_plane_idx = int(_tp if _tp is not None else int(x.shape[1]))
        trainer._aug_mirror_idx = torch.tensor(mirror_idx, dtype=torch.long, device=trainer.device)
        trainer._aug_rot180_idx = torch.tensor(rot180_idx, dtype=torch.long, device=trainer.device)
        trainer._aug_vflip_idx = torch.tensor(vflip_idx, dtype=torch.long, device=trainer.device)
        trainer._aug_vflip_plane_perm = torch.tensor(plane_perm, dtype=torch.long, device=trainer.device)

        def _inv_perm(perm: Sequence[int]) -> np.ndarray:
            perm_np = np.asarray(perm, dtype=np.int64)
            inv = np.empty_like(perm_np)
            inv[perm_np] = np.arange(perm_np.size, dtype=np.int64)
            return inv

        trainer._aug_mirror_inv_np = _inv_perm(mirror_idx.tolist())
        trainer._aug_rot180_inv_np = _inv_perm(rot180_idx.tolist())
        trainer._aug_vflip_inv_np = _inv_perm(vflip_idx.tolist())

    if np.random.rand() < C.AUGMENT.MIRROR_PROB:
        x = torch.flip(x, dims=[-1])
        indices_i32_list = [
            trainer._aug_mirror_inv_np[np.asarray(idx, dtype=np.int64)].astype(np.int32) for idx in indices_i32_list
        ]
    if np.random.rand() < C.AUGMENT.ROT180_PROB:
        x = torch.flip(x, dims=[-1, -2])
        indices_i32_list = [
            trainer._aug_rot180_inv_np[np.asarray(idx, dtype=np.int64)].astype(np.int32) for idx in indices_i32_list
        ]
    if np.random.rand() < C.AUGMENT.VFLIP_CS_PROB:
        x = torch.flip(x, dims=[-2])
        x = x.index_select(1, trainer._aug_vflip_plane_perm)
        if 0 <= getattr(trainer, "_turn_plane_idx", x.shape[1]) < x.shape[1]:
            x[:, trainer._turn_plane_idx] = 1.0 - x[:, trainer._turn_plane_idx]
        indices_i32_list = [
            trainer._aug_vflip_inv_np[np.asarray(idx, dtype=np.int64)].astype(np.int32) for idx in indices_i32_list
        ]
        v_target = -v_target
    trainer.model.train()
    autocast_device = "cuda" if trainer.device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.bfloat16
    with torch.autocast(
        device_type=autocast_device,
        dtype=autocast_dtype,
        enabled=getattr(trainer, "_amp_enabled", False),
    ):
        policy_logits, value_pred = trainer.model(x)
        value_pred = value_pred.squeeze(-1)
        log_probs = F.log_softmax(policy_logits, dim=1)
        rows_np_list: list[np.ndarray] = []
        cols_np_list: list[np.ndarray] = []
        cnts_np_list: list[np.ndarray] = []
        for i, (idx_arr, cnt_arr) in enumerate(zip(indices_i32_list, counts_u16_list, strict=False)):
            idx_np = np.asarray(idx_arr, dtype=np.int64)
            cnt_np = np.asarray(cnt_arr, dtype=np.float32)
            if idx_np.size == 0:
                continue
            valid = (idx_np >= 0) & (idx_np < log_probs.shape[1]) & (cnt_np > 0)
            if not np.any(valid):
                continue
            idx_np = idx_np[valid]
            cnt_np = cnt_np[valid]
            cols_np_list.append(idx_np)
            cnts_np_list.append(cnt_np)
            rows_np_list.append(np.full((idx_np.size,), i, dtype=np.int64))
        if rows_np_list:
            cols_cpu = torch.from_numpy(np.concatenate(cols_np_list, axis=0))
            cnts_cpu = torch.from_numpy(np.concatenate(cnts_np_list, axis=0).astype(np.float32, copy=False))
            rows_cpu = torch.from_numpy(np.concatenate(rows_np_list, axis=0))
            cols = cols_cpu.to(trainer.device, non_blocking=True, dtype=torch.long)
            cnts = cnts_cpu.to(trainer.device, non_blocking=True, dtype=torch.float32)
            rows = rows_cpu.to(trainer.device, non_blocking=True, dtype=torch.long)
            denom_rows = torch.zeros((x.shape[0],), device=trainer.device, dtype=torch.float32)
            denom_rows.index_add_(0, rows, cnts)
            denom = denom_rows.index_select(0, rows).clamp_min(1e-9)
            weights = cnts / denom
            eps = float(C.TRAIN.LOSS_POLICY_LABEL_SMOOTH)
            if eps > 0.0:
                ones = torch.ones_like(weights)
                m_rows = torch.zeros((x.shape[0],), device=trainer.device, dtype=torch.float32)
                m_rows.index_add_(0, rows, ones)
                m = m_rows.index_select(0, rows).clamp_min(1.0)
                weights = (1.0 - eps) * weights + (eps / m)
            sel = log_probs[rows, cols]
            per_entry = -weights * sel
            loss_rows = torch.zeros((x.shape[0],), device=trainer.device, dtype=torch.float32)
            loss_rows.index_add_(0, rows, per_entry)
            policy_loss = loss_rows.mean()
        else:
            policy_loss = torch.tensor(0.0, device=trainer.device, dtype=torch.float32)
        value_loss = F.mse_loss(value_pred, v_target)
        entropy_coef = _entropy_coefficient(trainer.iteration)
        entropy = (-(F.softmax(policy_logits, dim=1) * F.log_softmax(policy_logits, dim=1)).sum(dim=1)).mean()
        value_loss_weight = _value_loss_weight(trainer.iteration)
        total_loss = (
            C.TRAIN.LOSS_POLICY_WEIGHT * policy_loss
            + value_loss_weight * value_loss
            - entropy_coef * entropy
        )
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.scaler.scale(total_loss).backward()
    trainer.scaler.unscale_(trainer.optimizer)
    grad_total_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), C.TRAIN.GRAD_CLIP_NORM)
    if torch.is_tensor(grad_total_norm):
        grad_finite = bool(torch.isfinite(grad_total_norm))
        grad_value = float(grad_total_norm.detach().cpu())
    else:
        grad_value = float(grad_total_norm)
        grad_finite = math.isfinite(grad_value)
    if not grad_finite:
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.scaler.update()
        trainer.scheduler.step()
        if hasattr(trainer, "_grad_skip_count"):
            trainer._grad_skip_count += 1
        return None

    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    trainer.scheduler.step()
    if trainer.ema is not None:
        trainer.ema.update(trainer.model)
    return (
        policy_loss.detach(),
        value_loss.detach(),
        grad_value,
        float(entropy.detach().cpu()),
    )
