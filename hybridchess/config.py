from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    input_planes: int = 119
    planes_per_position: int = 14
    policy_output: int = 73 * 64
    blocks: int = 6
    channels: int = 128
    value_conv_channels: int = 4
    value_hidden_dim: int = 256

    batch_size: int = 1024
    learning_rate_init: float = 0.02
    learning_rate_schedule: Tuple[Tuple[int, float], ...] = (
        (200, 0.01),
        (450, 0.003),
        (550, 0.001),
    )
    weight_decay: float = 1e-4
    momentum: float = 0.9
    gradient_clip: float = 1.0
    policy_weight: float = 1.0
    value_weight: float = 1.0

    iterations: int = 600
    games_per_iteration: int = 180
    train_steps_per_iter: int = 1024
    checkpoint_freq: int = 25
    iteration_ema_alpha: float = 0.3

    buffer_size: int = 40000
    selfplay_workers: int = 12
    history_length: int = 8
    max_game_moves: int = 512
    resign_threshold: float = -0.9
    resign_consecutive: int = 0
    temp_moves: int = 30
    temp_high: float = 1.0
    temp_low: float = 0.01
    temp_deterministic_threshold: float = 0.01

    eval_max_batch: int = 512
    eval_batch_timeout_ms: int = 8
    eval_cache_size: int = 80000

    simulations_train: int = 160
    mcts_min_sims: int = 32
    simulations_decay_interval: int = 30
    c_puct: float = 1.2
    c_puct_base: float = 19652.0
    c_puct_init: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25

    simulations_eval: int = 400
    arena_eval_every: int = 10
    arena_games: int = 100
    arena_openings_path: str = ""
    arena_temperature: float = 0.25
    arena_temp_moves: int = 8
    arena_dirichlet_weight: float = 0.03
    arena_opening_random_plies: int = 6
    arena_opening_temperature: float = 0.9
    arena_opening_temperature_epsilon: float = 1e-6
    arena_draw_score: float = 0.5
    arena_confidence_z: float = 1.0
    arena_threshold_base: float = 0.5
    arena_win_rate: float = 0.52

    augment_mirror_prob: float = 0.5
    augment_rot180_prob: float = 0.25
    augment_vflip_cs_prob: float = 0.25


CONFIG = Config()
