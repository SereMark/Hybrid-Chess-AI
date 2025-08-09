from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    blocks: int = 6
    channels: int = 128
    input_planes: int = 119
    policy_output: int = 73 * 64
    planes_per_position: int = 14

    batch_size: int = 512
    learning_rate_init: float = 0.02
    learning_rate_schedule: Tuple[Tuple[int, float], ...] = (
        (400, 0.01),
        (800, 0.001),
        (1200, 0.0001),
    )
    weight_decay: float = 1e-4
    momentum: float = 0.9
    gradient_clip: float = 1.0
    policy_weight: float = 1.0
    value_weight: float = 1.0
    value_conv_channels: int = 4
    value_hidden_dim: int = 256

    games_per_iteration: int = 180
    iterations: int = 1000
    train_steps_per_iter: int = 512
    eval_refresh_steps: int = 128
    checkpoint_freq: int = 100
    iteration_ema_alpha: float = 0.3

    simulations_train: int = 128
    mcts_min_sims: int = 32
    simulations_decay_interval: int = 40
    temp_moves: int = 30
    temp_high: float = 1.0
    temp_low: float = 0.01
    temp_deterministic_threshold: float = 0.01
    history_length: int = 8
    max_game_moves: int = 512
    buffer_size: int = 25000
    selfplay_workers: int = 10
    resign_threshold: float = -0.9
    resign_consecutive: int = 3

    eval_max_batch: int = 512
    eval_batch_timeout_ms: int = 3
    eval_cache_size: int = 20000
    eval_pin_memory: bool = True

    simulations_eval: int = 800
    arena_games: int = 60
    arena_eval_every: int = 3
    arena_openings_path: str = ""
    arena_openings_random: bool = True
    arena_accumulate: bool = True
    arena_use_noise: bool = False
    arena_temperature: float = 0.25
    arena_temp_moves: int = 8
    arena_dirichlet_weight: float = 0.03
    arena_opening_random_plies: int = 6
    arena_opening_temperature: float = 0.9
    arena_opening_temperature_epsilon: float = 1e-6
    arena_draw_score: float = 0.5
    arena_confidence: bool = True
    arena_confidence_z: float = 1.64
    arena_threshold_base: float = 0.5
    arena_win_rate: float = 0.55
    arena_sprt_enable: bool = False
    arena_sprt_alpha: float = 0.05
    arena_sprt_beta: float = 0.10
    arena_sprt_p0: float = 0.50
    arena_sprt_p1: float = 0.55
    arena_sprt_epsilon: float = 1e-9

    c_puct: float = 1.2
    c_puct_base: float = 19652.0
    c_puct_init: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25

    use_torch_compile: bool = True
    use_torch_compile_eval: bool = False
    use_channels_last: bool = True
    torch_num_threads: int = 1
    use_cudnn_benchmark: bool = True
    matmul_precision: str = "high"

    augment_mirror: bool = True
    augment_rotate180: bool = True
    augment_vflip_cs: bool = True
    augment_mirror_prob: float = 0.5
    augment_rot180_prob: float = 0.25
    augment_vflip_cs_prob: float = 0.25


CONFIG = Config()
