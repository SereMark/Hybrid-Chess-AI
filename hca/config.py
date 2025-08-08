import os
from dataclasses import dataclass
from typing import Tuple

try:
    import chesscore as _chessai

    _INPUT_PLANES = int(getattr(_chessai, "INPUT_PLANES", 119))
    _PLANES_PER_POSITION = int(getattr(_chessai, "PLANES_PER_POSITION", 14))
except Exception:
    _INPUT_PLANES = 119
    _PLANES_PER_POSITION = 14


def _env_int(key: str, default: int) -> int:
    value = os.getenv(f"HCAI_{key}")
    return int(value) if value is not None else default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(f"HCAI_{key}")
    return float(value) if value is not None else default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(f"HCAI_{key}")
    if value is None:
        return default
    return value not in ("0", "false", "False", "no", "No")


@dataclass(frozen=True)
class Config:
    blocks: int = _env_int("BLOCKS", 6)
    channels: int = _env_int("CHANNELS", 128)
    input_planes: int = _INPUT_PLANES
    policy_output: int = 73 * 64
    planes_per_position: int = _PLANES_PER_POSITION

    simulations_train: int = _env_int("SIMULATIONS_TRAIN", 128)
    simulations_eval: int = _env_int("SIMULATIONS_EVAL", 800)
    c_puct: float = _env_float("C_PUCT", 1.2)
    c_puct_base: float = _env_float("C_PUCT_BASE", 19652.0)
    c_puct_init: float = _env_float("C_PUCT_INIT", 1.25)
    dirichlet_alpha: float = _env_float("DIRICHLET_ALPHA", 0.3)
    dirichlet_weight: float = _env_float("DIRICHLET_WEIGHT", 0.25)
    mcts_min_sims: int = _env_int("MCTS_MIN_SIMS", 32)

    batch_size: int = _env_int("BATCH_SIZE", 256)
    learning_rate_init: float = _env_float("LEARNING_RATE_INIT", 0.02)
    learning_rate_schedule: Tuple[Tuple[int, float], ...] = (
        (400, 0.01),
        (800, 0.001),
        (1200, 0.0001),
    )
    weight_decay: float = _env_float("WEIGHT_DECAY", 1e-4)
    momentum: float = _env_float("MOMENTUM", 0.9)
    gradient_clip: float = _env_float("GRADIENT_CLIP", 1.0)

    games_per_iteration: int = _env_int("GAMES_PER_ITERATION", 300)
    buffer_size: int = _env_int("BUFFER_SIZE", 80_000)
    temp_moves: int = _env_int("TEMP_MOVES", 20)
    temp_high: float = _env_float("TEMP_HIGH", 1.0)
    temp_low: float = _env_float("TEMP_LOW", 0.01)
    history_length: int = _env_int("HISTORY_LENGTH", 8)

    iterations: int = _env_int("ITERATIONS", 1000)
    train_steps_per_iter: int = _env_int("TRAIN_STEPS_PER_ITER", 512)
    checkpoint_freq: int = _env_int("CHECKPOINT_FREQ", 100)

    policy_weight: float = _env_float("POLICY_WEIGHT", 1.0)
    value_weight: float = _env_float("VALUE_WEIGHT", 1.0)

    selfplay_workers: int = _env_int("SELFPLAY_WORKERS", os.cpu_count() or 12)
    eval_max_batch: int = _env_int("EVAL_MAX_BATCH", 1024)
    eval_batch_timeout_ms: int = _env_int("EVAL_BATCH_TIMEOUT_MS", 4)
    eval_cache_size: int = _env_int("EVAL_CACHE_SIZE", 100_000)
    arena_games: int = _env_int("ARENA_GAMES", 100)
    arena_win_rate: float = _env_float("ARENA_WIN_RATE", 0.55)
    arena_temperature: float = _env_float("ARENA_TEMPERATURE", 0.25)
    arena_temp_moves: int = _env_int("ARENA_TEMP_MOVES", 8)
    arena_dirichlet_weight: float = _env_float("ARENA_DIRICHLET_WEIGHT", 0.03)
    resign_threshold: float = _env_float("RESIGN_THRESHOLD", -0.95)
    resign_consecutive: int = _env_int("RESIGN_CONSECUTIVE", 3)
    use_torch_compile: bool = _env_bool("USE_TORCH_COMPILE", True)
    use_channels_last: bool = _env_bool("USE_CHANNELS_LAST", True)
    augment_mirror: bool = _env_bool("AUGMENT_MIRROR", True)
    augment_rotate180: bool = _env_bool("AUGMENT_ROT180", True)
    augment_vflip_cs: bool = _env_bool("AUGMENT_VFLIP_CS", True)
    augment_mirror_prob: float = _env_float("AUGMENT_MIRROR_PROB", 0.5)
    augment_rot180_prob: float = _env_float("AUGMENT_ROT180_PROB", 0.25)
    augment_vflip_cs_prob: float = _env_float("AUGMENT_VFLIP_CS_PROB", 0.25)


CONFIG = Config()
