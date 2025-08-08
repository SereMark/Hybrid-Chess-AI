import os
from dataclasses import dataclass
from typing import Tuple

try:
    import chessai as _chessai

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
    dirichlet_alpha: float = _env_float("DIRICHLET_ALPHA", 0.3)
    dirichlet_weight: float = _env_float("DIRICHLET_WEIGHT", 0.25)
    mcts_min_sims: int = _env_int("MCTS_MIN_SIMS", 32)

    batch_size: int = _env_int("BATCH_SIZE", 128)
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
    checkpoint_freq: int = _env_int("CHECKPOINT_FREQ", 10)

    policy_weight: float = _env_float("POLICY_WEIGHT", 1.0)
    value_weight: float = _env_float("VALUE_WEIGHT", 1.0)

    selfplay_workers: int = _env_int("SELFPLAY_WORKERS", 10)
    eval_max_batch: int = _env_int("EVAL_MAX_BATCH", 512)
    eval_batch_timeout_ms: int = _env_int("EVAL_BATCH_TIMEOUT_MS", 3)
    use_torch_compile: bool = _env_bool("USE_TORCH_COMPILE", True)
    use_channels_last: bool = _env_bool("USE_CHANNELS_LAST", True)


CONFIG = Config()
