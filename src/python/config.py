"""Training configuration for the Hybrid Chess AI project."""

from __future__ import annotations

import contextlib
import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple, cast

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

__all__ = [
    "SEED",
    "LOG",
    "TORCH",
    "DATA",
    "MODEL",
    "EVAL",
    "SELFPLAY",
    "REPLAY",
    "SAMPLING",
    "AUGMENT",
    "MCTS",
    "RESIGN",
    "TRAIN",
    "ARENA",
    "MANAGER",
    "snapshot",
    "reset",
    "load_file",
    "load_override",
]


SEED: int = 0


@dataclass(frozen=True)
class LoggingConfig:
    """Filesystem and console settings for long running experiments."""

    level: str = "INFO"  # Global logging verbosity for console output.
    runs_dir: str = "runs"  # Root directory for run artifacts.
    archive_checkpoints: bool = True  # Keep timestamped checkpoint archives per run.
    checkpoint_interval_iters: int = 16  # Save model every N training iterations.
    empty_cache_interval_iters: int = 120  # Optional CUDA cache flush cadence.
    metrics_csv_enable: bool = True  # Append training rows to CSV when True.


@dataclass(frozen=True)
class TorchConfig:
    """Runtime directives influencing PyTorch backend behaviour."""

    amp_enabled: bool = True  # Enable automatic mixed precision when CUDA is used.
    matmul_float32_precision: str = "medium"  # torch.set_float32_matmul_precision().
    threads_intra: int = 0  # Override for intra-op threads; 0 derives from CPU count.
    threads_inter: int = 0  # Override for inter-op threads; 0 derives from CPU count.
    model_channels_last: bool = True  # Store training model tensors in NHWC layout.
    eval_model_channels_last: bool = True  # Same for evaluation-only models.
    cudnn_benchmark: bool = True  # Allow cuDNN to autotune kernel selection.
    cuda_allow_tf32: bool = True  # Permit TensorFloat-32 on supported hardware.
    cudnn_allow_tf32: bool = True  # Toggle TF32 kernels inside cuDNN.
    cuda_allow_fp16_reduced_reduction: bool = True  # Optimise mixed precision sums.


@dataclass(frozen=True)
class DataConfig:
    """Numeric conventions when encoding board states and targets."""

    u8_scale: float = 255.0  # Divisor applied when normalising uint8 feature planes.
    value_i8_scale: float = 127.0  # Scale factor for signed int8 value targets.


@dataclass(frozen=True)
class ModelConfig:
    """Neural network architecture hyperparameters."""

    blocks: int = 5  # Number of residual tower blocks.
    channels: int = 96  # Channel width of each convolution.
    value_conv_channels: int = 12  # Channels in the value head convolution.
    value_hidden_dim: int = 320  # Width of the value head fully-connected layer.


@dataclass(frozen=True)
class EvalConfig:
    """Inference batching and caching settings."""

    batch_size_max: int = 192  # Maximum positions per evaluation batch.
    coalesce_ms: int = 20  # Milliseconds to wait for batching self-play requests.
    cache_capacity: int = 512  # Shared inference cache size (positions).
    arena_cache_capacity: int = 512  # Cache capacity reserved for arena runs.
    encode_cache_capacity: int = 16_000  # Cached encoded positions for self-play reuse.
    value_cache_capacity: int = 40_000  # Cached scalar evaluations.
    use_fp16_cache: bool = True  # Store cached logits in half precision where legal.


DEFAULT_CURRICULUM_FENS: Tuple[str, ...] = (
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "r2q1rk1/pp3ppp/2n1pn2/2bp4/3P4/2P1PN2/PP1N1PPP/R1BQ1RK1 b - - 0 10",
    "2r2rk1/1bqn1ppp/p2ppn2/1p6/3NP3/1BN1B3/PPP2PPP/2KR3R w - - 0 13",
    "r1bq1rk1/ppp1bppp/2nppn2/8/2B1P3/2NP1N2/PPPQ1PPP/R1B2RK1 w - - 0 9",
    "2rq1rk1/pp2bppp/2n1pn2/2bp4/2P5/1PNPB3/PB2NPPP/R2Q1RK1 w - - 0 11",
)


@dataclass(frozen=True)
class CurriculumConfig:
    """Optional curriculum to bias self-play openings."""

    sample_probability: float = 0.30  # Probability of using a curriculum FEN.
    fens: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_CURRICULUM_FENS)


@dataclass(frozen=True)
class SelfPlayConfig:
    """Parameters controlling self-play data generation."""

    num_workers: int = 2  # Parallel game workers launched per iteration.
    game_max_plies: int = 90  # Hard cap on plies to avoid runaway games.
    temperature_moves: int = 40  # Use high temperature for the opening N moves.
    temperature_high: float = 1.60  # Exploration temperature for early moves.
    temperature_low: float = 0.65  # Cooler temperature once the game progresses.
    deterministic_temp_eps: float = 0.01  # Temperature floor for deterministic play.
    opening_book_path: str | None = "opening_book.json"  # Path to the JSON opening book.
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)  # Optional curriculum settings.
    opening_random_moves: int = 2  # Number of plies forced to random moves to diversify.
    adjudication_enabled: bool = True  # Permit early endings once conditions are met.
    adjudication_min_plies: int = 55  # Require this many plies before adjudication.
    adjudication_value_margin: float = 0.08  # Value threshold to call the result.
    adjudication_persist_plies: int = 6  # Persistent advantage length before ending.
    adjudication_material_margin: float = 3.0  # Material threshold (in pawns) to call the result.


@dataclass(frozen=True)
class ReplayConfig:
    """Replay buffer sizing."""

    capacity: int = 8_000  # Number of games stored before overwriting oldest data.


@dataclass(frozen=True)
class SamplingConfig:
    """Batch sampling policy for mixing fresh and historical games."""

    recent_ratio: float = 0.65  # Target fraction of samples drawn from recent games.
    recent_ratio_min: float = 0.55  # Lower bound on recent sample fraction.
    recent_ratio_max: float = 0.75  # Upper bound on recent sample fraction.
    recent_window_frac: float = 0.25  # Fraction of buffer considered "recent".


@dataclass(frozen=True)
class AugmentConfig:
    """Symmetry-based data augmentation options."""

    mirror_prob: float = 0.50  # Probability of mirroring positions left-right.
    rot180_prob: float = 0.25  # Probability of rotating the board 180 degrees.
    vflip_prob: float = 0.25  # Probability of flipping ranks and swapping colours.


@dataclass(frozen=True)
class MCTSConfig:
    """Monte-Carlo Tree Search hyperparameters."""

    train_simulations: int = 96  # Simulations at the start of a self-play game.
    train_simulations_min: int = 32  # Minimum simulations in late-game decay.
    train_sim_decay_move_interval: int = 24  # Reduce sims every N moves.
    c_puct: float = 1.35  # Exploration constant.
    c_puct_base: float = 19652.0  # Base term for dynamic exploration schedule.
    c_puct_init: float = 1.55  # Initial exploration weight.
    dirichlet_alpha: float = 0.40  # Dirichlet noise concentration at the root.
    dirichlet_weight: float = 0.30  # Fraction of Dirichlet noise mixed into priors.
    fpu_reduction: float = 0.11  # First-play urgency reduction against visit count.
    visit_count_clamp: int = 65535  # Upper bound for visit counts (safety guard).


@dataclass(frozen=True)
class ResignConfig:
    """Automatic resignation policy tuned for reproducibility."""

    enabled: bool = True  # Allow the engine to resign games during self-play.
    value_threshold: float = -0.05  # Value head threshold to trigger resignation.
    min_plies: int = 22  # Wait this many plies before resigning.
    cooldown_iters: int = 4  # Hold-off period after disabling resignation.
    consecutive_required: int = 2  # Number of consecutive moves below the threshold.
    playthrough_fraction: float = 0.15  # Fraction of resignable games forced to finish.


@dataclass(frozen=True)
class TrainConfig:
    """End-to-end training loop schedule."""

    total_iterations: int = 768  # Overall training iterations for a full run.
    games_per_iter: int = 48  # Self-play games generated each iteration.
    batch_size: int = 160  # Primary training batch size (adjusted on CPU fallback).
    batch_size_min: int = 128  # Lower bound when adapting to resource pressure.
    batch_size_max: int = 192  # Upper bound guarding against OOM.
    learning_rate_init: float = 7.5e-4  # Initial LR when warmup begins.
    learning_rate_final: float = 3.0e-4  # Target LR after cosine decay.
    learning_rate_warmup_steps: int = 720  # Linear warmup duration in optimiser steps.
    lr_steps_per_iter_estimate: int = 60  # Expected optimiser steps per iteration.
    lr_restart_interval_iters: int = 0  # Cosine restart interval (0 disables restarts).
    lr_restart_decay: float = 1.0  # Scale applied when a restart occurs.
    weight_decay: float = 1.5e-4  # L2 regularisation strength.
    momentum: float = 0.90  # SGD momentum.
    grad_clip_norm: float = 1.35  # Gradient norm clipping value.
    loss_policy_weight: float = 1.0  # Weight applied to policy loss.
    loss_value_weight: float = 1.0  # Weight applied to value loss.
    loss_policy_label_smooth: float = 0.015  # Label smoothing factor for policy head.
    loss_entropy_coef: float = 3.0e-4  # Initial entropy regularisation strength.
    loss_entropy_iters: int = 256  # Iterations over which entropy is annealed.
    loss_entropy_min_coef: float = 2.0e-4  # Floor for entropy regularisation.
    ema_enabled: bool = True  # Maintain an exponential moving average of weights.
    ema_decay: float = 0.9995  # EMA decay factor.
    samples_per_new_game: float = 0.60  # Ratio of fresh samples relative to new games.
    update_steps_min: int = 24  # Minimum optimiser steps per iteration.
    update_steps_max: int = 60  # Maximum optimiser steps per iteration.


@dataclass(frozen=True)
class ArenaConfig:
    """Evaluation and gating schedule."""

    eval_every_iters: int = 8  # Frequency of arena evaluations.
    games_per_eval: int = 12  # Games played per arena round.
    mcts_simulations: int = 160  # MCTS simulations during evaluation games.
    temperature: float = 0.40  # Arena temperature for early moves.
    temperature_moves: int = 12  # Number of plies to apply temperature to.
    draw_score: float = 0.50  # Score assigned to a draw for Elo style gating.
    gate_baseline_p: float = 0.500  # Baseline win probability requirement.
    gate_margin: float = 0.004  # Minimum margin over baseline to accept candidate.
    gate_min_games: int = 12  # Minimum games before considering acceptance.
    gate_min_decisive: int = 4  # Require this many decisive results for confidence.
    gate_draw_weight: float = 0.40  # Weight of draws when computing acceptance score.
    max_game_plies: int = 110  # Safety cap on game length during evaluation.
    resign_enable: bool = False  # Disable arena resignations for unbiased scoring.


LOG = LoggingConfig()
TORCH = TorchConfig()
DATA = DataConfig()
MODEL = ModelConfig()
EVAL = EvalConfig()
SELFPLAY = SelfPlayConfig()
REPLAY = ReplayConfig()
SAMPLING = SamplingConfig()
AUGMENT = AugmentConfig()
MCTS = MCTSConfig()
RESIGN = ResignConfig()
TRAIN = TrainConfig()
ARENA = ArenaConfig()


def _apply_override(target: Any, overrides: Any) -> Any:
    if is_dataclass(target) and not isinstance(target, type) and isinstance(overrides, Mapping):
        updates: dict[str, Any] = {}
        for f in fields(target):  # type: ignore[arg-type]
            if f.name not in overrides:
                continue
            current = getattr(target, f.name)
            updates[f.name] = _apply_override(current, overrides[f.name])
        if updates:
            return replace(cast(Any, target), **updates)
        return target
    return overrides


def _clone(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value)
    return copy.deepcopy(value)


def _normalise_key(key: str) -> str:
    return key.upper()


def _load_payload_from_path(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Unsupported configuration format: {path.suffix}")
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install it with 'pip install pyyaml'.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Configuration in {path} must be a mapping")
    return payload


class ConfigManager:
    """Manages hierarchical configuration state and file overrides."""

    def __init__(self, defaults: Mapping[str, Any]) -> None:
        self._defaults = {k: _clone(v) for k, v in defaults.items()}
        self._state = {k: _clone(v) for k, v in defaults.items()}

    def reset(self) -> Mapping[str, Any]:
        self._state = {k: _clone(v) for k, v in self._defaults.items()}
        _update_module_state(self._state)
        return self.snapshot()

    def load_mapping(self, payload: Mapping[str, Any], *, replace: bool = False) -> Mapping[str, Any]:
        base = self._defaults if replace else self._state
        state = {k: _clone(v) for k, v in base.items()}
        for key, value in payload.items():
            key_norm = _normalise_key(str(key))
            current = state.get(key_norm)
            if key_norm == "SEED":
                with contextlib.suppress(Exception):
                    state[key_norm] = int(value)
                continue
            if current is None:
                state[key_norm] = value
            else:
                state[key_norm] = _apply_override(current, value)
        self._state = state
        _update_module_state(self._state)
        return self.snapshot()

    def load_file(self, path: str | os.PathLike[str], *, replace: bool = False) -> Mapping[str, Any]:
        payload = _load_payload_from_path(Path(path))
        return self.load_mapping(payload, replace=replace)

    def load_files(self, paths: Iterable[str | os.PathLike[str]]) -> Mapping[str, Any]:
        snapshot: Mapping[str, Any] = self.snapshot()
        for idx, path in enumerate(paths):
            snapshot = self.load_file(path, replace=(idx == 0))
        return snapshot

    def snapshot(self) -> Mapping[str, Any]:
        return {k: _clone(v) for k, v in self._state.items()}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self._state.items():
            if is_dataclass(value) and not isinstance(value, type):
                from dataclasses import asdict

                out[key] = asdict(value)
            else:
                out[key] = _clone(value)
        return out

    def apply_environment(self) -> None:
        env_vars = ["HYBRID_CHESS_CONFIG"]
        paths: list[str] = []
        for var in env_vars:
            raw = os.environ.get(var)
            if not raw:
                continue
            for chunk in raw.split(os.pathsep):
                candidate = chunk.strip()
                if candidate:
                    paths.append(candidate)
        for path in paths:
            try:
                self.load_file(path)
            except FileNotFoundError:
                continue


def _update_module_state(state: Mapping[str, Any]) -> None:
    globals().update(state)


DEFAULTS: dict[str, Any] = {
    "SEED": SEED,
    "LOG": LOG,
    "TORCH": TORCH,
    "DATA": DATA,
    "MODEL": MODEL,
    "EVAL": EVAL,
    "SELFPLAY": SELFPLAY,
    "REPLAY": REPLAY,
    "SAMPLING": SAMPLING,
    "AUGMENT": AUGMENT,
    "MCTS": MCTS,
    "RESIGN": RESIGN,
    "TRAIN": TRAIN,
    "ARENA": ARENA,
}


MANAGER = ConfigManager(DEFAULTS)


def snapshot() -> Mapping[str, Any]:
    return MANAGER.snapshot()


def reset() -> Mapping[str, Any]:
    return MANAGER.reset()


def load_override(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MANAGER.load_mapping(payload)


def load_file(path: str | os.PathLike[str], *, replace: bool = False) -> Mapping[str, Any]:
    return MANAGER.load_file(path, replace=replace)


MANAGER.apply_environment()
_update_module_state(MANAGER.snapshot())
