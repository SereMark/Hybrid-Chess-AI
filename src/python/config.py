"""Training configuration for the Hybrid Chess AI project."""
from __future__ import annotations

import contextlib
import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple, cast

try:
    import yaml
except Exception:
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
    """Filesystem and console settings for long runs."""
    level: str = "INFO"
    runs_dir: str = "runs"
    archive_checkpoints: bool = True
    checkpoint_interval_iters: int = 16
    empty_cache_interval_iters: int = 120
    metrics_csv_enable: bool = True


@dataclass(frozen=True)
class TorchConfig:
    """Runtime directives influencing PyTorch backend behaviour."""
    amp_enabled: bool = True
    matmul_float32_precision: str = "medium"
    threads_intra: int = 0
    threads_inter: int = 0
    model_channels_last: bool = True
    eval_model_channels_last: bool = True
    cudnn_benchmark: bool = True
    cuda_allow_tf32: bool = True
    cudnn_allow_tf32: bool = True
    cuda_allow_fp16_reduced_reduction: bool = True


@dataclass(frozen=True)
class DataConfig:
    """Numeric conventions for encoding states and targets."""
    u8_scale: float = 255.0
    value_i8_scale: float = 127.0


@dataclass(frozen=True)
class ModelConfig:
    """Network architecture hyperparameters."""
    blocks: int = 5
    channels: int = 96
    value_conv_channels: int = 12
    value_hidden_dim: int = 320


@dataclass(frozen=True)
class EvalConfig:
    """Inference batching and caching settings."""
    batch_size_max: int = 192
    coalesce_ms: int = 20
    cache_capacity: int = 512
    arena_cache_capacity: int = 512
    encode_cache_capacity: int = 16_000
    value_cache_capacity: int = 40_000
    use_fp16_cache: bool = True


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
    sample_probability: float = 0.30
    fens: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_CURRICULUM_FENS)


@dataclass(frozen=True)
class SelfPlayConfig:
    """Self-play generation parameters."""
    num_workers: int = 2
    game_max_plies: int = 90
    temperature_moves: int = 40
    temperature_high: float = 1.60
    temperature_low: float = 0.65
    deterministic_temp_eps: float = 0.01
    opening_book_path: str | None = "opening_book.json"
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    opening_random_moves: int = 2
    adjudication_enabled: bool = True
    adjudication_min_plies: int = 55
    adjudication_value_margin: float = 0.08
    adjudication_persist_plies: int = 6
    adjudication_material_margin: float = 3.0


@dataclass(frozen=True)
class ReplayConfig:
    """Replay buffer sizing."""
    capacity: int = 8_000


@dataclass(frozen=True)
class SamplingConfig:
    """Batch sampling policy mixing fresh and historical games."""
    recent_ratio: float = 0.65
    recent_ratio_min: float = 0.55
    recent_ratio_max: float = 0.75
    recent_window_frac: float = 0.25


@dataclass(frozen=True)
class AugmentConfig:
    """Symmetry-based data augmentation options."""
    mirror_prob: float = 0.50
    rot180_prob: float = 0.25
    vflip_prob: float = 0.25


@dataclass(frozen=True)
class MCTSConfig:
    """Monte-Carlo Tree Search hyperparameters."""
    train_simulations: int = 96
    train_simulations_min: int = 32
    train_sim_decay_move_interval: int = 24
    c_puct: float = 1.35
    c_puct_base: float = 19652.0
    c_puct_init: float = 1.55
    dirichlet_alpha: float = 0.40
    dirichlet_weight: float = 0.30
    fpu_reduction: float = 0.11
    visit_count_clamp: int = 65535


@dataclass(frozen=True)
class ResignConfig:
    """Automatic resignation policy."""
    enabled: bool = True
    value_threshold: float = -0.05
    min_plies: int = 22
    cooldown_iters: int = 4
    consecutive_required: int = 2
    playthrough_fraction: float = 0.15


@dataclass(frozen=True)
class TrainConfig:
    """End-to-end training schedule."""
    total_iterations: int = 768
    games_per_iter: int = 48
    batch_size: int = 160
    batch_size_min: int = 128
    batch_size_max: int = 192
    learning_rate_init: float = 7.5e-4
    learning_rate_final: float = 3.0e-4
    learning_rate_warmup_steps: int = 720
    lr_steps_per_iter_estimate: int = 60
    lr_restart_interval_iters: int = 0
    lr_restart_decay: float = 1.0
    weight_decay: float = 1.5e-4
    momentum: float = 0.90
    grad_clip_norm: float = 1.35
    loss_policy_weight: float = 1.0
    loss_value_weight: float = 1.0
    loss_policy_label_smooth: float = 0.015
    loss_entropy_coef: float = 3.0e-4
    loss_entropy_iters: int = 256
    loss_entropy_min_coef: float = 2.0e-4
    ema_enabled: bool = True
    ema_decay: float = 0.9995
    samples_per_new_game: float = 0.60
    update_steps_min: int = 24
    update_steps_max: int = 60


@dataclass(frozen=True)
class ArenaConfig:
    """Evaluation and gating schedule."""
    eval_every_iters: int = 8
    games_per_eval: int = 12
    mcts_simulations: int = 160
    temperature: float = 0.40
    temperature_moves: int = 12
    draw_score: float = 0.50
    gate_baseline_p: float = 0.500
    gate_margin: float = 0.004
    gate_min_games: int = 12
    gate_min_decisive: int = 4
    gate_draw_weight: float = 0.40
    max_game_plies: int = 110
    resign_enable: bool = False


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
        for f in fields(target):
            if f.name not in overrides:
                continue
            current = getattr(target, f.name)
            updates[f.name] = _apply_override(current, overrides[f.name])
        return replace(cast(Any, target), **updates) if updates else target
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
    """Manages configuration state and file overrides."""

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
            state[key_norm] = value if current is None else _apply_override(current, value)
        self._state = state
        _update_module_state(self._state)
        return self.snapshot()

    def load_file(self, path: str | os.PathLike[str], *, replace: bool = False) -> Mapping[str, Any]:
        payload = _load_payload_from_path(Path(path))
        return self.load_mapping(payload, replace=replace)

    def load_files(self, paths: Iterable[str | os.PathLike[str]]) -> Mapping[str, Any]:
        snap: Mapping[str, Any] = self.snapshot()
        for idx, p in enumerate(paths):
            snap = self.load_file(p, replace=(idx == 0))
        return snap

    def snapshot(self) -> Mapping[str, Any]:
        return {k: _clone(v) for k, v in self._state.items()}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self._state.items():
            if is_dataclass(v) and not isinstance(v, type):
                from dataclasses import asdict
                out[k] = asdict(v)
            else:
                out[k] = _clone(v)
        return out

    def apply_environment(self) -> None:
        raw = os.environ.get("HYBRID_CHESS_CONFIG")
        if not raw:
            return
        paths = [p.strip() for p in raw.split(os.pathsep) if p.strip()]
        for p in paths:
            try:
                self.load_file(p)
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