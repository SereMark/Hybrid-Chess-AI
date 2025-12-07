from __future__ import annotations

import contextlib
import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple, cast

try:
    import yaml  # type: ignore[import-untyped]
except Exception:
    yaml = None

SEED: int = 0


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    runs_dir: str = "runs"
    archive_checkpoints: bool = True
    checkpoint_interval_iters: int = 16
    empty_cache_interval_iters: int = 120
    metrics_csv_enable: bool = True


@dataclass(frozen=True)
class TorchConfig:
    amp_enabled: bool = True
    matmul_float32_precision: str = "medium"
    cuda_matmul_fp32_precision: str = "tf32"
    cudnn_conv_fp32_precision: str = "tf32"
    threads_intra: int = 0
    threads_inter: int = 0
    model_channels_last: bool = True
    eval_model_channels_last: bool = True
    cudnn_benchmark: bool = True
    cuda_allow_fp16_reduced_reduction: bool = True


@dataclass(frozen=True)
class DataConfig:
    u8_scale: float = 255.0
    value_i8_scale: float = 127.0


@dataclass(frozen=True)
class ModelConfig:
    blocks: int = 6
    channels: int = 96
    value_conv_channels: int = 12
    value_hidden_dim: int = 256


@dataclass(frozen=True)
class EvalConfig:
    batch_size_max: int = 256
    coalesce_ms: int = 15
    cache_capacity: int = 512
    arena_cache_capacity: int = 512
    encode_cache_capacity: int = 32_000
    value_cache_capacity: int = 64_000
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
    sample_probability: float = 0.30
    fens: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_CURRICULUM_FENS)


@dataclass(frozen=True)
class SelfPlayConfig:
    num_workers: int = 3
    game_max_plies: int = 100
    temperature_moves: int = 30
    temperature_high: float = 1.50
    temperature_low: float = 0.60
    deterministic_temp_eps: float = 0.01
    opening_book_path: str | None = "opening_book.json"
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    opening_random_moves: int = 2
    adjudication_enabled: bool = True
    adjudication_min_plies: int = 55
    adjudication_value_margin: float = 0.08
    adjudication_persist_plies: int = 6
    adjudication_material_margin: float = 3.0
    adjudication_warmup_iters: int = 32
    adjudication_ramp_iters: int = 96
    adjudication_min_plies_init: int = 120
    adjudication_value_margin_init: float = 0.35
    adjudication_persist_init: int = 12
    adjudication_material_margin_init: float = 6.0
    simulation_scale_min: float = 0.2
    simulation_scale_max: float = 1.0
    simulation_scale_buffer_target: float = 0.35


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int = 40_000


@dataclass(frozen=True)
class SamplingConfig:
    recent_ratio: float = 0.75
    recent_ratio_min: float = 0.55
    recent_ratio_max: float = 0.85
    recent_window_frac: float = 0.20


@dataclass(frozen=True)
class MCTSConfig:
    train_simulations: int = 128
    train_simulations_min: int = 32
    train_sim_decay_move_interval: int = 24
    c_puct: float = 1.35
    c_puct_base: float = 19652.0
    c_puct_init: float = 1.55
    dirichlet_alpha: float = 0.30
    dirichlet_weight: float = 0.25
    fpu_reduction: float = 0.2
    visit_count_clamp: int = 65535


@dataclass(frozen=True)
class ResignConfig:
    enabled: bool = True
    value_threshold: float = -0.90
    min_plies: int = 20
    cooldown_iters: int = 4
    consecutive_required: int = 2
    playthrough_fraction: float = 0.10


@dataclass(frozen=True)
class TrainConfig:
    total_iterations: int = 400
    games_per_iter: int = 64
    games_per_iter_scale_min: float = 0.5
    games_per_iter_warmup_iters: int = 4
    batch_size: int = 256
    batch_size_min: int = 128
    batch_size_max: int = 256
    learning_rate_init: float = 2.0e-3
    learning_rate_final: float = 2.0e-4
    learning_rate_warmup_steps: int = 1000
    lr_steps_per_iter_estimate: int = 80
    lr_restart_interval_iters: int = 0
    lr_restart_decay: float = 1.0
    weight_decay: float = 1.0e-4
    momentum: float = 0.90
    grad_clip_norm: float = 2.0
    loss_policy_weight: float = 1.0
    loss_value_weight: float = 1.0
    loss_policy_label_smooth: float = 0.0
    loss_entropy_coef: float = 2.0e-4
    loss_entropy_iters: int = 256
    loss_entropy_min_coef: float = 1.0e-4
    ema_enabled: bool = True
    ema_decay: float = 0.999
    samples_per_new_game: float = 1.5
    update_steps_min: int = 32
    update_steps_max: int = 128


@dataclass(frozen=True)
class ArenaConfig:
    eval_every_iters: int = 8
    games_per_eval: int = 12
    mcts_simulations: int = 160
    temperature: float = 0.40
    temperature_moves: int = 24
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
        raise ValueError(f"Nem támogatott konfigurációs formátum: {path.suffix}")
    if yaml is None:
        raise RuntimeError("A PyYAML csomag szükséges.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"A(z) {path} konfigurációjának leképezésnek (mappingnek) kell lennie")
    return payload


class ConfigManager:
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
