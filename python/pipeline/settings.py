from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import config as C


@dataclass(frozen=True, slots=True)
class SelfPlaySettings:
    num_workers: int
    game_max_plies: int
    temp_moves: int
    temp_high: float
    temp_low: float
    color_balance_tol_pct: float
    adjudicate_enabled: bool
    adjudicate_margin_start: float
    adjudicate_margin_end: float
    adjudicate_min_start: int
    adjudicate_min_end: int
    adjudicate_decay_iter: int
    curriculum_sample_prob: float


@dataclass(frozen=True, slots=True)
class MCTSSettings:
    train_simulations_base: int
    train_simulations_min: int
    train_sim_decay_interval: int
    c_puct: float
    c_puct_base: float
    c_puct_init: float
    dirichlet_alpha: float
    dirichlet_weight: float


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    total_iterations: int
    games_per_iteration: int
    batch_size: int
    lr_init: float
    lr_final: float
    update_steps_min: int
    update_steps_max: int


@dataclass(frozen=True, slots=True)
class ArenaSettings:
    eval_every_iters: int
    games_per_eval: int
    temperature: float
    temp_moves: int
    gate_baseline_margin: float
    gate_min_games: int


@dataclass(frozen=True, slots=True)
class ResignSettings:
    enabled: bool
    threshold_init: float
    threshold_final: float
    ramp_start: int
    ramp_end: int
    min_plies_init: int
    min_plies_final: int
    disable_until_iters: int


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    csv_path: str
    arena_pgn_dir: str
    enable_csv: bool


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    selfplay: SelfPlaySettings
    mcts: MCTSSettings
    training: TrainingSettings
    arena: ArenaSettings
    resign: ResignSettings
    logging: LoggingSettings

    def to_dict(self) -> dict[str, Any]:
        return {
            "selfplay": self.selfplay,
            "mcts": self.mcts,
            "training": self.training,
            "arena": self.arena,
            "resign": self.resign,
            "logging": self.logging,
        }


def _get_attr(namespace: Any, name: str, default: Any) -> Any:
    return getattr(namespace, name, default)


def load_settings() -> PipelineSettings:
    selfplay = SelfPlaySettings(
        num_workers=int(C.SELFPLAY.NUM_WORKERS),
        game_max_plies=int(C.SELFPLAY.GAME_MAX_PLIES),
        temp_moves=int(C.SELFPLAY.TEMP_MOVES),
        temp_high=float(C.SELFPLAY.TEMP_HIGH),
        temp_low=float(C.SELFPLAY.TEMP_LOW),
        color_balance_tol_pct=float(_get_attr(C.SELFPLAY, "COLOR_BALANCE_TOLERANCE_PCT", 5.0)),
        adjudicate_enabled=bool(_get_attr(C.SELFPLAY, "ADJUDICATE_ENABLED", False)),
        adjudicate_margin_start=float(_get_attr(C.SELFPLAY, "ADJUDICATE_MARGIN_START", 0.0)),
        adjudicate_margin_end=float(_get_attr(C.SELFPLAY, "ADJUDICATE_MARGIN_END", 0.0)),
        adjudicate_min_start=int(_get_attr(C.SELFPLAY, "ADJUDICATE_MIN_PLIES_START", 0)),
        adjudicate_min_end=int(_get_attr(C.SELFPLAY, "ADJUDICATE_MIN_PLIES_END", 0)),
        adjudicate_decay_iter=int(_get_attr(C.SELFPLAY, "ADJUDICATE_MARGIN_DECAY_ITER", 0)),
        curriculum_sample_prob=float(_get_attr(C.SELFPLAY, "CURRICULUM_SAMPLE_PROB", 0.0)),
    )

    mcts = MCTSSettings(
        train_simulations_base=int(C.MCTS.TRAIN_SIMULATIONS_BASE),
        train_simulations_min=int(C.MCTS.TRAIN_SIMULATIONS_MIN),
        train_sim_decay_interval=int(C.MCTS.TRAIN_SIM_DECAY_MOVE_INTERVAL),
        c_puct=float(C.MCTS.C_PUCT),
        c_puct_base=float(C.MCTS.C_PUCT_BASE),
        c_puct_init=float(C.MCTS.C_PUCT_INIT),
        dirichlet_alpha=float(C.MCTS.DIRICHLET_ALPHA),
        dirichlet_weight=float(C.MCTS.DIRICHLET_WEIGHT),
    )

    training = TrainingSettings(
        total_iterations=int(C.TRAIN.TOTAL_ITERATIONS),
        games_per_iteration=int(C.TRAIN.GAMES_PER_ITER),
        batch_size=int(C.TRAIN.BATCH_SIZE),
        lr_init=float(C.TRAIN.LR_INIT),
        lr_final=float(C.TRAIN.LR_FINAL),
        update_steps_min=int(C.TRAIN.UPDATE_STEPS_MIN),
        update_steps_max=int(C.TRAIN.UPDATE_STEPS_MAX),
    )

    arena = ArenaSettings(
        eval_every_iters=int(C.ARENA.EVAL_EVERY_ITERS),
        games_per_eval=int(C.ARENA.GAMES_PER_EVAL),
        temperature=float(C.ARENA.TEMPERATURE),
        temp_moves=int(C.ARENA.TEMP_MOVES),
        gate_baseline_margin=float(_get_attr(C.ARENA, "GATE_BASELINE_MARGIN", 0.0)),
        gate_min_games=int(_get_attr(C.ARENA, "GATE_MIN_GAMES", 0)),
    )

    resign = ResignSettings(
        enabled=bool(_get_attr(C.RESIGN, "ENABLED", True)),
        threshold_init=float(_get_attr(C.RESIGN, "VALUE_THRESHOLD_INIT", -0.05)),
        threshold_final=float(_get_attr(C.RESIGN, "VALUE_THRESHOLD", -0.1)),
        ramp_start=int(_get_attr(C.RESIGN, "VALUE_THRESHOLD_RAMP_START", 0)),
        ramp_end=int(_get_attr(C.RESIGN, "VALUE_THRESHOLD_RAMP_END", 0)),
        min_plies_init=int(_get_attr(C.RESIGN, "MIN_PLIES_INIT", 0)),
        min_plies_final=int(_get_attr(C.RESIGN, "MIN_PLIES_FINAL", 0)),
        disable_until_iters=int(_get_attr(C.RESIGN, "DISABLE_UNTIL_ITERS", 0)),
    )

    logging = LoggingSettings(
        csv_path=str(_get_attr(C.LOG, "METRICS_LOG_CSV_PATH", "")),
        arena_pgn_dir=str(_get_attr(C.LOG, "ARENA_PGN_DIR", "")),
        enable_csv=bool(_get_attr(C.LOG, "METRICS_LOG_CSV_ENABLE", False)),
    )

    return PipelineSettings(
        selfplay=selfplay,
        mcts=mcts,
        training=training,
        arena=arena,
        resign=resign,
        logging=logging,
    )
