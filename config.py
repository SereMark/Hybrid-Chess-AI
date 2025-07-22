from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True)
class Training:
    iterations: int = 32
    buffer_size: int = 3000
    gpu_check_every: int = 5
    save_every: int = 5


@final
@dataclass(frozen=True)
class Optimizer:
    learning_rate: float = 0.001
    batch_size: int = 384
    clip_norm: float = 5.0
    eta_min_mult: float = 0.1
    t_max: int = 32


@final
@dataclass(frozen=True)
class Game:
    games_per_iteration: int = 25
    max_moves_per_game: int = 40
    high_temperature: float = 1.0
    low_temperature: float = 0.1
    temp_moves: int = 10
    exploration_epsilon: float = 0.05


@final
@dataclass(frozen=True)
class MCTS:
    c_puct: float = 1.0
    simulations: int = 35
    batch_size: int = 96
    default_prior: float = 0.001
    log_batch_size: int = 32
    slow_threshold: float = 5.0
    min_log_size: int = 8
    visit_divisor: int = 4


@final
@dataclass(frozen=True)
class Model:
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.1
    weight_init_gain: float = 1.0

    board_size: int = 8
    total_squares: int = 64
    piece_types: int = 6
    colors: int = 2
    move_space_size: int = 4208
    encoding_channels: int = 18

    turn_channel: int = 12
    castling_white_ks_channel: int = 13
    castling_white_qs_channel: int = 14
    castling_black_ks_channel: int = 15
    castling_black_qs_channel: int = 16
    en_passant_channel: int = 17

    white_promotion_rank_start: int = 56
    white_promotion_rank_end: int = 63
    black_promotion_rank_start: int = 0
    black_promotion_rank_end: int = 7


@final
@dataclass(frozen=True)
class System:
    policy_epsilon: float = 1e-8
    slow_operation_threshold: float = 1.0
    loss_history_size: int = 100
    training_batch_threshold: int = 64
    loss_trend_window: int = 10
    percentage_multiplier: int = 100
    bytes_to_gb: int = 1024**3


@final
@dataclass(frozen=True)
class Config:
    training: Training = Training()
    optimizer: Optimizer = Optimizer()
    game: Game = Game()
    mcts: MCTS = MCTS()
    model: Model = Model()
    system: System = System()

    @property
    def board_encoding_size(self) -> int:
        return self.model.total_squares * self.model.encoding_channels

    @property
    def token_dim(self) -> int:
        return self.model.hidden_dim

    @property
    def visit_threshold(self) -> int:
        return self.mcts.simulations // self.mcts.visit_divisor


config = Config()
