from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True)
class Training:
    iterations: int = 150
    buffer_size: int = 25000
    gpu_check_every: int = 10
    save_every: int = 25


@final
@dataclass(frozen=True)
class Optimizer:
    learning_rate: float = 0.0005
    batch_size: int = 512
    clip_norm: float = 5.0
    eta_min_mult: float = 0.1
    t_max: int = 150
    gradient_accumulation_steps: int = 2


@final
@dataclass(frozen=True)
class Game:
    games_per_iteration: int = 8
    max_moves_per_game: int = 45
    high_temperature: float = 1.0
    low_temperature: float = 0.1
    temp_moves: int = 20
    exploration_epsilon: float = 0.15
    resignation_threshold: float = -0.85
    resignation_disabled_rate: float = 0.1


@final
@dataclass(frozen=True)
class MCTS:
    c_puct: float = 1.25
    simulations: int = 60
    batch_size: int = 512
    default_prior: float = 0.001
    visit_divisor: int = 4
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    virtual_loss: float = 3.0


@final
@dataclass(frozen=True)
class Model:
    hidden_dim: int = 256
    num_layers: int = 10
    num_heads: int = 8
    move_space_size: int = 1858
    encoding_channels: int = 8


@final
@dataclass(frozen=True)
class System:
    policy_epsilon: float = 1e-8
    slow_operation_threshold: float = 1.0
    percentage_multiplier: int = 100
    bytes_to_gb: int = 1024**3


@final
@dataclass(frozen=True)
class Logging:
    verbosity: str = "normal"
    detail_interval: int = 100
    use_colors: bool = True


@final
@dataclass(frozen=True)
class Config:
    training: Training = Training()
    optimizer: Optimizer = Optimizer()
    game: Game = Game()
    mcts: MCTS = MCTS()
    model: Model = Model()
    system: System = System()
    logging: Logging = Logging()

    @property
    def board_encoding_size(self) -> int:
        return 64 * self.model.encoding_channels


config = Config()
