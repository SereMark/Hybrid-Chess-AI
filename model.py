from collections import OrderedDict
from typing import NamedTuple, Union

import chess
import torch
from config import (
    BOARD_DIM,
    BOARD_SIZE,
    CACHE_SIZE,
    HIDDEN_DIM,
    MOVE_COUNT,
    NUM_LAYERS,
    POLICY_CHANNELS,
    SQUARES_COUNT,
    VALUE_CHANNELS,
    VALUE_FC_HIDDEN,
)
from torch import nn
from torch.nn import functional


class ModelOutput(NamedTuple):
    policy: torch.Tensor
    value: torch.Tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return functional.relu(x + residual)


class ChessModel(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

        self.input_conv = nn.Conv2d(8, HIDDEN_DIM, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(HIDDEN_DIM)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(HIDDEN_DIM) for _ in range(NUM_LAYERS)]
        )

        self.policy_conv = nn.Conv2d(HIDDEN_DIM, POLICY_CHANNELS, 1)
        self.policy_bn = nn.BatchNorm2d(POLICY_CHANNELS)
        self.policy_fc = nn.Linear(POLICY_CHANNELS * SQUARES_COUNT, MOVE_COUNT)

        self.value_conv = nn.Conv2d(HIDDEN_DIM, VALUE_CHANNELS, 1)
        self.value_bn = nn.BatchNorm2d(VALUE_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_CHANNELS * SQUARES_COUNT, VALUE_FC_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_FC_HIDDEN, 1)

        self.cache = OrderedDict()
        self.reset_inference_stats()

    def reset_inference_stats(self) -> None:
        self.forward_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def forward(self, board_input: torch.Tensor) -> ModelOutput:
        self.forward_calls += 1
        x = board_input.reshape(-1, BOARD_DIM, BOARD_DIM, BOARD_DIM).permute(0, 3, 1, 2)

        x = functional.relu(self.bn_input(self.input_conv(x)))
        for block in self.residual_blocks:
            x = block(x)

        policy = functional.relu(self.policy_bn(self.policy_conv(x)))
        policy = self.policy_fc(policy.reshape(-1, POLICY_CHANNELS * SQUARES_COUNT))
        policy = functional.softmax(policy, dim=-1)

        value = functional.relu(self.value_bn(self.value_conv(x)))
        value = self.value_fc1(value.reshape(-1, VALUE_CHANNELS * SQUARES_COUNT))
        value = torch.tanh(self.value_fc2(value))

        return ModelOutput(policy=policy, value=value)

    def encode_board(
        self, board: Union[chess.Board, list[chess.Board]]
    ) -> torch.Tensor:
        boards = [board] if isinstance(board, chess.Board) else board
        batch_size = len(boards)
        tensor = torch.zeros(
            (batch_size, BOARD_SIZE), dtype=torch.float32, device=self.device
        )

        for i, b in enumerate(boards):
            board_key = b.fen()
            if board_key in self.cache:
                self.cache_hits += 1
                self.cache.move_to_end(board_key)
                tensor[i] = self.cache[board_key]
                continue
            else:
                self.cache_misses += 1

            board_tensor = torch.zeros(
                BOARD_SIZE, dtype=torch.float32, device=self.device
            )

            piece_map = b.piece_map()
            if piece_map:
                squares = torch.tensor(list(piece_map.keys()), device=self.device)
                piece_types = torch.tensor(
                    [p.piece_type - 1 for p in piece_map.values()], device=self.device
                )
                colors = torch.tensor(
                    [1.0 if p.color else -1.0 for p in piece_map.values()],
                    dtype=torch.float32,
                    device=self.device,
                )
                indices = squares * BOARD_DIM + piece_types
                board_tensor.index_put_((indices,), colors)

            turn_value = 1.0 if b.turn else 0.0
            board_tensor[6::BOARD_DIM] = turn_value

            castling_value = (
                0.1 * int(b.has_kingside_castling_rights(chess.WHITE))
                + 0.2 * int(b.has_queenside_castling_rights(chess.WHITE))
                + 0.3 * int(b.has_kingside_castling_rights(chess.BLACK))
                + 0.4 * int(b.has_queenside_castling_rights(chess.BLACK))
                + 0.01 * (b.ep_square % BOARD_DIM + 1 if b.ep_square else 0)
            )
            board_tensor[7::BOARD_DIM] = castling_value

            tensor[i] = board_tensor
            self.cache[board_key] = board_tensor.clone()
            if len(self.cache) > CACHE_SIZE:
                self.cache.popitem(last=False)

        return tensor.squeeze(0) if isinstance(board, chess.Board) else tensor

    def get_inference_stats(self) -> dict[str, float]:
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(total_cache_requests, 1) * 100

        return {
            "forward_calls": self.forward_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
            "cache_utilization": len(self.cache) / CACHE_SIZE * 100,
        }
