"""Neural network architecture definition for Hybrid Chess AI."""

from __future__ import annotations

import chesscore as ccore
import config as C
import torch
import torch.nn.functional as F
from torch import nn

BOARD_SIZE: int = 8
NSQUARES: int = 64
INPUT_PLANES: int = int(getattr(ccore, "INPUT_PLANES", 14 * 8 + 7))
POLICY_OUTPUT: int = int(getattr(ccore, "POLICY_SIZE", 73 * NSQUARES))

__all__ = [
    "BOARD_SIZE",
    "NSQUARES",
    "INPUT_PLANES",
    "POLICY_OUTPUT",
    "ResidualBlock",
    "ChessNet",
]


class ResidualBlock(nn.Module):
    """Two-layer residual block with BatchNorm and ReLU activations."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + skip)


class ChessNet(nn.Module):
    """ResNet-style policy and value network for chess."""

    def __init__(self, num_blocks: int | None = None, channels: int | None = None) -> None:
        super().__init__()
        num_blocks = int(num_blocks if num_blocks is not None else C.MODEL.blocks)
        channels = int(channels if channels is not None else C.MODEL.channels)
        policy_planes = POLICY_OUTPUT // NSQUARES

        self.conv_in = nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.residual_stack = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_conv = nn.Conv2d(channels, policy_planes, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_planes)
        self.policy_fc = nn.Linear(policy_planes * NSQUARES, POLICY_OUTPUT)

        self.value_conv = nn.Conv2d(channels, C.MODEL.value_conv_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(C.MODEL.value_conv_channels)
        self.value_fc1 = nn.Linear(C.MODEL.value_conv_channels * NSQUARES, C.MODEL.value_hidden_dim)
        self.value_fc2 = nn.Linear(C.MODEL.value_hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.residual_stack(x)

        policy_logits = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = policy_logits.flatten(1)
        policy_logits = self.policy_fc(policy_logits)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.flatten(1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)
        return policy_logits, value
