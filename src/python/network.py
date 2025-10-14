"""Neural network for Hybrid Chess AI."""
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

__all__ = ["BOARD_SIZE", "NSQUARES", "INPUT_PLANES", "POLICY_OUTPUT", "ResidualBlock", "ChessNet"]


class ResidualBlock(nn.Module):
    """2×Conv3x3 + BN + ReLU with skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # NCHW
        s = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + s)


class ChessNet(nn.Module):
    """ResNet-style policy+value head."""

    def __init__(self, num_blocks: int | None = None, channels: int | None = None) -> None:
        super().__init__()
        b = int(C.MODEL.blocks if num_blocks is None else num_blocks)
        c = int(C.MODEL.channels if channels is None else channels)
        pplanes = POLICY_OUTPUT // NSQUARES

        self.conv_in = nn.Conv2d(INPUT_PLANES, c, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(c)
        self.residual_stack = nn.Sequential(*[ResidualBlock(c) for _ in range(b)])

        # Policy head
        self.policy_conv = nn.Conv2d(c, pplanes, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(pplanes)
        self.policy_fc = nn.Linear(pplanes * NSQUARES, POLICY_OUTPUT)

        # Value head
        vch = int(C.MODEL.value_conv_channels)
        vhid = int(C.MODEL.value_hidden_dim)
        self.value_conv = nn.Conv2d(c, vch, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(vch)
        self.value_fc1 = nn.Linear(vch * NSQUARES, vhid)
        self.value_fc2 = nn.Linear(vhid, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.residual_stack(x)

        p = F.relu(self.policy_bn(self.policy_conv(x))).flatten(1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return policy_logits, value