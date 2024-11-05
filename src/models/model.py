import torch, torch.nn as nn, src.utils.chess_utils as chess_utils
from typing import Tuple, Optional

class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        bottleneck_size = max(channel // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, bottleneck_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_size, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale.expand_as(x)

class SEResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 16) -> None:
        super().__init__()
        mid_channels = out_channels // 4
        self.conv_block = self._make_conv_block(in_channels, mid_channels, out_channels, stride)
        self.se = SELayer(out_channels, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = self._make_downsample(in_channels, out_channels, stride)

    def _make_conv_block(self, in_channels, mid_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def _make_downsample(self, in_channels, out_channels, stride):
        if stride != 1 or in_channels != out_channels:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample else x
        out = self.conv_block(x)
        out = self.se(out)
        return self.relu(out + identity)

class ChessModel(nn.Module):
    def __init__(self, filters: int = 64, res_blocks: int = 10, num_moves: Optional[int] = None) -> None:
        super().__init__()
        self.num_moves = num_moves or chess_utils.TOTAL_MOVES
        self.initial_block = self._make_initial_block(filters)
        self.residual_layers = nn.Sequential(*[SEResidualUnit(filters, filters) for _ in range(res_blocks)])
        self.policy_head = self._make_policy_head(filters)
        self.value_head = self._make_value_head(filters)
        self._initialize_weights()

    def _make_initial_block(self, filters):
        return nn.Sequential(
            nn.Conv2d(20, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )

    def _make_policy_head(self, filters):
        return nn.Sequential(
            nn.Conv2d(filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, self.num_moves)
        )

    def _make_value_head(self, filters):
        return nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_block(x)
        x = self.residual_layers(x)
        policy_output = self.policy_head(x)
        value_output = self.value_head(x)
        return policy_output, value_output