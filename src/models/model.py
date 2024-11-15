import torch, torch.nn as nn, src.utils.chess_utils as chess_utils
from typing import Tuple, Optional

class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        bottleneck_size = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, bottleneck_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_size, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 16) -> None:
        super().__init__()
        mid_channels = out_channels // 4
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm3 = nn.GroupNorm(num_groups=8, num_channels=mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.se = SELayer(out_channels, reduction)
        self.downsample = self._make_downsample(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)

    def _make_downsample(self, in_channels, out_channels, stride):
        if stride != 1 or in_channels != out_channels:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.norm1(x)
        out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.se(out)

        out += identity
        return out

class ChessModel(nn.Module):
    def __init__(self, filters: int = 128, res_blocks: int = 20, num_moves: Optional[int] = None) -> None:
        super().__init__()
        self.num_moves = num_moves or chess_utils.TOTAL_MOVES
        self.initial_block = nn.Sequential(
            nn.Conv2d(20, filters, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=filters),
            nn.ReLU(inplace=True)
        )
        self.residual_layers = nn.Sequential(*[SEResidualUnit(filters, filters) for _ in range(res_blocks)])
        self.attention = nn.MultiheadAttention(embed_dim=filters, num_heads=8, batch_first=True)
        self.policy_head = self._make_policy_head(filters)
        self.value_head = self._make_value_head(filters)
        self._initialize_weights()
    
    def _make_policy_head(self, filters):
        return nn.Sequential(
            nn.Conv2d(filters, 2, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=2, num_channels=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, self.num_moves)
        )
    
    def _make_value_head(self, filters):
        return nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                if module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_block(x)
        x = self.residual_layers(x)
    
        b, c, h, w = x.size()
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)
        x = x + attn_output
    
        policy_output = self.policy_head(x)
        value_output = self.value_head(x)
        return policy_output, value_output