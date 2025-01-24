import torch
import torch.nn as nn
import math
from typing import Tuple

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.kaiming_normal_(self.pos_embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embedding[:, :x.size(1), :]
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.activation = nn.SiLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        x = self.norm(x)
        return x

class TransformerChessModel(nn.Module):
    def __init__(self, num_moves: int, input_channels: int = 25, d_model: int = 256, nhead: int = 16, num_layers: int = 12, dim_feedforward: int = 1024, dropout: float = 0.1, max_seq_len: int = 128):
        super().__init__()
        self.num_moves = num_moves
        self.d_model = d_model
        self.seq_len = max_seq_len

        # 1. Efficient Patch Embedding using Depthwise Separable Convolution
        self.patch_embedding = DepthwiseSeparableConv(input_channels, d_model)

        # 2. Positional Encoding
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=self.seq_len)

        # 3. Transformer Encoder with Pre-Layer Normalization
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Shared LayerNorm for both heads
        self.shared_norm = nn.LayerNorm(d_model)

        # 5. Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_moves)
        )

        # 6. Value Head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()
        )

        # 7. Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Parameter):
                nn.init.kaiming_normal_(module, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Patch Embedding
        x = self.patch_embedding(x)  # Shape: (batch_size, d_model, H, W)

        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, H*W, d_model)
        x = self.pos_encoder(x)           # Add positional encoding

        # 2. Transformer Encoding
        x = self.transformer_encoder(x)    # Shape: (batch_size, H*W, d_model)

        # 3. Shared LayerNorm
        x = self.shared_norm(x)           # Shape: (batch_size, H*W, d_model)

        # 4. Aggregate features (e.g., via mean pooling)
        x = x.mean(dim=1)                  # Shape: (batch_size, d_model)

        # 5. Policy and Value Heads
        policy = self.policy_head(x)       # Shape: (batch_size, num_moves)
        value = self.value_head(x)         # Shape: (batch_size, 1)

        return policy, value