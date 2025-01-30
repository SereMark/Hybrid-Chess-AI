import math
import torch
import torch.nn as nn
from typing import Tuple

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.uniform_(self.pos_embedding, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embedding[:, :x.size(1), :]
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.activation = nn.SiLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        x = self.norm(x)
        return x

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(x), dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled

class TransformerChessModel(nn.Module):
    def __init__(self, num_moves: int, input_channels: int = 25, d_model: int = 256, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 512, dropout: float = 0.1, max_seq_len: int = 16):
        super().__init__()
        self.num_moves = num_moves
        self.d_model = d_model
        self.seq_len = max_seq_len

        self.patch_embedding = DepthwiseSeparableConv(input_channels, d_model)

        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=self.seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.shared_norm = nn.LayerNorm(d_model)

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_moves)
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Tanh()
        )

        self.attention_pooling = AttentionPooling(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is not self.pos_encoder.pos_embedding:
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
                pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.patch_embedding(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        x = self.shared_norm(x)

        x = self.attention_pooling(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value