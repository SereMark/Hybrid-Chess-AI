import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerChessModel(nn.Module):
    def __init__(self, num_moves: int, input_channels: int = 25, d_model: int = 128, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.num_moves = num_moves
        self.d_model = d_model
        self.seq_len = 64

        # Patch embedding
        self.patch_embedding = nn.Linear(input_channels, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len)

        # Transformer encoder
        encoder_layer = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.ModuleList([encoder_layer for _ in range(num_layers)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(2 * self.seq_len, num_moves)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(self.seq_len, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        x = x.view(batch_size, 25, 8 * 8).permute(0, 2, 1)

        x = self.patch_embedding(x)
        x = self.pos_encoder(x)

        x = x.permute(1, 0, 2)

        for layer in self.transformer_encoder:
            x = layer(x)

        x = x.permute(1, 0, 2)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value