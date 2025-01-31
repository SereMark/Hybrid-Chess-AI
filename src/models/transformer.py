import torch, math
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.inst_norm1 = nn.InstanceNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.inst_norm2 = nn.InstanceNorm2d(out_c)
        self.res_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else None

    def forward(self, x):
        res = self.res_conv(x) if self.res_conv else x
        out = nn.GELU()(self.inst_norm1(self.conv1(x)))
        out = nn.GELU()(self.inst_norm2(self.conv2(out)))
        return out + res

class TransformerEncoderLayerPreLN(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, src, attn_mask=None, key_pad_mask=None):
        src2 = self.norm1(src)
        attn_out, _ = self.attn(src2, src2, src2, attn_mask=attn_mask, key_padding_mask=key_pad_mask)
        src = src + self.dropout1(attn_out)
        src2 = self.norm2(src)
        return src + self.ffn(src2)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:seq_len].unsqueeze(0)

class TransformerCNNChessModel(nn.Module):
    def __init__(self, num_moves, feature_dim=18, d_model=96, nhead=3, layers=6, dim_ff=192, dropout=0.1, max_pos=64):
        super().__init__()
        self.initial_conv = nn.Conv2d(feature_dim, d_model, 1)
        self.conv_blocks = nn.Sequential(
            ResidualConvBlock(d_model, d_model),
            ResidualConvBlock(d_model, d_model)
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_pos)
        encoder_layer = TransformerEncoderLayerPreLN(d_model, nhead, dim_ff, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, num_moves)
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, 1),
            nn.Tanh()
        )

    def forward(self, x, key_pad_mask=None):
        b = x.size(0)
        x = x.view(b, 8, 8, -1).permute(0, 3, 1, 2)
        x = self.initial_conv(x)
        x = self.conv_blocks(x)
        x = x.view(b, self.initial_conv.out_channels, -1).permute(0, 2, 1)
        x = x + self.pos_enc(64).to(x.device)
        x = self.transformer(x, src_key_padding_mask=key_pad_mask)
        x_pooled = torch.mean(x, dim=1)
        policy = self.policy_head(x_pooled)
        value = self.value_head(x_pooled)
        return policy, value