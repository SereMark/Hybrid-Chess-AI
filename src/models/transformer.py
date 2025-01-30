import torch
import torch.nn as nn

class SimpleTransformerChessModel(nn.Module):
    def __init__(self, num_moves, feature_dim=18, d_model=96, nhead=3, num_layers=6, dim_feedforward=384, dropout=0.1):
        super().__init__()
        self.num_moves = num_moves
        self.d_model = d_model
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 64, d_model))
        e = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, 'gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(e, num_layers)
        self.attention_pool = nn.Linear(d_model, 1)
        self.policy_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, num_moves))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1), nn.Tanh())
        nn.init.zeros_(self.pos_embedding)
    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        w = torch.softmax(self.attention_pool(x), 1)
        p = (x*w).sum(1)
        return self.policy_head(p), self.value_head(p)