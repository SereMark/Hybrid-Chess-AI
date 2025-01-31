import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

class GNNChessModel(nn.Module):
    def __init__(self, num_moves, feature_dim=18, hidden_dim=128, gnn_layers=4, heads=4, dropout=0.2):
        super().__init__()
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(pyg_nn.GATConv(feature_dim, hidden_dim, heads=heads, dropout=dropout, concat=False))
        for _ in range(gnn_layers - 2): self.gnn_layers.append(pyg_nn.GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False))
        self.gnn_layers.append(pyg_nn.GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False))
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(gnn_layers)])
        self.res_proj = nn.Linear(feature_dim, hidden_dim) if feature_dim != hidden_dim else nn.Identity()
        self.global_pooling = pyg_nn.GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_moves)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def forward(self, x, edge_index, batch=None):
        if batch is None: batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.res_proj(x)
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_res = x
            x = gnn_layer(x, edge_index)
            x = self.norm_layers[i](x)
            x = F.gelu(x) + x_res
        x_pooled = self.global_pooling(x, batch)
        policy = self.policy_head(x_pooled)
        value = self.value_head(x_pooled)
        return policy, value