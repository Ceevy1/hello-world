from __future__ import annotations

import torch
from torch import nn


class SeqEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.proj(x_seq))
        return self.norm(h.mean(dim=1))


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(node_features)
        agg.index_add_(0, dst, node_features[src])
        deg = torch.bincount(dst, minlength=node_features.size(0)).clamp_min(1).unsqueeze(1)
        pooled = agg / deg
        return torch.relu(self.linear(pooled))


class StatEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicFusion(nn.Module):
    def __init__(self, seq_dim: int = 128, kg_dim: int = 64, stat_dim: int = 64, week_vocab: int = 64, week_emb_dim: int = 8) -> None:
        super().__init__()
        self.week_emb = nn.Embedding(week_vocab, week_emb_dim)
        self.gate = nn.Sequential(nn.Linear(seq_dim + kg_dim + stat_dim + week_emb_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.seq_head = nn.Linear(seq_dim, 1)
        self.kg_head = nn.Linear(kg_dim, 1)
        self.stat_head = nn.Linear(stat_dim, 1)

    def forward(self, seq_repr: torch.Tensor, kg_repr: torch.Tensor, stat_repr: torch.Tensor, week_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wfeat = self.week_emb(week_index)
        alpha = torch.softmax(self.gate(torch.cat([seq_repr, kg_repr, stat_repr, wfeat], dim=-1)), dim=-1)

        p_seq = torch.sigmoid(self.seq_head(seq_repr)).squeeze(-1)
        p_kg = torch.sigmoid(self.kg_head(kg_repr)).squeeze(-1)
        p_stat = torch.sigmoid(self.stat_head(stat_repr)).squeeze(-1)
        pred_stack = torch.stack([p_seq, p_kg, p_stat], dim=-1)
        y = torch.sum(alpha * pred_stack, dim=-1)
        return y, alpha, pred_stack


class DynamicFusionEnhanced(nn.Module):
    def __init__(self, seq_input_dim: int, stat_input_dim: int, graph_input_dim: int, seq_dim: int = 128, kg_dim: int = 64, stat_dim: int = 64) -> None:
        super().__init__()
        self.seq = SeqEncoder(seq_input_dim, d_model=seq_dim)
        self.graph = GraphEncoder(graph_input_dim, out_dim=kg_dim)
        self.stat = StatEncoder(stat_input_dim, out_dim=stat_dim)
        self.fusion = DynamicFusion(seq_dim=seq_dim, kg_dim=kg_dim, stat_dim=stat_dim)

    def forward(self, x_seq: torch.Tensor, x_stat: torch.Tensor, node_index: torch.Tensor, week_index: torch.Tensor, node_features: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_repr = self.seq(x_seq)
        graph_repr_all = self.graph(node_features, edge_index)
        kg_repr = graph_repr_all[node_index]
        stat_repr = self.stat(x_stat)
        return self.fusion(seq_repr, kg_repr, stat_repr, week_index)
