from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class StudentTransformerEncoder(nn.Module):
    """Input [B, 200, 32] -> Output student embedding [B, 128]."""

    def __init__(self, feature_dim: int = 32, d_model: int = 128, nhead: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(sequence)
        h = self.pos(h)
        h = self.encoder(h)
        return self.norm(h[:, -1, :])
