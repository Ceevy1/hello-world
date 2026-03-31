from __future__ import annotations

import torch
import torch.nn as nn


class LightDynamicFusion(nn.Module):
    def __init__(self, d_perf: int, d_behav: int, d_eng: int, hidden: int = 16, dropout: float = 0.5):
        super().__init__()
        self.e_perf = nn.Sequential(nn.Linear(d_perf, hidden), nn.ReLU())
        self.e_behav = nn.Sequential(nn.Linear(d_behav, hidden), nn.ReLU())
        self.e_eng = nn.Sequential(nn.Linear(d_eng, hidden), nn.ReLU())
        self.attn = nn.Linear(hidden * 3, 3)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_perf, x_behav, x_eng):
        h1 = self.e_perf(x_perf)
        h2 = self.e_behav(x_behav)
        h3 = self.e_eng(x_eng)
        cat = torch.cat([h1, h2, h3], dim=1)
        w = torch.softmax(self.attn(cat), dim=1)
        fused = w[:, 0:1] * h1 + w[:, 1:2] * h2 + w[:, 2:3] * h3
        out = self.fc(fused).squeeze(-1)
        return out, w

    def freeze_encoders(self, freeze: bool = True):
        for module in [self.e_perf, self.e_behav, self.e_eng]:
            for p in module.parameters():
                p.requires_grad = not freeze


class StudentRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
