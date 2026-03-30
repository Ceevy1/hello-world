from __future__ import annotations

import torch
import torch.nn as nn


class DynamicFusionModel(nn.Module):
    def __init__(self, static_dim: int, hidden_dim: int = 64, task: str = "regression", num_classes: int = 4):
        super().__init__()
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'.")
        self.task = task

        self.exercise_encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.lab_encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.static_encoder = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.ReLU())

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(hidden_dim, 1 if task == "regression" else num_classes)

    def forward(self, exercise: torch.Tensor, lab: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        ex = exercise.unsqueeze(-1)
        lb = lab.unsqueeze(-1)
        _, (h_ex, _) = self.exercise_encoder(ex)
        _, (h_lb, _) = self.lab_encoder(lb)
        h_static = self.static_encoder(static)

        x = torch.cat([h_ex[-1], h_lb[-1], h_static], dim=-1)
        z = self.fusion(x)
        out = self.head(z)
        if self.task == "regression":
            return out.squeeze(-1)
        return out
