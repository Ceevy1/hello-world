"""Hierarchical Adaptive Fusion Module (HAFM)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


class HAFM(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x_tab: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x_tab), dim=-1)


@dataclass
class FusionOutput:
    y_pred: torch.Tensor
    weights: torch.Tensor


def fuse_predictions(
    x_tab: torch.Tensor,
    base_preds: torch.Tensor,
    model: HAFM,
) -> FusionOutput:
    """Fuse three base predictions with dynamic softmax weights.

    Args:
        x_tab: [N, F]
        base_preds: [N, 3] aligned to [lstm, xgb, cat]
    """
    weights = model(x_tab)
    y_pred = (weights * base_preds).sum(dim=1)
    return FusionOutput(y_pred=y_pred, weights=weights)
