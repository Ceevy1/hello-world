"""Unified optimization loss for MT-HAFNet."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


@dataclass
class UnifiedLossConfig:
    lambda_transfer: float = 0.1
    lambda_diversity: float = 0.1
    lambda_stability: float = 0.1


class UnifiedLoss:
    """Composite loss: regression + transfer + diversity + stability."""

    def __init__(self, config: UnifiedLossConfig) -> None:
        self.config = config

    @staticmethod
    def regression_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_pred.view(-1), y_true.view(-1))

    @staticmethod
    def transfer_loss(hidden_by_module: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean pairwise squared distance of module mean embeddings."""
        if len(hidden_by_module) < 2:
            return torch.tensor(0.0)
        mus = [h.mean(dim=0) for h in hidden_by_module.values()]
        loss = torch.tensor(0.0, device=mus[0].device)
        count = 0
        for i in range(len(mus)):
            for j in range(i + 1, len(mus)):
                loss = loss + torch.sum((mus[i] - mus[j]) ** 2)
                count += 1
        return loss / max(count, 1)

    @staticmethod
    def diversity_loss(base_preds: torch.Tensor) -> torch.Tensor:
        """Negative variance across sub-model predictions [N, K]."""
        if base_preds.numel() == 0:
            return torch.tensor(0.0)
        return -torch.var(base_preds, dim=1).mean()

    @staticmethod
    def stability_loss(weights: torch.Tensor) -> torch.Tensor:
        """Distance to batch mean weights [N, K]."""
        if weights.numel() == 0:
            return torch.tensor(0.0)
        w_mean = weights.mean(dim=0, keepdim=True)
        return ((weights - w_mean) ** 2).sum(dim=1).mean()

    def total_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        hidden_by_module: Dict[str, torch.Tensor],
        base_preds: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        l_reg = self.regression_loss(y_pred, y_true)
        l_transfer = self.transfer_loss(hidden_by_module)
        l_div = self.diversity_loss(base_preds)
        l_stab = self.stability_loss(weights)
        return (
            l_reg
            + self.config.lambda_transfer * l_transfer
            + self.config.lambda_diversity * l_div
            + self.config.lambda_stability * l_stab
        )


def lambda_grid_search_space(
    lambda1: Iterable[float], lambda2: Iterable[float], lambda3: Iterable[float]
) -> List[UnifiedLossConfig]:
    return [
        UnifiedLossConfig(a, b, c)
        for a, b, c in product(lambda1, lambda2, lambda3)
    ]
