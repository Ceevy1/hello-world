from __future__ import annotations

import torch
import torch.nn.functional as F

from models.dynamic_fusion_enhanced import compute_weight_regularization


def compute_early_consistency_loss(y_4week: torch.Tensor, y_full: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_4week - y_full))


def dynamic_fusion_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights: torch.Tensor,
    y_4week: torch.Tensor,
    y_full: torch.Tensor,
    lambda1: float = 0.1,
    lambda2: float = 0.2,
) -> torch.Tensor:
    l_pred = F.binary_cross_entropy(y_pred, y_true.float())
    l_weight = compute_weight_regularization(weights)
    l_early = compute_early_consistency_loss(y_4week, y_full)
    return l_pred + lambda1 * l_weight + lambda2 * l_early
