from __future__ import annotations

import torch
import torch.nn.functional as F


def weight_stability_regularization(weights: torch.Tensor) -> torch.Tensor:
    target = torch.full_like(weights, 1 / weights.size(-1))
    return torch.mean((weights - target) ** 2)


def early_consistency_regularization(pred_early: torch.Tensor, pred_full: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_early, pred_full)


def contrastive_loss(repr_a: torch.Tensor, repr_b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    a = F.normalize(repr_a, dim=-1)
    b = F.normalize(repr_b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return F.cross_entropy(logits, labels)


def loss_fn(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights: torch.Tensor,
    pred_early: torch.Tensor,
    pred_full: torch.Tensor,
    repr_a: torch.Tensor | None = None,
    repr_b: torch.Tensor | None = None,
    lambda1: float = 0.5,
    lambda2: float = 0.1,
    lambda3: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_pred = F.binary_cross_entropy(y_pred, y_true)
    l_early = early_consistency_regularization(pred_early, pred_full)
    l_weight = weight_stability_regularization(weights)
    l_cl = contrastive_loss(repr_a, repr_b) if repr_a is not None and repr_b is not None else torch.tensor(0.0, device=y_pred.device)
    total = l_pred + lambda1 * l_early + lambda2 * l_weight + lambda3 * l_cl
    return total, {"pred": float(l_pred.detach()), "early": float(l_early.detach()), "weight": float(l_weight.detach()), "cl": float(l_cl.detach())}
