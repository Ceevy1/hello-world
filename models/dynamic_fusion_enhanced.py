from __future__ import annotations

import torch
from torch import nn


def compute_weight_regularization(weights: torch.Tensor) -> torch.Tensor:
    n = weights.shape[-1]
    target = torch.full_like(weights, 1.0 / n)
    return ((weights - target) ** 2).sum(dim=-1).mean()


class DynamicFusionEnhanced(nn.Module):
    """Temporal-aware dynamic fusion for deep + tree predictions."""

    def __init__(self, deep_dim: int = 128, week_emb_dim: int = 8, n_tree_models: int = 2, max_week: int = 64) -> None:
        super().__init__()
        self.n_outputs = n_tree_models + 1
        self.week_emb = nn.Embedding(max_week + 1, week_emb_dim)
        self.weight_layer = nn.Linear(deep_dim + n_tree_models + week_emb_dim, self.n_outputs)

    def forward(
        self,
        deep_representation: torch.Tensor,
        tree_predictions: torch.Tensor,
        week_index: torch.Tensor,
        deep_prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        week_feature = self.week_emb(week_index.long())
        fusion_input = torch.cat([deep_representation, tree_predictions, week_feature], dim=-1)
        weights = torch.softmax(self.weight_layer(fusion_input), dim=-1)

        all_preds = torch.cat([deep_prediction.unsqueeze(-1), tree_predictions], dim=-1)
        y = torch.sum(weights * all_preds, dim=-1)
        return y, weights
