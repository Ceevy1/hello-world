"""Generic sklearn-style trainer helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import RegressorMixin


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


def fit_regressor(
    model: RegressorMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> TrainingHistory:
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    train_loss = float(np.mean((train_pred - y_train) ** 2))

    val_losses: list[float] = []
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_losses.append(float(np.mean((val_pred - y_val) ** 2)))
    return TrainingHistory(train_loss=[train_loss], val_loss=val_losses)
