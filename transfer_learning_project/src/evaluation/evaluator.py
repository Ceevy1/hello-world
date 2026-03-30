"""Evaluation orchestration with small-sample-aware CV."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from src.evaluation.metrics import classification_metrics, regression_metrics


class Evaluator:
    def __init__(self, task_type: str = "regression") -> None:
        self.task_type = task_type

    def evaluate(self, model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        pred = model.predict(X)
        if self.task_type == "classification":
            return classification_metrics(y, pred)
        return regression_metrics(y, pred)

    def cross_validate_transfer(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv_strategy: str = "auto",
        n_splits: int = 5,
        seed: int = 42,
    ) -> pd.DataFrame:
        if cv_strategy == "auto":
            if len(X) < 100:
                cv_strategy = "loo"
            else:
                cv_strategy = "stratified_k"

        cv = LeaveOneOut() if cv_strategy == "loo" else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rows = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            model.fit(X[train_idx], y[train_idx])
            metrics = self.evaluate(model, X[test_idx], y[test_idx])
            rows.append({"fold": fold, **metrics})
        return pd.DataFrame(rows)
