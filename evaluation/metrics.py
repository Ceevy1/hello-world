from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    std = float(np.std(y_pred - y_true))
    ci_low, ci_high = stats.t.interval(
        0.95, len(y_true) - 1, loc=np.mean(y_pred - y_true), scale=stats.sem(y_pred - y_true)
    )
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Std": std, "CI95_low": float(ci_low), "CI95_high": float(ci_high)}


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(auc),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
