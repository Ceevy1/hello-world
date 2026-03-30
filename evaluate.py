from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_logits: np.ndarray) -> Dict[str, float]:
    probs = _softmax(y_logits)
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": float(accuracy_score(y_true, preds)),
        "F1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "AUC": float(auc),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)
