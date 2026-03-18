from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred_prob = np.asarray(y_pred_prob).astype(float)
    y_pred = (y_pred_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(auc),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
