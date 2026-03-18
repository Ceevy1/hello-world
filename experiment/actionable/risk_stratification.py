from __future__ import annotations

import numpy as np
import pandas as pd



def risk_level(prob: float) -> str:
    if prob < 0.4:
        return "High Risk"
    if prob < 0.7:
        return "Medium Risk"
    return "Low Risk"



def stratify_predictions(probabilities: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    labels = np.array([risk_level(prob) for prob in probabilities])
    y_pred = (probabilities >= 0.5).astype(int)

    rows = []
    for level in ["High Risk", "Medium Risk", "Low Risk"]:
        mask = labels == level
        if not np.any(mask):
            rows.append({"Risk Level": level, "Count": 0, "Accuracy": np.nan})
            continue
        rows.append(
            {
                "Risk Level": level,
                "Count": int(mask.sum()),
                "Accuracy": float(np.mean(y_pred[mask] == y_true[mask])),
            }
        )
    return pd.DataFrame(rows)
