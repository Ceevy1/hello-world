from __future__ import annotations

import numpy as np



def compute_stability(pred_dict: dict[object, np.ndarray]) -> float:
    if not pred_dict or len(pred_dict) < 2:
        return 0.0

    weeks = sorted(pred_dict.keys(), key=lambda x: (x == "full", float("inf") if x == "full" else int(x)))
    stability = []

    for i in range(len(weeks) - 1):
        diff = np.abs(np.asarray(pred_dict[weeks[i + 1]]) - np.asarray(pred_dict[weeks[i]]))
        stability.append(float(np.mean(diff)))

    return float(np.mean(stability)) if stability else 0.0
