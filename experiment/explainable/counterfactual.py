from __future__ import annotations

import numpy as np



def generate_counterfactual(model, x: np.ndarray, feature_names: list[str], step: float = 0.2, threshold: float = 0.5) -> list[tuple[str, float, float]]:
    x = np.asarray(x, dtype=float)
    base_pred = float(model.predict(x.reshape(1, -1))[0])
    suggestions: list[tuple[str, float, float]] = []

    for i, feature in enumerate(feature_names):
        x_new = x.copy()
        delta = step * max(abs(x[i]), 1.0)
        x_new[i] = x[i] + delta
        pred_new = float(model.predict(x_new.reshape(1, -1))[0])

        if base_pred < threshold and pred_new > threshold:
            suggestions.append((feature, float(x[i]), float(x_new[i])))

    return suggestions
