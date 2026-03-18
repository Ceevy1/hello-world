from __future__ import annotations

import numpy as np

from experiment.explainable.counterfactual import generate_counterfactual



def intervention_success_rate(model, data: np.ndarray, feature_names: list[str]) -> float:
    success = 0
    total = 0

    for x in np.asarray(data):
        cf = generate_counterfactual(model, x, feature_names)
        if cf:
            success += 1
        total += 1

    return float(success / total) if total else 0.0
