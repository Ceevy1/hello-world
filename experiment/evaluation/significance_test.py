from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import ttest_rel


def paired_t_test(results_model: np.ndarray, results_baseline: np.ndarray) -> Dict[str, float | bool]:
    a = np.asarray(results_model, dtype=float)
    b = np.asarray(results_baseline, dtype=float)
    stat, p_value = ttest_rel(a, b, nan_policy="omit")
    return {
        "t_stat": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
