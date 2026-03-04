from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def significance_tests(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    err_a = np.abs(y_true - pred_a)
    err_b = np.abs(y_true - pred_b)
    t_stat, t_p = ttest_rel(err_a, err_b, nan_policy="omit")
    try:
        w_stat, w_p = wilcoxon(err_a, err_b)
    except ValueError:
        w_stat, w_p = 0.0, 1.0
    return {
        "t_stat": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "cohen_d": cohen_d_paired(err_a, err_b),
    }
