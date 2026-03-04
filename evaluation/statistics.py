from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def significance_tests(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    err_a = np.abs(y_true - pred_a)
    err_b = np.abs(y_true - pred_b)
    t_p = float(ttest_rel(err_a, err_b, nan_policy="omit").pvalue)
    try:
        w_p = float(wilcoxon(err_a, err_b).pvalue)
    except ValueError:
        w_p = 1.0
    return {"paired_t_p": t_p, "wilcoxon_p": w_p}
