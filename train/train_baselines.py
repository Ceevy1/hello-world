from __future__ import annotations

from typing import Dict

import numpy as np

from evaluation.metrics import regression_metrics
from models.baselines import BaselineSuite, fit_predict_baselines


def run_baseline_benchmark(
    x_tab_train: np.ndarray,
    y_train: np.ndarray,
    x_tab_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    preds = fit_predict_baselines(x_tab_train, y_train, x_tab_test, suite=BaselineSuite())
    return {name: regression_metrics(y_test, pred) for name, pred in preds.items()}
