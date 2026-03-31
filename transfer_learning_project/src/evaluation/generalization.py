"""Generalization metrics and statistical testing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import entropy, ttest_ind
from scipy.stats import wasserstein_distance as scipy_wasserstein
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class MetricSummary:
    rmse: float
    mae: float
    r2: float


def regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> MetricSummary:
    return MetricSummary(
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


def generalization_drop(in_domain_rmse: float, cross_domain_rmse: float) -> float:
    return float((cross_domain_rmse - in_domain_rmse) / max(in_domain_rmse, 1e-8))


def feature_distribution_shift(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    features: list[str],
    bins: int = 30,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for feature in features:
        src = source_df[feature].dropna().to_numpy(dtype=float)
        tgt = target_df[feature].dropna().to_numpy(dtype=float)
        if len(src) == 0 or len(tgt) == 0:
            continue

        lo = float(min(src.min(), tgt.min()))
        hi = float(max(src.max(), tgt.max()))
        if np.isclose(lo, hi):
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, bins + 1)

        src_hist, _ = np.histogram(src, bins=edges, density=True)
        tgt_hist, _ = np.histogram(tgt, bins=edges, density=True)
        src_prob = src_hist + 1e-8
        tgt_prob = tgt_hist + 1e-8
        src_prob = src_prob / src_prob.sum()
        tgt_prob = tgt_prob / tgt_prob.sum()

        rows.append(
            {
                "feature": feature,
                "wasserstein": float(scipy_wasserstein(src, tgt)),
                "kl_divergence": float(entropy(src_prob, tgt_prob)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    draws = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        draws.append(np.mean(sample))
    lower = np.percentile(draws, (1 - ci) / 2 * 100)
    upper = np.percentile(draws, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lower), float(upper)


def performance_significance(in_domain_errors: np.ndarray, cross_domain_errors: np.ndarray) -> dict[str, float]:
    t_stat, p_val = ttest_ind(in_domain_errors, cross_domain_errors, equal_var=False)
    return {"t_stat": float(t_stat), "p_value": float(p_val)}
