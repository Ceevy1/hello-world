"""Phase 6: cross-domain generalization evaluation for DynamicFusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.data.data_loader import DataLoader
from src.data.feature_aligner import FeatureAligner
from src.evaluation.generalization import (
    bootstrap_ci,
    feature_distribution_shift,
    generalization_drop,
    performance_significance,
    regression_summary,
)
from src.models.dynamic_fusion import DynamicFusionRegressor


def _split_semantic(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_perf = df[["performance"]].to_numpy(dtype=float)
    x_behav = df[["behavior"]].to_numpy(dtype=float)
    x_eng = df[["engagement", "background"]].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=float)
    return x_perf, x_behav, x_eng, y


def _counterfactual_effect(model: DynamicFusionRegressor, x_perf: np.ndarray, x_behav: np.ndarray, x_eng: np.ndarray) -> float:
    base = model.predict(x_perf, x_behav, x_eng)
    x_eng_new = x_eng.copy()
    x_eng_new[:, 0] = x_eng_new[:, 0] + 0.1
    moved = model.predict(x_perf, x_behav, x_eng_new)
    return float(np.mean(moved - base))


def run_phase6(config: dict, logger) -> None:
    loader = DataLoader(config.get("custom_dataset", {}).get("columns", {}))
    aligner = FeatureAligner()

    source_raw = loader.load_oulad_features(config["data"]["source"]["path"])
    target_raw = loader.load_custom_dataset(config["data"]["target"]["path"], anonymize=True)

    source_aligned = aligner.align(source_raw, domain="oulad").dropna(subset=["target"]).reset_index(drop=True)
    target_aligned = aligner.align(target_raw, domain="custom").dropna(subset=["target"]).reset_index(drop=True)

    Path("results/tables").mkdir(parents=True, exist_ok=True)

    shift_df = feature_distribution_shift(source_aligned, target_aligned, ["performance", "engagement", "behavior", "background"])
    shift_df.to_csv("results/tables/distribution_shift.csv", index=False)

    s_perf, s_behav, s_eng, s_y = _split_semantic(source_aligned)
    t_perf, t_behav, t_eng, t_y = _split_semantic(target_aligned)

    s_train_idx, s_test_idx = train_test_split(np.arange(len(s_y)), test_size=0.2, random_state=42)
    t_train_idx, t_test_idx = train_test_split(np.arange(len(t_y)), test_size=0.2, random_state=42)

    model = DynamicFusionRegressor(input_dims=(1, 1, 2), lr=1e-3, batch_size=32, epochs=80)
    model.fit(s_perf[s_train_idx], s_behav[s_train_idx], s_eng[s_train_idx], s_y[s_train_idx])

    pred_in, attn_in = model.predict_with_attention(s_perf[s_test_idx], s_behav[s_test_idx], s_eng[s_test_idx])
    pred_cross_before, attn_cross_before = model.predict_with_attention(t_perf[t_test_idx], t_behav[t_test_idx], t_eng[t_test_idx])

    model.fit(t_perf[t_train_idx], t_behav[t_train_idx], t_eng[t_train_idx], t_y[t_train_idx])
    pred_cross_after, attn_cross_after = model.predict_with_attention(t_perf[t_test_idx], t_behav[t_test_idx], t_eng[t_test_idx])

    m_in = regression_summary(s_y[s_test_idx], pred_in)
    m_cross_before = regression_summary(t_y[t_test_idx], pred_cross_before)
    m_cross_after = regression_summary(t_y[t_test_idx], pred_cross_after)

    drop_before = generalization_drop(m_in.rmse, m_cross_before.rmse)
    drop_after = generalization_drop(m_in.rmse, m_cross_after.rmse)

    in_abs_err = np.abs(s_y[s_test_idx] - pred_in)
    cross_abs_err = np.abs(t_y[t_test_idx] - pred_cross_before)
    sig = performance_significance(in_abs_err, cross_abs_err)
    err_mean, err_ci_low, err_ci_high = bootstrap_ci(cross_abs_err)

    summary = pd.DataFrame(
        [
            {"setting": "in_domain_source", "rmse": m_in.rmse, "mae": m_in.mae, "r2": m_in.r2, "generalization_drop": 0.0},
            {
                "setting": "cross_domain_before_finetune",
                "rmse": m_cross_before.rmse,
                "mae": m_cross_before.mae,
                "r2": m_cross_before.r2,
                "generalization_drop": drop_before,
            },
            {
                "setting": "cross_domain_after_finetune",
                "rmse": m_cross_after.rmse,
                "mae": m_cross_after.mae,
                "r2": m_cross_after.r2,
                "generalization_drop": drop_after,
            },
        ]
    )
    summary.to_csv("results/tables/generalization_summary.csv", index=False)

    attention = pd.DataFrame(
        [
            {"domain": "source_test", "w_perf": attn_in[:, 0].mean(), "w_behav": attn_in[:, 1].mean(), "w_eng": attn_in[:, 2].mean()},
            {
                "domain": "target_test_before_ft",
                "w_perf": attn_cross_before[:, 0].mean(),
                "w_behav": attn_cross_before[:, 1].mean(),
                "w_eng": attn_cross_before[:, 2].mean(),
            },
            {
                "domain": "target_test_after_ft",
                "w_perf": attn_cross_after[:, 0].mean(),
                "w_behav": attn_cross_after[:, 1].mean(),
                "w_eng": attn_cross_after[:, 2].mean(),
            },
        ]
    )
    attention.to_csv("results/tables/attention_shift.csv", index=False)

    # SHAP-like consistency proxy with permutation importance on each domain split.
    class _SkCompat:
        def __init__(self, m: DynamicFusionRegressor):
            self.m = m

        def fit(self, X: np.ndarray, y: np.ndarray):
            return self

        def predict(self, X: np.ndarray):
            return self.m.predict(X[:, [0]], X[:, [1]], X[:, 2:4])

    sk_model = _SkCompat(model)
    feat_names = ["performance", "behavior", "engagement", "background"]
    X_s = np.hstack([s_perf[s_test_idx], s_behav[s_test_idx], s_eng[s_test_idx]])
    X_t = np.hstack([t_perf[t_test_idx], t_behav[t_test_idx], t_eng[t_test_idx]])
    imp_s = permutation_importance(sk_model, X_s, s_y[s_test_idx], n_repeats=10, random_state=42)
    imp_t = permutation_importance(sk_model, X_t, t_y[t_test_idx], n_repeats=10, random_state=42)
    shap_df = pd.DataFrame(
        {
            "feature": feat_names,
            "importance_source": imp_s.importances_mean,
            "importance_target": imp_t.importances_mean,
            "delta": imp_t.importances_mean - imp_s.importances_mean,
        }
    )
    shap_df.to_csv("results/tables/shap_consistency_proxy.csv", index=False)

    counterfactual = pd.DataFrame(
        [
            {"intervention": "engagement +0.1", "source_delta": _counterfactual_effect(model, s_perf[s_test_idx], s_behav[s_test_idx], s_eng[s_test_idx]), "target_delta": _counterfactual_effect(model, t_perf[t_test_idx], t_behav[t_test_idx], t_eng[t_test_idx])}
        ]
    )
    counterfactual.to_csv("results/tables/counterfactual_effect.csv", index=False)

    stat_row = pd.DataFrame(
        [
            {
                "mean_abs_error": err_mean,
                "ci_low": err_ci_low,
                "ci_high": err_ci_high,
                "t_stat": sig["t_stat"],
                "p_value": sig["p_value"],
            }
        ]
    )
    stat_row.to_csv("results/tables/significance_ci.csv", index=False)

    logger.info("Phase6 complete: generalization summaries exported to results/tables")
