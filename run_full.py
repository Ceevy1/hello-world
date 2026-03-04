from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.statistics import significance_tests
from loss.unified_loss import UnifiedLossConfig
from train.logger import ExperimentLogger
from train.train_baselines import run_baseline_benchmark
from train.train_full import train_full_pipeline


def synthetic_data(n: int = 300, t: int = 20, d: int = 4, f: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)
    x_seq = rng.normal(size=(n, t, d)).astype(np.float32)
    x_tab = rng.normal(size=(n, f)).astype(np.float32)
    clicks = np.exp(x_tab[:, 0]) * 10
    attempt = (x_tab[:, 1] > 0).astype(int)
    signal = x_tab[:, 0] * 4 + x_seq[:, :, 0].mean(axis=1) * 5 + attempt * 2
    y = (55 + signal + rng.normal(scale=5, size=n)).clip(0, 100).astype(np.float32)
    modules = rng.choice(["AAA", "BBB", "CCC", "DDD"], size=n)
    return x_seq, x_tab, y, modules, clicks, attempt


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "loss_curves").mkdir(exist_ok=True)

    rows = []
    for seed in [0, 1, 2, 3, 4]:
        x_seq, x_tab, y, modules, clicks, attempt = synthetic_data(seed=seed)
        split = int(0.8 * len(y))

        baseline = run_baseline_benchmark(x_tab[:split], y[:split], x_tab[split:], y[split:])
        logger = ExperimentLogger(output_dir="outputs")
        out = train_full_pipeline(
            x_seq[:split], x_tab[:split], y[:split], x_seq[split:], x_tab[split:], y[split:],
            loss_cfg=UnifiedLossConfig(0.1, 0.1, 0.1), modules_train=modules[:split], logger=logger
        )
        logger.save(f"loss_history_seed{seed}.csv")
        logger.export_loss_series({
            "loss_reg": "loss_reg_curve.csv",
            "loss_transfer": "loss_transfer_curve.csv",
            "loss_diversity": "loss_diversity_curve.csv",
            "loss_stability": "loss_stability_curve.csv",
        })

        for model, m in out.metrics.items():
            rows.append({"seed": seed, "model": model, **m})
        for model, m in baseline.items():
            rows.append({"seed": seed, "model": model, **m})

        w = out.weights
        w_df = pd.DataFrame({"w_lstm": w[:, 0], "w_xgb": w[:, 1], "w_cat": w[:, 2], "clicks": clicks[split:], "attempt": attempt[split:]})
        grp = []
        high = w_df[w_df["clicks"] >= w_df["clicks"].median()]
        low = w_df[w_df["clicks"] < w_df["clicks"].median()]
        grp.append({"Group": "high_click", "w_lstm": high.w_lstm.mean(), "w_xgb": high.w_xgb.mean(), "w_cat": high.w_cat.mean()})
        grp.append({"Group": "low_click", "w_lstm": low.w_lstm.mean(), "w_xgb": low.w_xgb.mean(), "w_cat": low.w_cat.mean()})
        grp.append({"Group": "first_attempt", "w_lstm": w_df[w_df.attempt==1].w_lstm.mean(), "w_xgb": w_df[w_df.attempt==1].w_xgb.mean(), "w_cat": w_df[w_df.attempt==1].w_cat.mean()})
        grp.append({"Group": "resit", "w_lstm": w_df[w_df.attempt==0].w_lstm.mean(), "w_xgb": w_df[w_df.attempt==0].w_xgb.mean(), "w_cat": w_df[w_df.attempt==0].w_cat.mean()})
        pd.DataFrame(grp).to_csv(out_dir / "weight_group_comparison.csv", index=False)

        err = np.abs(y[split:] - out.predictions["HAFM"])
        corr_df = pd.DataFrame([
            {"model_weight": "w_lstm", "corr_with_error": np.corrcoef(w[:, 0], err)[0, 1]},
            {"model_weight": "w_xgb", "corr_with_error": np.corrcoef(w[:, 1], err)[0, 1]},
            {"model_weight": "w_cat", "corr_with_error": np.corrcoef(w[:, 2], err)[0, 1]},
        ])
        corr_df.to_csv(out_dir / "weight_error_correlation.csv", index=False)

        weight_stats = pd.DataFrame([
            {"Model": "LSTM", "Mean Weight": w[:, 0].mean(), "Std Weight": w[:, 0].std()},
            {"Model": "XGB", "Mean Weight": w[:, 1].mean(), "Std Weight": w[:, 1].std()},
            {"Model": "CAT", "Mean Weight": w[:, 2].mean(), "Std Weight": w[:, 2].std()},
        ])
        weight_stats.to_csv(out_dir / "weight_distribution.csv", index=False)

        div = np.corrcoef(out.base_predictions.T)
        pd.DataFrame(div, index=["LSTM", "XGB", "CAT"], columns=["LSTM", "XGB", "CAT"]).to_csv(out_dir / "diversity_matrix.csv")

        sig_rows = []
        for m in ["LSTM", "XGBoost", "CatBoost"]:
            s = significance_tests(y[split:], out.predictions["HAFM"], out.predictions[m])
            sig_rows.append({"Comparison": f"HAFM vs {m}", **s})
        pd.DataFrame(sig_rows).to_csv(out_dir / "significance_tests.csv", index=False)

    df = pd.DataFrame(rows)
    main = df.groupby("model", as_index=False)[["RMSE", "MAE", "R2"]].mean()
    main.to_csv(out_dir / "main_results.csv", index=False)


if __name__ == "__main__":
    main()
