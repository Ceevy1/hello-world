from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from loss.unified_loss import UnifiedLossConfig
from preprocess.data_builder import split_lomo
from train.train_full import train_full_pipeline
from run_full import synthetic_data


def _distance_matrix(x: np.ndarray, modules: np.ndarray) -> pd.DataFrame:
    uniq = np.unique(modules)
    means = {m: x[modules == m].mean(axis=0) for m in uniq}
    rows = []
    for i in uniq:
        for j in uniq:
            dist = float(np.linalg.norm(means[i] - means[j]))
            rows.append({"Module_i": i, "Module_j": j, "Distance": dist})
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    x_seq, x_tab, y, modules, *_ = synthetic_data(seed=7)

    rows = []
    deltas = []
    for m in np.unique(modules):
        split = split_lomo(modules, m)
        out_transfer = train_full_pipeline(
            x_seq[split.train_idx], x_tab[split.train_idx], y[split.train_idx],
            x_seq[split.test_idx], x_tab[split.test_idx], y[split.test_idx],
            loss_cfg=UnifiedLossConfig(0.1, 0.1, 0.1), modules_train=modules[split.train_idx]
        )
        out_no_transfer = train_full_pipeline(
            x_seq[split.train_idx], x_tab[split.train_idx], y[split.train_idx],
            x_seq[split.test_idx], x_tab[split.test_idx], y[split.test_idx],
            loss_cfg=UnifiedLossConfig(0.0, 0.1, 0.1), modules_train=modules[split.train_idx]
        )
        mt = out_transfer.metrics["HAFM"]
        rows.append({"Module": m, "RMSE": mt["RMSE"], "MAE": mt["MAE"], "R2": mt["R2"]})
        deltas.append({"Module": m, "delta_rmse": mt["RMSE"] - out_no_transfer.metrics["HAFM"]["RMSE"]})

    lomo = pd.DataFrame(rows)
    lomo.to_csv(out_dir / "lomo_results.csv", index=False)
    summary = pd.DataFrame([{
        "mean_rmse": lomo["RMSE"].mean(),
        "std_rmse": lomo["RMSE"].std(),
        "mean_mae": lomo["MAE"].mean(),
        "mean_r2": lomo["R2"].mean(),
    }])
    summary.to_csv(out_dir / "lomo_summary.csv", index=False)
    pd.DataFrame(deltas).to_csv(out_dir / "generalization_delta.csv", index=False)

    before = _distance_matrix(x_tab, modules)
    after = _distance_matrix((x_tab - x_tab.mean(axis=0)) / (x_tab.std(axis=0) + 1e-8), modules)
    both = before.merge(after, on=["Module_i", "Module_j"], suffixes=("_before", "_after"))
    both.to_csv(out_dir / "transfer_distance.csv", index=False)


if __name__ == "__main__":
    main()
