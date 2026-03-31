#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from paper_generalization.common import TARGET_COL, infer_features, kfold, load_dataset


def rmse_func(y, pred):
    return np.sqrt(mean_squared_error(y, pred))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--out", default="outputs/baseline_results.csv")
    args = p.parse_args()

    df = load_dataset(args.data)
    feats = infer_features(df)
    X = df[feats].values
    y = df[TARGET_COL].values

    models = {
        "LinearRegression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42)),
            ]
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    except Exception:
        pass

    scoring = {
        "rmse": make_scorer(rmse_func, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": make_scorer(r2_score),
    }

    rows = []
    fold_rows = []
    for name, model in models.items():
        cv = kfold(len(y), args.cv)
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=None)
        rmse_folds = -res["test_rmse"]
        for i, v in enumerate(rmse_folds, start=1):
            fold_rows.append({"model": name, "fold": i, "RMSE": float(v)})
        rows.append(
            {
                "model": name,
                "RMSE_mean": float((-res["test_rmse"]).mean()),
                "RMSE_std": float((-res["test_rmse"]).std()),
                "MAE_mean": float((-res["test_mae"]).mean()),
                "MAE_std": float((-res["test_mae"]).std()),
                "R2_mean": float(res["test_r2"].mean()),
                "R2_std": float(res["test_r2"].std()),
            }
        )

    out = pd.DataFrame(rows).sort_values("RMSE_mean")
    out.to_csv(args.out, index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_path = args.out.replace(".csv", "_folds.csv")
    fold_df.to_csv(fold_path, index=False)
    rf_path = args.out.replace(".csv", "_rf_folds.csv")
    fold_df[fold_df["model"] == "RandomForest"][["fold", "RMSE"]].to_csv(rf_path, index=False)
    print(out)
    print(f"saved to {args.out}; fold metrics: {fold_path}; RF folds: {rf_path}")


if __name__ == "__main__":
    main()
