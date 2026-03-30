"""Phase 4: baseline comparisons."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def run_phase4(config: dict, logger) -> None:
    df = pd.read_pickle("data/processed/target_features.pkl")
    X = df.drop(columns=["target"]).select_dtypes("number")
    y = df["target"]

    rows = []
    for name, model in [("BL-1", LinearRegression()), ("BL-2", RandomForestRegressor(n_estimators=100, random_state=42))]:
        model.fit(X, y)
        pred = model.predict(X)
        rows.append(
            {
                "model": name,
                "mae": mean_absolute_error(y, pred),
                "rmse": mean_squared_error(y, pred) ** 0.5,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv("results/tables/baseline_results.csv", index=False)
    logger.info("Phase4 complete: baseline_results.csv generated")
