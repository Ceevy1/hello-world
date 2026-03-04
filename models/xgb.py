from __future__ import annotations

import numpy as np
from xgboost import XGBRegressor


class XGBModel:
    def __init__(self, max_depth: int = 6, n_estimators: int = 300, learning_rate: float = 0.05, use_gpu: bool = False):
        kwargs = {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        if use_gpu:
            kwargs.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
        self.model = XGBRegressor(**kwargs)

    def fit(self, x_tab: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x_tab, y)

    def predict(self, x_tab: np.ndarray) -> np.ndarray:
        return self.model.predict(x_tab)
