from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor


class CatModel:
    def __init__(self) -> None:
        self.model = CatBoostRegressor(
            depth=8,
            learning_rate=0.05,
            n_estimators=400,
            loss_function="RMSE",
            verbose=False,
            random_state=42,
        )

    def fit(self, x_tab: np.ndarray, y: np.ndarray, cat_features: list[int] | None = None) -> None:
        self.model.fit(x_tab, y, cat_features=cat_features)

    def predict(self, x_tab: np.ndarray) -> np.ndarray:
        return self.model.predict(x_tab)
