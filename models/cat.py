from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor


class CatModel:
    def __init__(self, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu
        self.model = self._build_model(use_gpu)

    @staticmethod
    def _build_model(use_gpu: bool) -> CatBoostRegressor:
        kwargs = {
            "depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "loss_function": "RMSE",
            "verbose": False,
            "random_state": 42,
        }
        if use_gpu:
            kwargs["task_type"] = "GPU"
        return CatBoostRegressor(**kwargs)

    def fit(self, x_tab: np.ndarray, y: np.ndarray, cat_features: list[int] | None = None) -> None:
        if self.use_gpu:
            try:
                self.model.fit(x_tab, y, cat_features=cat_features)
                return
            except Exception:
                # Runtime fallback for environments where CatBoost GPU backend is unavailable.
                self.model = self._build_model(use_gpu=False)
                self.use_gpu = False
        self.model.fit(x_tab, y, cat_features=cat_features)

    def predict(self, x_tab: np.ndarray) -> np.ndarray:
        return self.model.predict(x_tab)
