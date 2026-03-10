from __future__ import annotations

import numpy as np


class CatBoostClassifierModel:
    def __init__(self, **kwargs) -> None:
        params = {
            "depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "loss_function": "Logloss",
            "verbose": False,
            "random_state": 42,
        }
        params.update(kwargs)
        try:
            from catboost import CatBoostClassifier

            self.model = CatBoostClassifier(**params)
        except Exception:
            from sklearn.ensemble import GradientBoostingClassifier

            self.model = GradientBoostingClassifier(random_state=42)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict(x)
