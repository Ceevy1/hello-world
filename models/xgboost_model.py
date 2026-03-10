from __future__ import annotations

import numpy as np


class XGBoostClassifierModel:
    def __init__(self, **kwargs) -> None:
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        params.update(kwargs)
        try:
            from xgboost import XGBClassifier

            self.model = XGBClassifier(**params)
        except Exception:
            from sklearn.ensemble import RandomForestClassifier

            self.model = RandomForestClassifier(n_estimators=300, random_state=42)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict(x)
