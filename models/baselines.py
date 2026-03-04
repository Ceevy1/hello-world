"""Classic regression baselines for paper-scale comparison experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


@dataclass
class BaselineSuite:
    """At least 8 common baselines, all tabular-input regressors."""

    random_state: int = 42

    def build(self) -> Dict[str, object]:
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=self.random_state),
            "Lasso": Lasso(alpha=0.001, random_state=self.random_state, max_iter=5000),
            "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=self.random_state, max_iter=5000),
            "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1),
            "KNN": KNeighborsRegressor(n_neighbors=7),
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=self.random_state, n_jobs=-1),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=self.random_state, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(random_state=self.random_state),
            "AdaBoost": AdaBoostRegressor(random_state=self.random_state),
        }


def fit_predict_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    suite: BaselineSuite | None = None,
) -> Dict[str, np.ndarray]:
    suite = suite or BaselineSuite()
    preds: Dict[str, np.ndarray] = {}
    for name, model in suite.build().items():
        model.fit(x_train, y_train)
        preds[name] = model.predict(x_test)
    return preds
