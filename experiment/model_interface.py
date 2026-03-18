from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from experiment.explainable.shap_analysis import summarize_shap


class Model(ABC):
    """Unified PIA interface."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Model":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def explain(self, x: np.ndarray, feature_names: list[str], save_dir: str | Path | None = None) -> dict[str, Any]:
        raise NotImplementedError


class SklearnPIAModel(Model):
    """Lightweight classifier adapter exposing the required PIA interface."""

    def __init__(self, estimator: Any | None = None) -> None:
        if estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            estimator = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
            )
        self.estimator = estimator
        self.model = estimator

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SklearnPIAModel":
        from sklearn.base import clone

        self.model = clone(self.estimator)
        self.model.fit(np.asarray(x), np.asarray(y).astype(int))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        score = self.model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-score))

    def explain(self, x: np.ndarray, feature_names: list[str], save_dir: str | Path | None = None) -> dict[str, Any]:
        x = np.asarray(x)
        return summarize_shap(self.model, x, feature_names=feature_names, save_dir=save_dir)
