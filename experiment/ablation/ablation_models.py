from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class FullFusionModel:
    def __init__(self) -> None:
        self.model = RandomForestClassifier(n_estimators=250, random_state=42)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


class StaticFusionModel:
    def __init__(self) -> None:
        self.m1 = LogisticRegression(max_iter=500)
        self.m2 = RandomForestClassifier(n_estimators=120, random_state=42)
        self.m3 = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.m1.fit(x, y)
        self.m2.fit(x, y)
        self.m3.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = self.m1.predict_proba(x)[:, 1] + self.m2.predict_proba(x)[:, 1] + self.m3.predict_proba(x)[:, 1]
        return p / 3.0


class NoEntropyModel(FullFusionModel):
    @staticmethod
    def remove_entropy(x: np.ndarray) -> np.ndarray:
        # feature_engineering.extract_features puts entropy at index -3 in current implementation
        if x.shape[1] < 3:
            return x
        return np.delete(x, -3, axis=1)


class LSTMOnlyModel:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=500)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]
