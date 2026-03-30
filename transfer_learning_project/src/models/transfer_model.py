"""Main transfer learning model wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from src.training.early_stopping import EarlyStopping


class TransferLearningModel:
    def __init__(
        self,
        pretrained_model,
        transfer_strategy: str,
        domain_adapter=None,
        finetune_strategy: str = "FT-1",
        task_type: str = "regression",
    ):
        self.base_model = pretrained_model
        self.transfer_strategy = transfer_strategy
        self.domain_adapter = domain_adapter
        self.finetune_strategy = finetune_strategy
        self.task_type = task_type
        self.task_head = None
        self.is_fitted = False
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def fit(
        self,
        X_target_train: np.ndarray,
        y_target_train: np.ndarray,
        X_target_val: np.ndarray | None = None,
        y_target_val: np.ndarray | None = None,
        X_source: np.ndarray | None = None,
    ) -> "TransferLearningModel":
        X_train = X_target_train
        if self.domain_adapter is not None and X_source is not None:
            X_train = self.domain_adapter.align(X_source, X_target_train)

        if self.task_type == "classification":
            self.task_head = RandomForestClassifier(n_estimators=300, random_state=42)
        else:
            self.task_head = RandomForestRegressor(n_estimators=300, random_state=42)

        self.task_head.fit(X_train, y_target_train)

        train_pred = self.task_head.predict(X_train)
        train_loss = float(np.mean((train_pred - y_target_train) ** 2))
        self.history["train_loss"].append(train_loss)

        stopper = EarlyStopping(patience=10, mode="min")
        if X_target_val is not None and y_target_val is not None:
            X_val = X_target_val
            if self.domain_adapter is not None and X_source is not None:
                X_val = self.domain_adapter.align(X_source, X_target_val)
            val_pred = self.task_head.predict(X_val)
            val_loss = float(np.mean((val_pred - y_target_val) ** 2))
            self.history["val_loss"].append(val_loss)
            stopper.step(val_loss)

        self.is_fitted = True
        return self

    def predict(self, X_target_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.task_head.predict(X_target_test)

    def predict_proba(self, X_target_test: np.ndarray) -> np.ndarray:
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        if hasattr(self.task_head, "predict_proba"):
            return self.task_head.predict_proba(X_target_test)
        # fallback
        scores = LogisticRegression().fit(X_target_test, self.predict(X_target_test)).predict_proba(X_target_test)
        return scores

    def get_feature_embedding(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.base_model, "get_embedding"):
            return self.base_model.get_embedding(X)
        return X
