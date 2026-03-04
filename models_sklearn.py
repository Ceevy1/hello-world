"""
Sklearn-based model implementations.
Used as fallbacks when PyTorch/XGBoost/CatBoost are unavailable.
These match the same interface as models.py.
"""

import logging
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple
import copy

logger = logging.getLogger(__name__)


class LSTMModelSklearn:
    """
    MLP-based surrogate for LSTM when PyTorch is unavailable.
    Flattens sequence input and uses MLP.
    """

    def __init__(self, config: Dict, input_dim: int, seed: int = 42):
        self.config = config
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=config.get("epochs", 100),
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate_init=config.get("learning_rate", 1e-3),
        )

    def fit(self, X_seq: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        X_flat = X_seq.reshape(len(X_seq), -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model.fit(X_scaled, y)
        logger.info("[LSTM-MLP] Training complete.")

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        X_flat = X_seq.reshape(len(X_seq), -1)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)

    def save(self, path: str):
        import joblib
        joblib.dump((self.model, self.scaler), path)

    def load(self, path: str):
        import joblib
        self.model, self.scaler = joblib.load(path)


class XGBoostModelSklearn:
    """GradientBoosting surrogate for XGBoost."""

    def __init__(self, config: Dict):
        self.model = GradientBoostingRegressor(
            n_estimators=min(config.get("n_estimators", 300), 100),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.05),
            random_state=config.get("random_state", 42),
            subsample=config.get("subsample", 0.8),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        self.model.fit(X, y)
        logger.info("[GBM/XGBoost] Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str):
        import joblib
        self.model = joblib.load(path)


class CatBoostModelSklearn:
    """RandomForest surrogate for CatBoost."""

    def __init__(self, config: Dict):
        self.model = RandomForestRegressor(
            n_estimators=min(config.get("iterations", 300), 100),
            max_depth=config.get("depth", 6),
            random_state=config.get("random_seed", 42),
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        self.model.fit(X, y)
        logger.info("[RF/CatBoost] Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str):
        import joblib
        self.model = joblib.load(path)


class DynamicFusionSklearn:
    """MLP-based dynamic fusion."""

    def __init__(self, tabular_dim: int, config: Dict, seed: int = 42):
        self.scaler = StandardScaler()
        self.weight_model = MLPRegressor(
            hidden_layer_sizes=(32,),
            max_iter=config.get("epochs", 50),
            random_state=seed,
        )
        # Separate models per base learner weight
        self.meta = Ridge()

    def fit(self, X_tab: np.ndarray, base_preds: np.ndarray, y: np.ndarray):
        X_all = np.hstack([self.scaler.fit_transform(X_tab), base_preds])
        self.meta.fit(X_all, y)
        logger.info("[DynamicFusion-sklearn] Training complete.")

    def predict(self, X_tab: np.ndarray, base_preds: np.ndarray):
        X_all = np.hstack([self.scaler.transform(X_tab), base_preds])
        preds = self.meta.predict(X_all)
        # Generate pseudo weights from Ridge coefficients
        n_models = base_preds.shape[1]
        weights = np.abs(self.meta.coef_[-n_models:])
        weights = weights / (weights.sum() + 1e-10)
        dummy_weights = np.tile(weights, (len(preds), 1))
        return preds, dummy_weights


class StackingFusionSklearn:
    """Stacking with sklearn meta-learner."""

    def __init__(self, meta_model_type: str = "linear"):
        if meta_model_type == "linear":
            self.meta_model = Ridge(alpha=1.0)
        else:
            self.meta_model = MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=42)

    def fit(self, base_preds: np.ndarray, y: np.ndarray):
        self.meta_model.fit(base_preds, y)

    def predict(self, base_preds: np.ndarray) -> np.ndarray:
        return self.meta_model.predict(base_preds)


class MAMLSklearn:
    """
    Simplified MAML analogue using per-task fine-tuning of a base MLP.
    True MAML requires gradient-through-gradient (PyTorch).
    This implements a multi-task initialization + task-specific fine-tuning.
    """

    def __init__(self, input_dim: int, config: Dict, seed: int = 42):
        self.config = config
        self.seed = seed
        self.meta_model = MLPRegressor(
            hidden_layer_sizes=(config.get("hidden_size", 64), 32),
            max_iter=config.get("meta_epochs", 100),
            random_state=seed,
        )
        self.scaler = StandardScaler()

    def meta_train(self, tasks):
        """Train on all tasks combined (multi-task initialization)."""
        all_X = np.vstack([t["tabular"] for t in tasks])
        all_y = np.concatenate([t["y_reg"] for t in tasks])
        X_scaled = self.scaler.fit_transform(all_X)
        self.meta_model.fit(X_scaled, all_y)
        logger.info("[MAML-sklearn] Meta-training complete.")

    def fine_tune(self, support_x: np.ndarray, support_y: np.ndarray, n_steps: int = 10):
        """Fine-tune a copy of the meta-model on task support set."""
        adapted = copy.deepcopy(self.meta_model)
        adapted.max_iter = n_steps
        X_scaled = self.scaler.transform(support_x)
        adapted.fit(X_scaled, support_y)
        return adapted

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return model.predict(X_scaled)

    def save(self, path: str):
        import joblib
        joblib.dump((self.meta_model, self.scaler), path)

    def load(self, path: str):
        import joblib
        self.meta_model, self.scaler = joblib.load(path)
