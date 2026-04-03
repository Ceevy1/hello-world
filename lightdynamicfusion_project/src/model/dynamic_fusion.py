import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.feature_engineering import FeatureEngineering
from src.data.augmentation import gaussian_augment, mixup_augment
from src.model.group_predictor import build_estimator
from src.model.attention_fusion import AttentionFusion
from src.model.stage_manager import StageManager


class LightDynamicFusion:
    def __init__(self, stage: str = 'T4', base_estimator: str = 'ridge',
                 use_data_augmentation: bool = True, aug_noise_std: float = 1.5,
                 aug_multiplier: int = 5, aug_method: str = 'gaussian', random_state: int = 42,
                 active_sources=None, use_attention: bool = True):
        self.stage = stage
        self.base_estimator = base_estimator
        self.use_data_augmentation = use_data_augmentation
        self.aug_noise_std = aug_noise_std
        self.aug_multiplier = aug_multiplier
        self.aug_method = aug_method
        self.random_state = random_state
        self.active_sources = set(active_sources) if active_sources else {'source1', 'source2', 'source3'}
        self.use_attention = use_attention

        self.feature_engineer = FeatureEngineering()
        self.group_predictors = {}
        self.attention_fusion = AttentionFusion()
        self.stage_manager = StageManager()
        self.is_fitted = False

    def fit(self, X_raw: pd.DataFrame, y: np.ndarray):
        stage = self.stage_manager.validate(self.stage)
        X = self.feature_engineer.build_features(X_raw, stage)
        groups = self.feature_engineer.get_feature_groups(stage)

        if self.use_data_augmentation:
            if self.aug_method in {'gaussian', 'combined'}:
                X, y = gaussian_augment(X, y, std=self.aug_noise_std, multiplier=self.aug_multiplier,
                                        random_state=self.random_state)
            if self.aug_method in {'mixup', 'combined'}:
                X, y = mixup_augment(X, y, alpha=0.2, random_state=self.random_state)

        group_preds = {}
        self.group_predictors = {}
        for source, cols in groups.items():
            if source not in self.active_sources:
                continue
            cols = [c for c in cols if c in X.columns]
            if not cols:
                continue
            scaler = StandardScaler()
            Xg = scaler.fit_transform(X[cols])
            est = build_estimator(self.base_estimator, self.random_state)
            est.fit(Xg, y)
            self.group_predictors[source] = (cols, scaler, est)
            group_preds[source] = est.predict(Xg)

        self.attention_fusion.fit(group_preds, y, stage, use_attention=self.use_attention)
        self.is_fitted = True
        return self

    def _group_predict(self, X_raw: pd.DataFrame):
        stage = self.stage_manager.validate(self.stage)
        X = self.feature_engineer.build_features(X_raw, stage)
        out = {}
        for source, (cols, scaler, est) in self.group_predictors.items():
            Xg = scaler.transform(X[cols])
            out[source] = est.predict(Xg)
        return out

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        gp = self._group_predict(X_raw)
        return self.attention_fusion.predict(gp, self.stage)

    def get_source_contributions(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        gp = self._group_predict(X_raw)
        y_pred = self.attention_fusion.predict(gp, self.stage)
        w = self.attention_fusion.get_attention_weights(self.stage)
        n = len(y_pred)
        return pd.DataFrame({
            'source1_weight': np.full(n, w['source1']),
            'source2_weight': np.full(n, w['source2']),
            'source3_weight': np.full(n, w['source3']),
            'prediction': y_pred,
        })

    def get_risk_level(self, X_raw: pd.DataFrame, thresholds: list = [60, 70, 80]) -> pd.Series:
        y = self.predict(X_raw)
        low, mid, _ = thresholds
        return pd.Series(np.where(y < low, '高风险', np.where(y < mid, '中风险', '低风险')))
