"""Feature engineering for transfer learning."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class FeatureEngineer:
    """Feature cleaning, derivation, and selection utilities."""

    def handle_missing(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result = df.copy()
        if strategy == "mean":
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
        elif strategy == "median":
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
        elif strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
            result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
        else:
            raise ValueError(f"未知缺失值策略: {strategy}")
        return result

    def remove_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result = df.copy()
        if method == "iqr":
            q1 = result[numeric_cols].quantile(0.25)
            q3 = result[numeric_cols].quantile(0.75)
            iqr = q3 - q1
            mask = ~((result[numeric_cols] < (q1 - 1.5 * iqr)) | (result[numeric_cols] > (q3 + 1.5 * iqr))).any(axis=1)
            result = result[mask]
        elif method == "zscore":
            z = (result[numeric_cols] - result[numeric_cols].mean()) / (result[numeric_cols].std() + 1e-8)
            result = result[(z.abs() < 3.0).all(axis=1)]
        elif method == "isolation_forest":
            iso = IsolationForest(random_state=42, contamination=0.05)
            keep = iso.fit_predict(result[numeric_cols]) == 1
            result = result[keep]
        else:
            raise ValueError(f"未知异常值处理方法: {method}")
        return result.reset_index(drop=True)

    def construct_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        exercise_cols = ["exercise_1", "exercise_2", "exercise_3"]
        lab_cols = [f"lab_{i}" for i in range(1, 8)]

        result["exercise_cv"] = result[exercise_cols].std(axis=1) / (result[exercise_cols].mean(axis=1) + 1e-6)

        def calc_trend(row: pd.Series) -> float:
            scores = row[lab_cols].values.astype(float)
            x = np.arange(len(scores))
            mask = ~np.isnan(scores)
            if mask.sum() < 2:
                return 0.0
            return float(np.polyfit(x[mask], scores[mask], 1)[0])

        result["lab_trend"] = result.apply(calc_trend, axis=1)
        result["lab_exam_ratio"] = result["lab_total"] / (result["exam_score"] + 1e-6)
        result["effort_index"] = result["regular_score"] / (result["final_score"] + 1e-6)
        result["report_ratio"] = result["report"] / (result["lab_total"] + 1e-6)
        return result

    def normalize_features(self, df: pd.DataFrame, method: str = "standard") -> tuple[pd.DataFrame, object]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result = df.copy()
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"未知归一化方法: {method}")
        result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
        return result, scaler

    def encode_target(self, df: pd.DataFrame, bins: Iterable[float], labels: Iterable[str]) -> pd.DataFrame:
        result = df.copy()
        result["target_label"] = pd.cut(result["target"], bins=bins, labels=labels, include_lowest=True)
        return result

    def feature_importance_filter(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.01) -> list[str]:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        return [col for col, imp in zip(X.columns, model.feature_importances_) if imp >= threshold]
