"""Feature semantic alignment between OULAD and custom domain."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler


class FeatureAligner:
    """Map source/target features into shared semantic latent factors."""

    def __init__(self) -> None:
        self.scalers: dict[str, object] = {}

    def align_custom(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["performance"] = self._fit_transform("performance", MinMaxScaler(), df[["final_score"]]).ravel()
        out["engagement"] = self._fit_transform(
            "engagement",
            StandardScaler(),
            df[["exercise_1", "exercise_2", "exercise_3", "lab_total"]].mean(axis=1).to_frame(),
        ).ravel()
        out["behavior"] = self._fit_transform(
            "behavior",
            StandardScaler(),
            df[["regular_score", "report"]].mean(axis=1).to_frame(),
        ).ravel()

        q_scaler = QuantileTransformer(output_distribution="normal", n_quantiles=min(100, len(df)))
        out["background"] = q_scaler.fit_transform(df[["exam_score"]]).ravel()
        self.scalers["background_quantile"] = q_scaler

        if "target" in df.columns:
            out["target"] = pd.to_numeric(df["target"], errors="coerce")
        return out

    def align_oulad(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["performance"] = self._safe_col(df, ["score", "assessment_score", "final_result_score"]).astype(float)
        out["engagement"] = self._safe_col(df, ["num_clicks", "sum_click", "mean_click_per_activity"]).astype(float)
        out["behavior"] = self._safe_col(df, ["studied_credits", "num_of_prev_attempts", "click_stability"]).astype(float)
        out["background"] = self._safe_col(df, ["imd_band", "highest_education", "age_band_code"]).astype(float)

        for col in ["performance", "engagement", "behavior", "background"]:
            scaler = StandardScaler()
            out[col] = scaler.fit_transform(out[[col]]).ravel()
            self.scalers[f"source_{col}"] = scaler

        target_col = None
        for cand in ["target", "score", "assessment_score", "final_result_score"]:
            if cand in df.columns:
                target_col = cand
                break
        if target_col is not None:
            out["target"] = pd.to_numeric(df[target_col], errors="coerce")
        return out

    def align(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        domain = domain.lower()
        if domain == "custom":
            return self.align_custom(df)
        if domain == "oulad":
            return self.align_oulad(df)
        raise ValueError(f"未知domain: {domain}")

    def _fit_transform(self, key: str, scaler: object, values: pd.DataFrame) -> np.ndarray:
        transformed = scaler.fit_transform(values)
        self.scalers[key] = scaler
        return transformed

    @staticmethod
    def _safe_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
        for col in candidates:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").fillna(0)
                return s
        return pd.Series(np.zeros(len(df)), index=df.index)
