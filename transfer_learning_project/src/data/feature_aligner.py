"""Feature semantic alignment between OULAD and custom domain."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler


class FeatureAligner:
    """Map custom features to source-domain-like aligned space."""

    def __init__(self) -> None:
        self.scalers: dict[str, object] = {}

    def align(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        out["assessment_score"] = self._fit_transform("exam", MinMaxScaler(), df[["exam_score"]]).ravel()

        out["mean_click_per_activity"] = self._fit_transform(
            "exercise_mean", StandardScaler(), df[["exercise_1", "exercise_2", "exercise_3"]].mean(axis=1).to_frame()
        ).ravel()

        lab_cols = [f"lab_{i}" for i in range(1, 8)]
        lab_seq = df[lab_cols].interpolate(axis=1, limit_direction="both")
        for idx, col in enumerate(lab_cols):
            out[f"vle_interaction_t{idx+1}"] = lab_seq[col]

        out["tma_score"] = self._fit_transform("report", MinMaxScaler(), df[["report"]]).ravel()

        q_scaler = QuantileTransformer(output_distribution="normal", n_quantiles=min(100, len(df)))
        out["sum_vle_interactions"] = q_scaler.fit_transform(df[["lab_total"]]).ravel()
        self.scalers["lab_total_quantile"] = q_scaler

        out["studied_credits"] = df["regular_score"].rank(method="average") / len(df)
        out["final_result_score"] = self._fit_transform("final", MinMaxScaler(), df[["final_score"]]).ravel()

        if "exercise_cv" in df.columns:
            out["click_stability"] = self._fit_transform("stability", StandardScaler(), df[["exercise_cv"]]).ravel()
        return out

    def _fit_transform(self, key: str, scaler: object, values: pd.DataFrame) -> np.ndarray:
        transformed = scaler.fit_transform(values)
        self.scalers[key] = scaler
        return transformed
