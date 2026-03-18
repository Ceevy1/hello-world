from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from feature_engineering import extract_features


@dataclass(frozen=True)
class TemporalPredictionResult:
    probabilities: dict[object, np.ndarray]
    features: dict[object, np.ndarray]
    week_indices: dict[object, np.ndarray]



def slice_temporal_data(x_seq: np.ndarray, x_stat: np.ndarray, week: int | str, max_weeks: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_seq = np.asarray(x_seq)
    x_stat = np.asarray(x_stat)
    total_weeks = max_weeks or x_seq.shape[1]
    if week == "full":
        week_idx = np.full(len(x_seq), total_weeks - 1, dtype=np.int64)
        return x_seq.copy(), extract_features(x_seq, x_stat), week_idx

    cutoff = int(week)
    subset = x_seq.copy()
    if cutoff < total_weeks:
        subset[:, cutoff:, :] = 0.0
    week_idx = np.full(len(x_seq), min(cutoff - 1, total_weeks - 1), dtype=np.int64)
    return subset, extract_features(subset, x_stat), week_idx



def temporal_predict(model, x_seq: np.ndarray, x_stat: np.ndarray, weeks: Iterable[int | str] = (4, 8, "full")) -> TemporalPredictionResult:
    results: dict[object, np.ndarray] = {}
    feature_views: dict[object, np.ndarray] = {}
    week_indices: dict[object, np.ndarray] = {}

    for week in weeks:
        _, x_features, week_idx = slice_temporal_data(x_seq, x_stat, week)
        results[week] = model.predict(x_features)
        feature_views[week] = x_features
        week_indices[week] = week_idx

    return TemporalPredictionResult(probabilities=results, features=feature_views, week_indices=week_indices)
