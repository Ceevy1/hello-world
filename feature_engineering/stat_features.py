from __future__ import annotations

import numpy as np


def behavior_entropy(click_series: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    probs = click_series / np.maximum(click_series.sum(axis=1, keepdims=True), eps)
    return -(probs * np.log(np.maximum(probs, eps))).sum(axis=1)


def activity_diversity(active_days: np.ndarray) -> np.ndarray:
    return (active_days > 0).sum(axis=1) / np.maximum(active_days.shape[1], 1)


def extract_features(x_seq: np.ndarray, x_stat: np.ndarray) -> np.ndarray:
    clicks = np.maximum(x_seq[:, :, 0], 0)
    days = np.maximum(x_seq[:, :, 1], 0) if x_seq.shape[-1] > 1 else np.zeros_like(clicks)

    engineered = np.column_stack(
        [
            clicks.sum(axis=1),
            clicks.mean(axis=1),
            clicks.std(axis=1),
            behavior_entropy(clicks),
            activity_diversity(days),
            (clicks > 0).sum(axis=1),
        ]
    )
    return np.concatenate([x_stat, engineered.astype(np.float32)], axis=1)
