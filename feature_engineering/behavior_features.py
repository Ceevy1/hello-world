from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def compute_activity_entropy(activity_counts: dict[str, float]) -> float:
    """Compute Shannon entropy from activity counts."""
    values = np.array([v for v in activity_counts.values() if v > 0], dtype=float)
    if values.size == 0:
        return 0.0
    probs = values / values.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def compute_procrastination_index(student_logs: pd.DataFrame) -> float:
    """last_week_activity / total_activity using action counts."""
    if student_logs.empty:
        return 0.0
    last_week = student_logs["week"].max()
    total_activity = max(len(student_logs), 1)
    last_week_activity = int((student_logs["week"] == last_week).sum())
    return float(last_week_activity / total_activity)


def compute_resource_switch_rate(activity_sequence: Iterable[str]) -> float:
    seq = list(activity_sequence)
    if not seq:
        return 0.0
    switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    return float(switches / len(seq))


def build_behavior_features(df: pd.DataFrame, student_col: str = "student_id") -> pd.DataFrame:
    """Aggregate student-level structure features used by tree models."""
    rows = []
    for student_id, g in df.groupby(student_col):
        week_counts = g.groupby("week").size()
        total_weeks = int(max(g["week"].max(), 1))
        activity_counts = g["activity_type"].value_counts().to_dict()

        rows.append(
            {
                student_col: student_id,
                "activity_entropy": compute_activity_entropy(activity_counts),
                "active_week_ratio": float((week_counts > 0).sum() / max(total_weeks, 1)),
                "procrastination_index": compute_procrastination_index(g),
                "resource_switch_rate": compute_resource_switch_rate(g["activity_type"].tolist()),
                "avg_session_length": float(g.get("elapsed_time", pd.Series(np.ones(len(g)))).mean()),
            }
        )
    return pd.DataFrame(rows)
