from __future__ import annotations

import pandas as pd

from feature_engineering.time_features import add_week_index


def preprocess_junyi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns and "week" not in out.columns:
        out = add_week_index(out, "timestamp")
    out["activity_type"] = out.get("exercise_id", "exercise").astype(str)
    out["click_count"] = 1
    out["pass_label"] = out["correct"].astype(int)
    return out
