from __future__ import annotations

import pandas as pd

from feature_engineering.time_features import add_week_index


PASS_LABELS = {"Pass", "Distinction"}


def preprocess_oulad(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns and "week" not in out.columns:
        out = add_week_index(out, "timestamp")
    out["pass_label"] = out["final_result"].isin(PASS_LABELS).astype(int)
    return out
