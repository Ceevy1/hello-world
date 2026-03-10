from __future__ import annotations

import pandas as pd


def add_week_index(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    start = ts.min()
    out["week"] = ((ts - start).dt.days // 7 + 1).astype(int)
    return out
