from __future__ import annotations

import pandas as pd


def align_features(oulad_df: pd.DataFrame, junyi_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_features = [c for c in ["activity_count", "avg_time", "correct_rate", "entropy"] if c in oulad_df.columns and c in junyi_df.columns]
    if not common_features:
        common_features = sorted(list((set(oulad_df.columns) & set(junyi_df.columns)) - {"label"}))[:4]
    return oulad_df[common_features + ["label"]].copy(), junyi_df[common_features + ["label"]].copy()
