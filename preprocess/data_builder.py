"""Data construction utilities aligned with MT-HAFNet PRD."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def reconstruct_score(student_assessment: pd.DataFrame, assessments: pd.DataFrame) -> pd.Series:
    merged = student_assessment.merge(
        assessments[["id_assessment", "weight"]], on="id_assessment", how="left"
    )
    merged["weighted"] = merged["score"] * merged["weight"] / 100.0
    score = merged.groupby("id_student")["weighted"].sum().clip(0, 100)
    return score


def _entropy(counts: pd.Series) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def build_weekly_sequence(student_vle: pd.DataFrame, vle: pd.DataFrame, t_weeks: int = 30) -> pd.DataFrame:
    df = student_vle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    df["week"] = (df["date"] // 7).clip(lower=0, upper=t_weeks - 1)

    rows = []
    for (sid, wk), g in df.groupby(["id_student", "week"]):
        rows.append(
            {
                "id_student": sid,
                "week": int(wk),
                "weekly_clicks": g["sum_click"].sum(),
                "active_days": g["date"].nunique(),
                "resource_types": g["activity_type"].nunique(),
                "behavior_entropy": _entropy(g.groupby("activity_type")["sum_click"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_tabular_features(student_vle: pd.DataFrame, student_info: pd.DataFrame) -> pd.DataFrame:
    agg = student_vle.groupby("id_student")["sum_click"].agg(["sum", "mean", "std", "count"]).reset_index()
    agg = agg.rename(
        columns={"sum": "total_clicks", "mean": "click_mean", "std": "click_std", "count": "active_weeks"}
    ).fillna(0)
    feat = student_info.merge(agg, on="id_student", how="left").fillna(0)
    return feat


@dataclass
class SplitPack:
    train_idx: np.ndarray
    test_idx: np.ndarray


def split_random(n: int, test_size: float = 0.2, seed: int = 42) -> SplitPack:
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    return SplitPack(tr, te)


def split_lomo(module_codes: Iterable[str], test_module: str) -> SplitPack:
    modules = np.asarray(list(module_codes))
    test_idx = np.where(modules == test_module)[0]
    train_idx = np.where(modules != test_module)[0]
    return SplitPack(train_idx, test_idx)


def split_lopo(presentations: Iterable[str], test_presentation: str) -> SplitPack:
    pres = np.asarray(list(presentations))
    test_idx = np.where(pres == test_presentation)[0]
    train_idx = np.where(pres != test_presentation)[0]
    return SplitPack(train_idx, test_idx)


def truncate_sequence(x_seq: np.ndarray, weeks: int | None) -> np.ndarray:
    return x_seq if weeks is None else x_seq[:, :weeks, :]
