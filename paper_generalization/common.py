from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

TARGET_COL = "总评成绩"
ID_COLS = {"序号", "id", "ID", "student_id"}


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def infer_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> List[str]:
    cols = []
    for c in df.columns:
        if c == target_col or c in ID_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_modal_splits(feature_cols: List[str]) -> Dict[str, List[str]]:
    perf, behav, eng = [], [], []
    for c in feature_cols:
        lc = c.lower()
        if any(k in c for k in ["实验", "期末", "总平时", "总实验", "报告"]) or any(
            k in lc for k in ["exam", "lab", "report", "practice"]
        ):
            perf.append(c)
        elif any(k in c for k in ["考勤", "平时"]) or "attendance" in lc:
            behav.append(c)
        else:
            eng.append(c)
    if not perf:
        perf = feature_cols[: max(1, len(feature_cols) // 3)]
    if not behav:
        behav = feature_cols[max(1, len(feature_cols) // 3) : max(2, 2 * len(feature_cols) // 3)]
    if not eng:
        eng = [c for c in feature_cols if c not in perf and c not in behav]
        if not eng:
            eng = feature_cols[-1:]
    return {"perf": perf, "behav": behav, "eng": eng}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_ci(metric_values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = np.percentile(metric_values, 100 * alpha / 2)
    hi = np.percentile(metric_values, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def ensure_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(data, out_path: str):
    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def kfold(n_samples: int, cv: int, seed: int = 42):
    cv = max(2, min(cv, n_samples))
    return KFold(n_splits=cv, shuffle=True, random_state=seed)


def make_scaler(X: np.ndarray) -> StandardScaler:
    return StandardScaler().fit(X)
