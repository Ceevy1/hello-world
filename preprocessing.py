from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


EXERCISE_COLS = ["练习1", "练习2", "练习3"]
LAB_COLS = ["实验1", "实验2", "实验3", "实验4", "实验5", "实验6", "实验7"]
STATIC_COLS = ["考勤", "报告", "平时成绩", "总平时 成绩", "总实验 成绩", "总期末成绩"]
TARGET_COL = "总评成绩"
ID_COL = "序号"


@dataclass
class PreparedData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    exercise_cols: List[str]
    lab_cols: List[str]
    static_cols: List[str]
    target_col: str



def preprocess_scores(
    csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    standardize: bool = True,
) -> PreparedData:
    df = pd.read_csv(csv_path)
    for col in [ID_COL] + EXERCISE_COLS + LAB_COLS + STATIC_COLS + [TARGET_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    data_cols = EXERCISE_COLS + LAB_COLS + STATIC_COLS + [TARGET_COL]
    df[data_cols] = df[data_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    df[data_cols] = df[data_cols].fillna(df[data_cols].median(numeric_only=True))

    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    adjusted_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=adjusted_val_size, random_state=random_state)

    if standardize:
        scaler = StandardScaler()
        feature_cols = EXERCISE_COLS + LAB_COLS + STATIC_COLS
        train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
        test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    return PreparedData(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        exercise_cols=EXERCISE_COLS,
        lab_cols=LAB_COLS,
        static_cols=STATIC_COLS,
        target_col=TARGET_COL,
    )


def build_classification_labels(df: pd.DataFrame, score_col: str = TARGET_COL) -> Tuple[pd.DataFrame, int]:
    """Generate 4-level labels: 0<60, 1:[60,70), 2:[70,85), 3>=85."""
    bins = [-float("inf"), 60, 70, 85, float("inf")]
    df = df.copy()
    df["label_cls"] = pd.cut(df[score_col], bins=bins, labels=False, right=False)
    return df, 4
