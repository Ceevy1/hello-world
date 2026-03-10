from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import classification_metrics
from feature_engineering.behavior_features import build_behavior_features
from preprocess.oulad_preprocess import preprocess_oulad
from training.trainer import DynamicFusionTrainer, TrainerConfig


def _build_sequence(merged: pd.DataFrame, max_len: int = 200, feature_dim: int = 32) -> np.ndarray:
    seq = np.zeros((len(merged), max_len, feature_dim), dtype=np.float32)
    seq[:, :, 0] = merged["activity_entropy"].to_numpy()[:, None]
    seq[:, :, 1] = merged["active_week_ratio"].to_numpy()[:, None]
    seq[:, :, 2] = merged["procrastination_index"].to_numpy()[:, None]
    seq[:, :, 3] = merged["resource_switch_rate"].to_numpy()[:, None]
    seq[:, :, 4] = merged["avg_session_length"].to_numpy()[:, None]
    return seq


def run_oulad_experiment(input_csv: str, out_csv: str = "results/oulad_results.csv") -> pd.DataFrame:
    raw = pd.read_csv(input_csv)
    data = preprocess_oulad(raw)
    features = build_behavior_features(data, student_col="student_id")
    y = data.groupby("student_id")["pass_label"].max().reset_index()
    week = data.groupby("student_id")["week"].max().reset_index(name="week")

    merged = features.merge(y, on="student_id").merge(week, on="student_id")
    tab = merged[["activity_entropy", "active_week_ratio", "procrastination_index", "resource_switch_rate", "avg_session_length"]].to_numpy()
    seq = _build_sequence(merged)

    trainer = DynamicFusionTrainer(TrainerConfig())
    trainer.fit(seq, tab, merged["pass_label"].to_numpy(), merged["week"].to_numpy())

    preds = trainer.predict_proba(seq, tab, merged["week"].to_numpy())
    metrics = classification_metrics(merged["pass_label"].to_numpy(), preds)
    result_df = pd.DataFrame([{"Model": "DynamicFusion-Enhanced", **metrics}])

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/oulad_results.csv")
    args = parser.parse_args()
    print(run_oulad_experiment(args.input, args.output))
