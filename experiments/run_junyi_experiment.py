from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import classification_metrics
from feature_engineering.behavior_features import build_behavior_features
from preprocess.junyi_preprocess import preprocess_junyi
from training.trainer import DynamicFusionTrainer, TrainerConfig


def run_junyi_experiment(input_csv: str, out_csv: str = "results/junyi_results.csv") -> pd.DataFrame:
    raw = pd.read_csv(input_csv)
    data = preprocess_junyi(raw)
    features = build_behavior_features(data, student_col="user_id")
    y = data.groupby("user_id")["pass_label"].mean().reset_index()
    y["pass_label"] = (y["pass_label"] >= 0.5).astype(int)
    week = data.groupby("user_id")["week"].max().reset_index(name="week")
    merged = features.merge(y, on="user_id").merge(week, on="user_id")

    tab = merged[["activity_entropy", "active_week_ratio", "procrastination_index", "resource_switch_rate", "avg_session_length"]].to_numpy()
    seq = np.repeat(tab[:, None, :], repeats=200, axis=1)
    seq = np.pad(seq, ((0, 0), (0, 0), (0, 27)))

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
    parser.add_argument("--output", default="results/junyi_results.csv")
    args = parser.parse_args()
    print(run_junyi_experiment(args.input, args.output))
