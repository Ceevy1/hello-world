from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junyi.dataloader import JunyiConfig, JunyiDataBuilder, index_samples, split_by_exercise
from src.models.dynamic_junyi import JunyiDynamicModel, JunyiModelConfig, JunyiTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv", type=str, required=True)
    parser.add_argument("--exercise_csv", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--out", type=str, default="outputs/junyi/generalization_results.csv")
    args = parser.parse_args()

    builder = JunyiDataBuilder(JunyiConfig(max_seq_len=args.max_seq_len))
    df = builder.load_logs(args.log_csv)
    samples = builder.build_samples(df)

    adj = None
    if args.exercise_csv:
        adj, _ = builder.build_exercise_graph(args.exercise_csv)

    train_idx, test_idx = split_by_exercise(samples, train_ratio=0.8)
    train_data = index_samples(samples, train_idx)
    test_data = index_samples(samples, test_idx)

    model_cfg = JunyiModelConfig(n_exercises=len(builder.exercise2id), epochs=8)
    model = JunyiDynamicModel(model_cfg, adj_matrix=adj)
    trainer = JunyiTrainer(model, model_cfg)
    trainer.fit(train_data)

    prob, _ = trainer.predict(test_data)
    y_true = test_data["target"]
    y_hat = (prob >= 0.5).astype(int)

    metrics = {
        "Split": "80% seen exercise train -> 20% unseen exercise test",
        "Accuracy": float(accuracy_score(y_true, y_hat)),
        "AUC": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "N_train": int(len(train_idx)),
        "N_test": int(len(test_idx)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
