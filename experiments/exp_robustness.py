from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junyi.dataloader import JunyiConfig, JunyiDataBuilder, index_samples
from src.models.dynamic_junyi import JunyiDynamicModel, JunyiModelConfig, JunyiTrainer


def inject_noise(data: dict[str, np.ndarray], p: float, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    noisy = {k: np.copy(v) for k, v in data.items()}

    flip = rng.random(size=noisy["target"].shape[0]) < p
    noisy["target"][flip] = 1.0 - noisy["target"][flip]

    drop = rng.random(size=noisy["continuous"].shape[0]) < p
    noisy["continuous"][drop, :, 0] = 0.0
    return noisy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--out", type=str, default="outputs/junyi/robustness_results.csv")
    args = parser.parse_args()

    builder = JunyiDataBuilder(JunyiConfig(max_seq_len=args.max_seq_len))
    df = builder.load_logs(args.log_csv)
    samples = builder.build_samples(df)

    idx = np.arange(len(samples["target"]))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    train_data = index_samples(samples, tr)
    test_data = index_samples(samples, te)

    cfg = JunyiModelConfig(n_exercises=len(builder.exercise2id), epochs=8)
    model = JunyiDynamicModel(cfg)
    trainer = JunyiTrainer(model, cfg)
    trainer.fit(train_data)

    rows = []
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        eval_data = inject_noise(test_data, p=p, seed=42)
        prob, _ = trainer.predict(eval_data)
        y = eval_data["target"]
        yh = (prob >= 0.5).astype(int)
        rows.append(
            {
                "noise_rate": p,
                "Accuracy": float(accuracy_score(y, yh)),
                "AUC": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
                "F1": float(f1_score(y, yh)),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
