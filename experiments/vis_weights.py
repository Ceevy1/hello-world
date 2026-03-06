from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junyi.dataloader import JunyiConfig, JunyiDataBuilder, index_samples
from src.models.dynamic_junyi import JunyiDynamicModel, JunyiModelConfig, JunyiTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="outputs/junyi/attention_weights.csv")
    parser.add_argument("--out_fig", type=str, default="figures/junyi/attention_heatmap.png")
    args = parser.parse_args()

    builder = JunyiDataBuilder(JunyiConfig(max_seq_len=args.max_seq_len))
    df = builder.load_logs(args.log_csv)
    samples = builder.build_samples(df)

    idx = np.arange(len(samples["target"]))
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]
    train_data = index_samples(samples, train_idx)
    test_data = index_samples(samples, test_idx)

    cfg = JunyiModelConfig(n_exercises=len(builder.exercise2id), epochs=8)
    model = JunyiDynamicModel(cfg)
    trainer = JunyiTrainer(model, cfg)
    trainer.fit(train_data)

    _, attn = trainer.predict(test_data)
    k = int(np.clip(args.sample_idx, 0, len(attn) - 1))
    w = attn[k]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestep": np.arange(1, len(w) + 1), "attention": w}).to_csv(out_csv, index=False)

    # 细粒度展示为 3 行：content/response/continuous（同一时间权重映射到三模态，便于论文可视化）
    mat = np.vstack([w, w, w])
    plt.figure(figsize=(12, 2.8))
    sns.heatmap(mat, cmap="YlOrRd", cbar=True, xticklabels=10, yticklabels=["content", "response", "continuous"])
    plt.xlabel("Time step")
    plt.title("Temporal Attention Weights (Sample)")
    plt.tight_layout()

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=220)
    plt.close()
    print(f"Saved: {out_csv}\nSaved: {out_fig}")


if __name__ == "__main__":
    main()
