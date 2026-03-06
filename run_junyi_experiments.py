from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from junyi.dataloader import JunyiConfig, JunyiDataBuilder, index_samples, split_by_exercise
from src.models.dynamic_junyi import JunyiDynamicModel, JunyiModelConfig, JunyiTrainer


def _inject_noise(data: Dict[str, np.ndarray], p: float, seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    noisy = {k: np.copy(v) for k, v in data.items()}
    flip = rng.random(noisy["target"].shape[0]) < p
    noisy["target"][flip] = 1.0 - noisy["target"][flip]
    drop = rng.random(noisy["continuous"].shape[0]) < p
    noisy["continuous"][drop, :, 0] = 0.0
    return noisy


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    y_hat = (prob >= 0.5).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, y_hat)),
        "AUC": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "F1": float(f1_score(y_true, y_hat)),
    }


def _synthetic_junyi(path: Path, n_users: int = 120, n_ex: int = 60, n_steps: int = 50, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    base_time = pd.Timestamp("2024-01-01")
    for u in range(n_users):
        ability = rng.normal(0, 1)
        for i in range(n_steps):
            ex_id = int(rng.integers(1, n_ex + 1))
            elapsed = max(1.0, rng.gamma(shape=2.0, scale=8.0) - ability * 1.5)
            hint = int(rng.integers(0, 2) if elapsed < 12 else rng.integers(0, 3))
            p_correct = 1 / (1 + np.exp(-(ability + (0.2 if hint == 0 else -0.3) - 0.02 * elapsed)))
            corr = int(rng.random() < p_correct)
            rows.append(
                {
                    "user_id": f"u{u}",
                    "exercise": f"ex{ex_id}",
                    "problem_type": "math",
                    "correct": corr,
                    "elapsed_time": elapsed,
                    "total_sec_taken": elapsed,
                    "hint_used": hint,
                    "timestamp": base_time + pd.Timedelta(minutes=(u * n_steps + i)),
                }
            )
    out = path / "junyi_ProblemLog_synthetic.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def run_all(log_csv: Path, exercise_csv: Path | None, out_dir: Path, fig_dir: Path, max_seq_len: int, epochs: int, sample_idx: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    builder = JunyiDataBuilder(JunyiConfig(max_seq_len=max_seq_len))
    logs = builder.load_logs(log_csv)
    samples = builder.build_samples(logs)

    adj = None
    if exercise_csv is not None and exercise_csv.exists():
        adj, _ = builder.build_exercise_graph(exercise_csv)

    # 1) Standard split + baseline comparison
    idx = np.arange(len(samples["target"]))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    train_data = index_samples(samples, tr)
    test_data = index_samples(samples, te)

    cfg = JunyiModelConfig(n_exercises=len(builder.exercise2id), epochs=epochs)
    model = JunyiDynamicModel(cfg, adj_matrix=adj)
    trainer = JunyiTrainer(model, cfg)
    trainer.fit(train_data)
    prob, attn = trainer.predict(test_data)

    base_x_train = np.column_stack([
        train_data["exercise_ids"].mean(axis=1),
        train_data["response_ids"].mean(axis=1),
        train_data["continuous"].mean(axis=(1, 2)),
    ])
    base_x_test = np.column_stack([
        test_data["exercise_ids"].mean(axis=1),
        test_data["response_ids"].mean(axis=1),
        test_data["continuous"].mean(axis=(1, 2)),
    ])
    lr = LogisticRegression(max_iter=1000)
    lr.fit(base_x_train, train_data["target"])
    lr_prob = lr.predict_proba(base_x_test)[:, 1]

    comparison = pd.DataFrame([
        {"Model": "LogisticRegression", **_metrics(test_data["target"], lr_prob)},
        {"Model": "JunyiDynamicModel", **_metrics(test_data["target"], prob)},
    ])
    comparison.to_csv(out_dir / "main_comparison.csv", index=False)

    # 2) Generalization (seen->unseen exercises)
    g_tr, g_te = split_by_exercise(samples, train_ratio=0.8)
    g_train = index_samples(samples, g_tr)
    g_test = index_samples(samples, g_te)
    g_model = JunyiDynamicModel(cfg, adj_matrix=adj)
    g_trainer = JunyiTrainer(g_model, cfg)
    g_trainer.fit(g_train)
    g_prob, _ = g_trainer.predict(g_test)
    generalization = pd.DataFrame([
        {
            "Split": "80% seen exercise -> 20% unseen exercise",
            **_metrics(g_test["target"], g_prob),
            "N_train": int(len(g_tr)),
            "N_test": int(len(g_te)),
        }
    ])
    generalization.to_csv(out_dir / "generalization_results.csv", index=False)

    # 3) Robustness
    robust_rows = []
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        noisy = _inject_noise(test_data, p)
        n_prob, _ = trainer.predict(noisy)
        robust_rows.append({"noise_rate": p, **_metrics(noisy["target"], n_prob)})
    robustness = pd.DataFrame(robust_rows)
    robustness.to_csv(out_dir / "robustness_results.csv", index=False)

    # 4) Attention export + visualization
    k = int(np.clip(sample_idx, 0, len(attn) - 1))
    w = attn[k]
    pd.DataFrame({"timestep": np.arange(1, len(w) + 1), "attention": w}).to_csv(out_dir / "attention_weights.csv", index=False)

    mat = np.vstack([w, w, w])
    plt.figure(figsize=(12, 3))
    sns.heatmap(mat, cmap="YlOrRd", yticklabels=["content", "response", "continuous"], xticklabels=10)
    plt.xlabel("Time step")
    plt.title("Junyi Temporal Attention Heatmap")
    plt.tight_layout()
    plt.savefig(fig_dir / "attention_heatmap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(robustness["noise_rate"], robustness["Accuracy"], marker="o", label="Accuracy")
    plt.plot(robustness["noise_rate"], robustness["F1"], marker="s", label="F1")
    plt.xlabel("Noise rate")
    plt.ylabel("Score")
    plt.title("Junyi Robustness Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "robustness_curve.png", dpi=220)
    plt.close()

    manifest = pd.DataFrame([
        {
            "main_comparison": str(out_dir / "main_comparison.csv"),
            "generalization": str(out_dir / "generalization_results.csv"),
            "robustness": str(out_dir / "robustness_results.csv"),
            "attention_weights": str(out_dir / "attention_weights.csv"),
            "attention_heatmap": str(fig_dir / "attention_heatmap.png"),
            "robustness_curve": str(fig_dir / "robustness_curve.png"),
        }
    ])
    manifest.to_csv(out_dir / "run_manifest.csv", index=False)
    print(f"Saved unified Junyi outputs to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click Junyi training + evaluation runner")
    parser.add_argument("--log_csv", type=str, default=None, help="Path to junyi_ProblemLog_original.csv")
    parser.add_argument("--exercise_csv", type=str, default=None, help="Path to junyi_Exercise_table.csv")
    parser.add_argument("--use_synthetic_if_missing", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs/junyi")
    parser.add_argument("--fig_dir", type=str, default="figures/junyi")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)

    log_csv = Path(args.log_csv) if args.log_csv else None
    exercise_csv = Path(args.exercise_csv) if args.exercise_csv else None

    if log_csv is None or not log_csv.exists():
        if not args.use_synthetic_if_missing:
            raise FileNotFoundError("Junyi log csv not found. Provide --log_csv or set --use_synthetic_if_missing.")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_csv = _synthetic_junyi(out_dir)
        print(f"Using synthetic Junyi log data: {log_csv}")

    run_all(
        log_csv=log_csv,
        exercise_csv=exercise_csv,
        out_dir=out_dir,
        fig_dir=fig_dir,
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        sample_idx=args.sample_idx,
    )


if __name__ == "__main__":
    main()
