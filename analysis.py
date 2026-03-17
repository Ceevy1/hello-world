from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wilcoxon
from sklearn.manifold import TSNE


def compute_group_representation_distances(repr_map: dict[str, np.ndarray]) -> pd.DataFrame:
    groups = sorted(repr_map)
    centroids = {g: np.mean(repr_map[g], axis=0) for g in groups if repr_map[g].size > 0}
    rows = []
    for g1 in groups:
        for g2 in groups:
            if g1 not in centroids or g2 not in centroids:
                dist = np.nan
            else:
                dist = float(np.linalg.norm(centroids[g1] - centroids[g2]))
            rows.append({"group_a": g1, "group_b": g2, "distance": dist})
    return pd.DataFrame(rows)


def compute_weight_analysis(weight_map: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, arr in weight_map.items():
        if arr.size == 0:
            continue
        rows.append({
            "scenario": name,
            "w_seq_mean": float(arr[:, 0].mean()),
            "w_seq_std": float(arr[:, 0].std()),
            "w_kg_mean": float(arr[:, 1].mean()),
            "w_kg_std": float(arr[:, 1].std()),
            "w_stat_mean": float(arr[:, 2].mean()),
            "w_stat_std": float(arr[:, 2].std()),
        })
    return pd.DataFrame(rows)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    s = diff.std(ddof=1)
    return 0.0 if s == 0 else float(diff.mean() / s)


def stat_tests_from_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = df[df["model"] == "DynamicFusion-Enhanced"]
    for model in sorted(m for m in df["model"].unique() if m != "DynamicFusion-Enhanced"):
        other = df[df["model"] == model]
        merged = target.merge(other, on=["scenario", "seed"], suffixes=("_dyn", "_other"))
        if merged.empty:
            continue
        for metric in ["AUC", "Accuracy", "F1"]:
            a = merged[f"{metric}_dyn"].to_numpy()
            b = merged[f"{metric}_other"].to_numpy()
            if len(a) < 2:
                continue
            try:
                _, p_w = wilcoxon(a, b)
            except ValueError:
                p_w = 1.0
            # chi-square on thresholded wins/losses
            wins = int((a > b).sum())
            losses = int((a < b).sum())
            ties = int((a == b).sum())
            chi2_p = 1.0
            if wins + losses > 0:
                chi2_p = float(chi2_contingency([[wins, losses], [losses, wins]])[1])
            rows.append(
                {
                    "scene_a": "DynamicFusion-Enhanced",
                    "scene_b": model,
                    "metric": metric,
                    "wilcoxon_p": float(p_w),
                    "chi2_p": float(chi2_p),
                    "cohen_d": cohen_d(a, b),
                    "ties": ties,
                }
            )
    return pd.DataFrame(rows)


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for name, (fpr, tpr) in curves.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves by Scenario")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_weight_trajectories(weight_map: dict[str, np.ndarray], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, arr in weight_map.items():
        if arr.size == 0:
            continue
        plt.plot(arr[:, 0], alpha=0.6, label=f"{name}-seq")
    plt.xlabel("Step")
    plt.ylabel("w_seq")
    plt.title("Weight Trajectories (seq branch)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_tsne_embeddings(repr_map: dict[str, np.ndarray], out_path: Path, max_points: int = 800) -> None:
    X, y = [], []
    for name, arr in repr_map.items():
        if arr.size == 0:
            continue
        sub = arr[: max_points // max(1, len(repr_map))]
        X.append(sub)
        y.extend([name] * len(sub))
    if not X:
        return
    X_all = np.vstack(X)
    if X_all.shape[0] < 5:
        return
    emb = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, X_all.shape[0] // 4))).fit_transform(X_all)
    plt.figure(figsize=(7, 5))
    y = np.array(y)
    for name in np.unique(y):
        m = y == name
        plt.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.7, label=name)
    plt.legend(fontsize=7)
    plt.title("t-SNE of Representations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_markdown_report(summary: pd.DataFrame, stat_df: pd.DataFrame, out_path: Path) -> None:
    try:
        summary_txt = summary.to_markdown(index=False)
        stat_txt = stat_df.to_markdown(index=False)
    except Exception:
        summary_txt = summary.to_csv(index=False)
        stat_txt = stat_df.to_csv(index=False)
    lines = ["# Generalization Experiment Report", "", "## Performance Summary", "", summary_txt, "", "## Statistical Tests", "", stat_txt]
    out_path.write_text("\n".join(lines), encoding="utf-8")
