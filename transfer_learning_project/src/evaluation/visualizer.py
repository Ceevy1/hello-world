"""Visualization utilities for transfer-learning experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


class Visualizer:
    def __init__(self, output_dir: str = "results/figures", dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def _save(self, fig: plt.Figure, name: str) -> None:
        fig.savefig(self.output_dir / f"{name}.png", dpi=self.dpi, bbox_inches="tight")
        fig.savefig(self.output_dir / f"{name}.pdf", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_tsne_domain(self, X_source, X_target, X_source_adapted, X_target_adapted, name: str = "viz_02_tsne"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (Xs, Xt), title in zip(
            axes,
            [(X_source, X_target), (X_source_adapted, X_target_adapted)],
            ["域适应前", "域适应后（CORAL）"],
        ):
            combined = np.vstack([Xs, Xt])
            labels = np.array([0] * len(Xs) + [1] * len(Xt))
            embedded = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined) - 1)).fit_transform(combined)
            ax.scatter(embedded[labels == 0, 0], embedded[labels == 0, 1], c="#2196F3", alpha=0.6, label="源域")
            ax.scatter(embedded[labels == 1, 0], embedded[labels == 1, 1], c="#FF5722", alpha=0.6, label="目标域")
            ax.set_title(title)
            ax.legend()
        self._save(fig, name)

    def plot_corr_heatmap(self, df, name: str = "viz_03_corr"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
        self._save(fig, name)
