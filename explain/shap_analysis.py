from __future__ import annotations

from pathlib import Path

import numpy as np
import shap


def run_tree_shap(model, x_tab: np.ndarray, save_dir: str = "outputs"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_tab)
    shap.summary_plot(shap_values, x_tab, show=False)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary.png", dpi=300)
    plt.close()

    shap.summary_plot(shap_values, x_tab, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_bar.png", dpi=300)
    plt.close()
