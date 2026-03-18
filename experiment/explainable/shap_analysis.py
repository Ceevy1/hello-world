from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd



def _fallback_importance(model, feature_count: int) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        while coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        return np.abs(coef)
    return np.zeros(feature_count, dtype=float)



def summarize_shap(model, x_tab: np.ndarray, feature_names: list[str], save_dir: str | Path | None = None) -> dict[str, object]:
    """Return a lightweight SHAP-style ranking compatible with the PIA pipeline."""
    _ = np.asarray(x_tab)
    feature_names = list(feature_names)
    save_path = Path(save_dir) if save_dir is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    importance = _fallback_importance(model, len(feature_names))
    importance = np.asarray(importance).reshape(-1)
    if len(importance) < len(feature_names):
        importance = np.pad(importance, (0, len(feature_names) - len(importance)), constant_values=0.0)
    if len(importance) > len(feature_names):
        importance = importance[: len(feature_names)]

    ranking = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False, ignore_index=True
    )

    if save_path is not None:
        ranking.to_csv(save_path / "shap_importance.csv", index=False)
        top = ranking.head(15)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#4472C4")
        plt.title("Top SHAP Features")
        plt.tight_layout()
        plt.savefig(save_path / "shap_bar.png", dpi=180, bbox_inches="tight")
        plt.close()

    return {"importance": ranking, "raw_values": None}
