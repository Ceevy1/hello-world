from __future__ import annotations

import numpy as np
import shap
import torch


def run_shap_analysis(model, exercise: np.ndarray, lab: np.ndarray, static: np.ndarray, max_samples: int = 128):
    model.eval()
    n = min(len(exercise), max_samples)
    ex = torch.tensor(exercise[:n], dtype=torch.float32)
    lb = torch.tensor(lab[:n], dtype=torch.float32)
    st = torch.tensor(static[:n], dtype=torch.float32)

    def predict_fn(static_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.tensor(static_np, dtype=torch.float32)
            pred = model(ex[: len(s)], lb[: len(s)], s)
            return pred.detach().cpu().numpy()

    explainer = shap.Explainer(predict_fn, static[:n])
    shap_values = explainer(static[:n])
    shap.summary_plot(shap_values, static[:n], show=False)
    return shap_values
