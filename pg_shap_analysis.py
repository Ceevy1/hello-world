#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from paper_generalization.common import TARGET_COL, infer_features, load_dataset
from paper_generalization.models import StudentRegressor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="output directory")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model, map_location="cpu")
    df = load_dataset(args.data)
    feats = ckpt.get("feature_cols", infer_features(df))
    X = df[feats].values.astype(np.float32)

    mean = np.array(ckpt.get("scaler_mean", np.zeros(len(feats))), dtype=np.float32)
    scale = np.array(ckpt.get("scaler_scale", np.ones(len(feats))), dtype=np.float32)
    X = (X - mean) / np.where(scale == 0, 1.0, scale)

    model = StudentRegressor(input_dim=X.shape[1])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    background = X[: min(20, len(X))]

    def pred_fn(x):
        with torch.no_grad():
            return model(torch.tensor(x.astype(np.float32))).numpy()

    explainer = shap.KernelExplainer(pred_fn, background)
    sv = explainer.shap_values(X, nsamples=min(100, len(X) * 20))
    shap_arr = np.array(sv)
    if shap_arr.ndim == 1:
        shap_arr = shap_arr[:, None]

    imp = np.mean(np.abs(shap_arr), axis=0)
    imp_df = pd.DataFrame({"feature": feats, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(out_dir / "shap_importance.csv", index=False)

    shap.summary_plot(shap_arr, X, feature_names=feats, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=200)
    plt.close()
    print(f"saved SHAP outputs to {out_dir}")


if __name__ == "__main__":
    main()
