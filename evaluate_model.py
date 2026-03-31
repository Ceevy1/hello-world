#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from paper_generalization.common import TARGET_COL, bootstrap_ci, load_dataset, regression_metrics
from paper_generalization.models import LightDynamicFusion, StudentRegressor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--metrics", nargs="+", default=["RMSE", "MAE", "R2"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    df = load_dataset(args.data)
    feats = ckpt["feature_cols"]
    X = df[feats].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    mean = np.array(ckpt.get("scaler_mean", np.zeros(len(feats))), dtype=np.float32)
    scale = np.array(ckpt.get("scaler_scale", np.ones(len(feats))), dtype=np.float32)
    X = (X - mean) / np.where(scale == 0, 1.0, scale)

    if "modal_splits" in ckpt:
        idx = {k: [feats.index(c) for c in v] for k, v in ckpt["modal_splits"].items()}
        model = LightDynamicFusion(len(idx["perf"]), len(idx["behav"]), len(idx["eng"]), hidden=ckpt["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            pred, w = model(
                torch.tensor(X[:, idx["perf"]]),
                torch.tensor(X[:, idx["behav"]]),
                torch.tensor(X[:, idx["eng"]]),
            )
            pred = pred.numpy()
            attn = w.numpy()
    else:
        model = StudentRegressor(input_dim=X.shape[1])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X)).numpy()
            attn = None

    m = regression_metrics(y, pred)
    boot = []
    rng = np.random.default_rng(42)
    err = (pred - y) ** 2
    for _ in range(1000):
        boot.append(np.sqrt(np.mean(rng.choice(err, len(err), replace=True))))
    m["RMSE_CI95"] = bootstrap_ci(np.array(boot))

    out_df = pd.DataFrame([m])
    out_df.to_csv(args.out, index=False)
    pred_out = args.out.replace(".csv", "_predictions.csv")
    d = pd.DataFrame({"y_true": y, "y_pred": pred})
    if attn is not None:
        d[["w_perf", "w_behav", "w_eng"]] = attn
    d.to_csv(pred_out, index=False)
    print(out_df)


if __name__ == "__main__":
    main()
