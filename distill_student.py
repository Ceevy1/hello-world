#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from paper_generalization.common import TARGET_COL, infer_features, load_dataset, regression_metrics
from paper_generalization.models import LightDynamicFusion, StudentRegressor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="outputs/student_model.pth")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--alpha", type=float, default=0.7)
    args = p.parse_args()

    tckpt = torch.load(args.teacher, map_location="cpu")
    df = load_dataset(args.data)
    feats = tckpt.get("feature_cols", infer_features(df))
    X = df[feats].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    mean = np.array(tckpt["scaler_mean"], dtype=np.float32)
    scale = np.array(tckpt["scaler_scale"], dtype=np.float32)
    X = (X - mean) / np.where(scale == 0, 1.0, scale)

    teacher = LightDynamicFusion(
        len(tckpt["modal_splits"]["perf"]),
        len(tckpt["modal_splits"]["behav"]),
        len(tckpt["modal_splits"]["eng"]),
        hidden=tckpt["hidden"],
    )
    teacher.load_state_dict(tckpt["state_dict"])
    teacher.eval()

    idx = {k: [feats.index(c) for c in v] for k, v in tckpt["modal_splits"].items()}
    with torch.no_grad():
        soft, _ = teacher(
            torch.tensor(X[:, idx["perf"]]),
            torch.tensor(X[:, idx["behav"]]),
            torch.tensor(X[:, idx["eng"]]),
        )
        soft = soft.numpy()

    student = StudentRegressor(input_dim=X.shape[1], hidden=16)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()

    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    st = torch.tensor(soft)
    for _ in range(args.epochs):
        pred = student(Xt)
        loss = args.alpha * mse(pred, st) + (1 - args.alpha) * mse(pred, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = student(Xt).numpy()
    metrics = regression_metrics(y, pred)
    torch.save({"state_dict": student.state_dict(), "feature_cols": feats, "metrics_train": metrics, "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist()}, args.out)
    pd.DataFrame([metrics]).to_csv(args.out.replace(".pth", "_metrics.csv"), index=False)
    print("student metrics:", metrics)


if __name__ == "__main__":
    main()
