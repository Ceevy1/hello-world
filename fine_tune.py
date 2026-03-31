#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from paper_generalization.common import TARGET_COL, bootstrap_ci, kfold, load_dataset, regression_metrics
from paper_generalization.models import LightDynamicFusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--freeze_encoder", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--out", default="outputs/model_ft.pth")
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    feat_cols = ckpt["feature_cols"]
    modal = ckpt["modal_splits"]

    df = load_dataset(args.data)
    X = df[feat_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
    X = (X - mean) / np.where(scale == 0, 1.0, scale)

    idx = {k: [feat_cols.index(c) for c in v] for k, v in modal.items()}
    fold_rows = []
    all_preds = np.zeros_like(y)

    for fold, (tr, va) in enumerate(kfold(len(y), args.cv).split(X), start=1):
        model = LightDynamicFusion(len(idx["perf"]), len(idx["behav"]), len(idx["eng"]), hidden=ckpt["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        model.freeze_encoders(args.freeze_encoder)

        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        tr_ds = TensorDataset(
            torch.tensor(X[tr][:, idx["perf"]]),
            torch.tensor(X[tr][:, idx["behav"]]),
            torch.tensor(X[tr][:, idx["eng"]]),
            torch.tensor(y[tr]),
        )
        va_tensors = (
            torch.tensor(X[va][:, idx["perf"]]),
            torch.tensor(X[va][:, idx["behav"]]),
            torch.tensor(X[va][:, idx["eng"]]),
        )
        tr_dl = DataLoader(tr_ds, batch_size=min(16, len(tr_ds)), shuffle=True)

        best_pred = None
        best = float("inf")
        for _ in range(args.epochs):
            model.train()
            for bp, bb, be, by in tr_dl:
                pred, _ = model(bp, bb, be)
                loss = loss_fn(pred, by)
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                pred, _ = model(*va_tensors)
                rmse = np.sqrt(np.mean((pred.numpy() - y[va]) ** 2))
                if rmse < best:
                    best = rmse
                    best_pred = pred.numpy()

        all_preds[va] = best_pred
        m = regression_metrics(y[va], best_pred)
        m["fold"] = fold
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "RMSE_mean": float(fold_df["RMSE"].mean()),
        "RMSE_std": float(fold_df["RMSE"].std()),
        "MAE_mean": float(fold_df["MAE"].mean()),
        "MAE_std": float(fold_df["MAE"].std()),
        "R2_mean": float(fold_df["R2"].mean()),
        "R2_std": float(fold_df["R2"].std()),
    }

    boot = []
    rng = np.random.default_rng(42)
    errors = (all_preds - y) ** 2
    for _ in range(1000):
        s = rng.choice(errors, len(errors), replace=True)
        boot.append(float(np.sqrt(np.mean(s))))
    summary["RMSE_CI95"] = bootstrap_ci(np.array(boot))

    save_payload = {**ckpt, "fine_tuned": True, "predictions": all_preds.tolist(), "summary": summary}
    torch.save(save_payload, args.out)
    fold_out = args.out.replace(".pth", "_cv_metrics.csv")
    fold_df.to_csv(fold_out, index=False)
    pd.DataFrame([summary]).to_csv(args.out.replace(".pth", "_summary.csv"), index=False)

    print("Fine-tune summary:", summary)
    print(f"Fold metrics -> {fold_out}")


if __name__ == "__main__":
    main()
