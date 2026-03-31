#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from paper_generalization.common import TARGET_COL, build_modal_splits, infer_features, load_dataset, make_scaler, save_json
from paper_generalization.models import LightDynamicFusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="OULAD")
    p.add_argument("--source_csv", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--save", default="outputs/model_oulad.pth")
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    df = load_dataset(args.source_csv)
    feat_cols = infer_features(df)
    y = df[TARGET_COL].values.astype(np.float32)
    X = df[feat_cols].values.astype(np.float32)
    scaler = make_scaler(X)
    X = scaler.transform(X).astype(np.float32)

    modal = build_modal_splits(feat_cols)
    idx = {k: [feat_cols.index(c) for c in v] for k, v in modal.items()}

    x_perf = torch.tensor(X[:, idx["perf"]])
    x_behav = torch.tensor(X[:, idx["behav"]])
    x_eng = torch.tensor(X[:, idx["eng"]])
    yt = torch.tensor(y)

    ds = TensorDataset(x_perf, x_behav, x_eng, yt)
    dl = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True)

    model = LightDynamicFusion(len(idx["perf"]), len(idx["behav"]), len(idx["eng"]), hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for _ in range(args.epochs):
        model.train()
        for bp, bb, be, by in dl:
            pred, _ = model(bp, bb, be)
            loss = loss_fn(pred, by)
            opt.zero_grad()
            loss.backward()
            opt.step()

    ckpt = {
        "domain": args.domain,
        "state_dict": model.state_dict(),
        "feature_cols": feat_cols,
        "modal_splits": modal,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "hidden": args.hidden,
    }
    torch.save(ckpt, args.save)
    save_json({"domain": args.domain, "features": feat_cols, "modal_splits": modal}, args.save + ".meta.json")
    print(f"saved pretrained model to {args.save}")


if __name__ == "__main__":
    main()
