#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from paper_generalization.common import TARGET_COL, infer_features, load_dataset


def mixup(X, y, alpha=0.2, repeats=1, seed=42):
    rng = np.random.default_rng(seed)
    out_x, out_y = [X], [y]
    for _ in range(repeats * len(X)):
        i, j = rng.integers(0, len(X), size=2)
        lam = rng.beta(alpha, alpha)
        out_x.append(lam * X[i] + (1 - lam) * X[j])
        out_y.append(lam * y[i] + (1 - lam) * y[j])
    return np.vstack(out_x), np.hstack(out_y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--method", choices=["SMOTE", "ADASYN", "noise", "mixup"], default="mixup")
    p.add_argument("--target", required=True)
    p.add_argument("--noise_std", type=float, default=0.02)
    args = p.parse_args()

    df = load_dataset(args.input)
    feats = infer_features(df)
    X = df[feats].values.astype(float)
    y = df[TARGET_COL].values.astype(float)

    if args.method in {"SMOTE", "ADASYN"}:
        y_bins = pd.qcut(y, q=min(3, len(np.unique(y))), duplicates="drop").codes
        try:
            if args.method == "SMOTE":
                from imblearn.over_sampling import SMOTE

                sampler = SMOTE(random_state=42, k_neighbors=min(2, len(X) - 1))
            else:
                from imblearn.over_sampling import ADASYN

                sampler = ADASYN(random_state=42, n_neighbors=min(2, len(X) - 1))
            X_res, y_bin_res = sampler.fit_resample(X, y_bins)
            bin_to_mean = {b: y[y_bins == b].mean() for b in np.unique(y_bins)}
            y_res = np.array([bin_to_mean[b] for b in y_bin_res])
        except Exception:
            X_res, y_res = mixup(X, y, repeats=2)
    elif args.method == "noise":
        rng = np.random.default_rng(42)
        noise = rng.normal(0, args.noise_std, size=X.shape) * X.std(axis=0, keepdims=True)
        X_res = np.vstack([X, X + noise])
        y_res = np.hstack([y, y])
    else:
        X_res, y_res = mixup(X, y, repeats=2)

    out_df = pd.DataFrame(X_res, columns=feats)
    out_df[TARGET_COL] = y_res
    out_df.to_csv(args.target, index=False)
    print(f"augmented {len(df)} -> {len(out_df)} rows, saved to {args.target}")


if __name__ == "__main__":
    main()
