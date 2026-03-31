#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def cohens_d(x, y):
    diff = np.array(x) - np.array(y)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="csv with fold-wise RMSE column")
    p.add_argument("--candidate", required=True, help="csv with fold-wise RMSE column")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    b = pd.read_csv(args.baseline)
    c = pd.read_csv(args.candidate)

    n = min(len(b), len(c))
    x = b["RMSE"].values[:n]
    y = c["RMSE"].values[:n]

    t_stat, t_p = ttest_rel(x, y)
    try:
        w_stat, w_p = wilcoxon(x, y)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    out = pd.DataFrame(
        [
            {
                "n_folds": n,
                "baseline_rmse_mean": x.mean(),
                "candidate_rmse_mean": y.mean(),
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
                "cohens_d": cohens_d(y, x),
            }
        ]
    )
    out.to_csv(args.out, index=False)
    print(out)


if __name__ == "__main__":
    main()
