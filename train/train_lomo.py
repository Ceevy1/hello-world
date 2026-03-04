from __future__ import annotations

from typing import Dict

import numpy as np

from preprocess.data_builder import split_lomo
from train.train_full import train_full_pipeline


def run_lomo(
    x_seq: np.ndarray,
    x_tab: np.ndarray,
    y: np.ndarray,
    modules: np.ndarray,
) -> Dict[str, float]:
    rmses = {}
    for m in np.unique(modules):
        split = split_lomo(modules, m)
        out = train_full_pipeline(
            x_seq[split.train_idx], x_tab[split.train_idx], y[split.train_idx],
            x_seq[split.test_idx], x_tab[split.test_idx], y[split.test_idx],
        )
        rmses[str(m)] = out.metrics["HAFM"]["RMSE"]
    values = np.array(list(rmses.values()), dtype=float)
    rmses["mean"] = float(values.mean())
    rmses["std"] = float(values.std())
    return rmses
