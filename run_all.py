"""MT-HAFNet runnable entrypoint (synthetic smoke pipeline)."""
from __future__ import annotations

import numpy as np

from evaluation.statistics import significance_tests
from preprocess.data_builder import truncate_sequence
from train.train_baselines import run_baseline_benchmark
from train.train_full import train_full_pipeline
from train.train_lomo import run_lomo


def _synthetic_data(n: int = 256, t: int = 30, d: int = 4, f: int = 10):
    rng = np.random.default_rng(42)
    x_seq = rng.normal(size=(n, t, d)).astype(np.float32)
    x_tab = rng.normal(size=(n, f)).astype(np.float32)
    signal = x_tab[:, 0] * 5 + x_seq[:, :, 0].mean(axis=1) * 8
    y = (60 + signal + rng.normal(scale=6, size=n)).clip(0, 100).astype(np.float32)
    modules = rng.choice(["AAA", "BBB", "CCC", "DDD"], size=n)
    return x_seq, x_tab, y, modules


def main() -> None:
    x_seq, x_tab, y, modules = _synthetic_data()
    split = int(0.8 * len(y))

    baseline_metrics = run_baseline_benchmark(x_tab[:split], y[:split], x_tab[split:], y[split:])
    print("[Baselines]", {k: round(v["RMSE"], 4) for k, v in baseline_metrics.items()})

    out = train_full_pipeline(x_seq[:split], x_tab[:split], y[:split], x_seq[split:], x_tab[split:], y[split:])
    print("[Full]", out.metrics)

    x4 = truncate_sequence(x_seq, 4)
    out4 = train_full_pipeline(x4[:split], x_tab[:split], y[:split], x4[split:], x_tab[split:], y[split:])
    print("[Week4 RMSE]", out4.metrics["HAFM"]["RMSE"])

    lomo = run_lomo(x_seq, x_tab, y, modules)
    print("[LOMO]", lomo)

    sig = significance_tests(y[split:], out.predictions["HAFM"], out.predictions["XGBoost"])
    print("[Significance HAFM vs XGB]", sig)


if __name__ == "__main__":
    main()
