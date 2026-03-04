from __future__ import annotations

from pathlib import Path

import pandas as pd

from loss.unified_loss import lambda_grid_search_space
from run_full import synthetic_data
from train.train_full import train_full_pipeline


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    x_seq, x_tab, y, modules, *_ = synthetic_data(seed=21)
    split = int(0.8 * len(y))

    rows = []
    grid = lambda_grid_search_space([0, 0.01, 0.1, 1], [0, 0.01, 0.1], [0, 0.01, 0.1])
    for cfg in grid:
        out = train_full_pipeline(
            x_seq[:split], x_tab[:split], y[:split],
            x_seq[split:], x_tab[split:], y[split:],
            loss_cfg=cfg, modules_train=modules[:split]
        )
        rows.append({
            "lambda_transfer": cfg.lambda_transfer,
            "lambda_diversity": cfg.lambda_diversity,
            "lambda_stability": cfg.lambda_stability,
            "RMSE": out.metrics["HAFM"]["RMSE"],
        })

    pd.DataFrame(rows).to_csv(out_dir / "lambda_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
