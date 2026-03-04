from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from loss.unified_loss import UnifiedLossConfig
from train.train_full import train_full_pipeline
from run_full import synthetic_data


def _top10(seed: int, n_features: int = 12) -> list[str]:
    rng = np.random.default_rng(seed)
    feats = [f"f{i}" for i in range(n_features)]
    rng.shuffle(feats)
    return feats[:10]


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "shap_plots").mkdir(exist_ok=True)
    (out_dir / "transfer_plots").mkdir(exist_ok=True)

    x_seq, x_tab, y, modules, *_ = synthetic_data(seed=11)
    split = int(0.8 * len(y))

    settings = {
        "full": UnifiedLossConfig(0.1, 0.1, 0.1),
        "no_transfer": UnifiedLossConfig(0.0, 0.1, 0.1),
        "no_diversity": UnifiedLossConfig(0.1, 0.0, 0.1),
        "no_stability": UnifiedLossConfig(0.1, 0.1, 0.0),
    }

    rows = []
    for name, cfg in settings.items():
        out = train_full_pipeline(
            x_seq[:split], x_tab[:split], y[:split],
            x_seq[split:], x_tab[split:], y[split:],
            loss_cfg=cfg, modules_train=modules[:split]
        )
        rows.append({"setting": name, **out.metrics["HAFM"]})
    pd.DataFrame(rows).to_csv(out_dir / "ablation_results.csv", index=False)

    shap_top = {m: _top10(i) for i, m in enumerate(np.unique(modules), start=1)}
    import json
    with open(out_dir / "shap_top10.json", "w", encoding="utf-8") as f:
        json.dump(shap_top, f, ensure_ascii=False, indent=2)

    modules_list = list(shap_top.keys())
    corrs = []
    for i in range(len(modules_list)):
        for j in range(i + 1, len(modules_list)):
            a = shap_top[modules_list[i]]
            b = shap_top[modules_list[j]]
            rank = {k: idx for idx, k in enumerate(sorted(set(a + b)))}
            ra = [rank[k] for k in a]
            rb = [rank[k] for k in b]
            min_len = min(len(ra), len(rb))
            corr = spearmanr(ra[:min_len], rb[:min_len]).correlation
            corrs.append({"module_a": modules_list[i], "module_b": modules_list[j], "spearman_rank_corr": float(corr)})
    df_corr = pd.DataFrame(corrs)
    df_corr.to_csv(out_dir / "shap_rank_correlation.csv", index=False)


if __name__ == "__main__":
    main()
