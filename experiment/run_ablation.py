from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from data_loader import extract_time_series, load_oulad_data
from experiment.ablation.ablation_models import FullFusionModel, LSTMOnlyModel, NoEntropyModel, StaticFusionModel
from experiment.evaluation.metrics import evaluate
from experiment.evaluation.significance_test import paired_t_test
from experiment.results.latex_generator import to_latex
from experiment.results.save_results import save_results
from feature_engineering import extract_features


def run_ablation(x_tab: np.ndarray, y: np.ndarray, seeds: list[int]) -> pd.DataFrame:
    exps = {
        "Full": FullFusionModel,
        "NoFusion": StaticFusionModel,
        "NoEntropy": NoEntropyModel,
        "LSTM-only": LSTMOnlyModel,
    }
    rows = []

    for sd in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=sd)
        tr, te = next(sss.split(x_tab, y))
        x_tr, x_te = x_tab[tr], x_tab[te]
        y_tr, y_te = y[tr], y[te]

        for name, cls in exps.items():
            model = cls()
            if name == "NoEntropy":
                x_tr_use = NoEntropyModel.remove_entropy(x_tr)
                x_te_use = NoEntropyModel.remove_entropy(x_te)
            else:
                x_tr_use, x_te_use = x_tr, x_te
            model.fit(x_tr_use, y_tr)
            prob = model.predict_proba(x_te_use)
            met = evaluate(y_te, prob)
            rows.append({"seed": sd, "model": name, **met})
            print(f"[Ablation][seed={sd}] {name}: Acc={met['Accuracy']:.4f} AUC={met['AUC']:.4f} F1={met['F1']:.4f}")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/experiment")
    parser.add_argument("--seeds", default="42,52,62,72,82")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    x_seq, x_stat, _, _, y, _, _ = extract_time_series(load_oulad_data(args.data_dir), max_weeks=16)
    x_tab = extract_features(x_seq, x_stat)

    df = run_ablation(x_tab, y, seeds)
    save_results(df, str(out / "ablation_results.csv"))

    summary = df.groupby("model", as_index=False)[["Accuracy", "AUC", "F1"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([c for c in col if c]).strip("_") for col in summary.columns]
    summary.to_csv(out / "ablation_summary.csv", index=False)

    # t-test Full vs others on accuracy
    full = df[df["model"] == "Full"].sort_values("seed")
    tests = []
    for m in ["NoFusion", "NoEntropy", "LSTM-only"]:
        oth = df[df["model"] == m].sort_values("seed")
        t = paired_t_test(full["Accuracy"].to_numpy(), oth["Accuracy"].to_numpy())
        tests.append({"Model": "Full", "Baseline": m, **t})
        print(f"[Ablation-TTest] Full vs {m}: p={t['p_value']:.6f}, significant={t['significant']}")
    pd.DataFrame(tests).to_csv(out / "ablation_ttest.csv", index=False)

    # plot
    agg = df.groupby("model", as_index=False)["AUC"].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(agg["model"], agg["AUC"])
    plt.title("Ablation AUC Comparison")
    plt.tight_layout()
    plt.savefig(out / "ablation_comparison.png", dpi=220)
    plt.close()

    (out / "ablation_table.tex").write_text(to_latex(df.groupby("model", as_index=False)[["Accuracy", "AUC", "F1"]].mean()), encoding="utf-8")


if __name__ == "__main__":
    main()
