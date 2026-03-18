from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_loader import extract_time_series, load_oulad_data, make_split_loaders
from experiment.ablation.ablation_models import FullFusionModel, LSTMOnlyModel, NoEntropyModel, StaticFusionModel
from experiment.evaluation.metrics import evaluate
from feature_engineering import extract_features
from model import DynamicFusionEnhanced
from trainer import TrainConfig, export_predictions, fit

WINDOWS = {"4w": 4, "8w": 8, "full": 16}
BASELINES = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
    "GaussianNB": GaussianNB(),
}


def _print_block(title: str) -> None:
    line = "=" * 24
    print(f"\n{line} {title} {line}")



def _print_table(title: str, df: pd.DataFrame, sort_by: list[str] | None = None) -> None:
    _print_block(title)
    view = df.copy()
    if sort_by:
        existing = [c for c in sort_by if c in view.columns]
        if existing:
            asc = [False] * len(existing)
            view = view.sort_values(existing, ascending=asc)
    print(view.to_string(index=False))



def _window_seq(x_seq: np.ndarray, week: int, max_weeks: int = 16) -> tuple[np.ndarray, np.ndarray]:
    out = np.copy(x_seq)
    if week < max_weeks:
        out[:, week:, :] = 0.0
    week_idx = np.full(len(x_seq), min(week - 1, max_weeks - 1), dtype=np.int64)
    return out, week_idx



def _predict_proba(clf, x: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x)[:, 1]
    score = clf.decision_function(x)
    return 1.0 / (1.0 + np.exp(-score))



def _run_dynamicfusion(
    x_seq: np.ndarray,
    x_stat: np.ndarray,
    y: np.ndarray,
    week_idx: np.ndarray,
    student_ids: np.ndarray,
    out_dir: Path,
    seed: int,
    epochs: int,
) -> tuple[dict[str, float], np.ndarray]:
    node_idx = np.arange(len(y), dtype=np.int64)
    tr, va, te, graph = make_split_loaders(x_seq, x_stat, node_idx, week_idx, y, random_state=seed)
    model = DynamicFusionEnhanced(seq_input_dim=x_seq.shape[-1], stat_input_dim=x_stat.shape[-1], graph_input_dim=16)
    fit(model, tr, va, graph, TrainConfig(epochs=epochs), output_dir=str(out_dir))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(splitter.split(np.zeros(len(y)), y))
    metrics = export_predictions(model, te, graph, output_dir=str(out_dir), student_ids=student_ids[test_idx])
    pred = pd.read_csv(out_dir / "predictions.csv")
    return metrics, pred["y_pred_prob"].to_numpy()



def run_main_comparison(
    x_seq: np.ndarray,
    x_stat: np.ndarray,
    y: np.ndarray,
    student_ids: np.ndarray,
    out: Path,
    seed: int,
    epochs: int,
) -> pd.DataFrame:
    rows = []
    for w_name, w in WINDOWS.items():
        xw, week_idx = _window_seq(x_seq, w)
        x_tab = extract_features(xw, x_stat)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, te_idx = next(sss.split(x_tab, y))
        x_tr, x_te = x_tab[tr_idx], x_tab[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        _print_block(f"Window: {w_name}")
        for name, clf in BASELINES.items():
            clf.fit(x_tr, y_tr)
            prob = _predict_proba(clf, x_te)
            met = evaluate(y_te, prob)
            met["AUPRC"] = float(average_precision_score(y_te, prob))
            rows.append({"scenario": w_name, "model": name, **met})
            print(
                f"[Baseline] {name:<18} "
                f"Acc={met['Accuracy']:.4f} AUC={met['AUC']:.4f} "
                f"Precision={met['Precision']:.4f} Recall={met['Recall']:.4f} F1={met['F1']:.4f} AUPRC={met['AUPRC']:.4f}"
            )

        df_dir = out / f"dynamic_{w_name}"
        df_dir.mkdir(parents=True, exist_ok=True)
        dm, prob = _run_dynamicfusion(xw, x_stat, y, week_idx, student_ids, df_dir, seed, epochs)
        dm["AUPRC"] = float(average_precision_score(y[te_idx], prob))
        rows.append({"scenario": w_name, "model": "DynamicFusion-Enhanced", **dm})
        print(
            f"[DynamicFusion-Enhanced] Acc={dm['Accuracy']:.4f} AUC={dm['AUC']:.4f} "
            f"Precision={dm['Precision']:.4f} Recall={dm['Recall']:.4f} F1={dm['F1']:.4f} AUPRC={dm['AUPRC']:.4f}"
        )

        window_df = pd.DataFrame([r for r in rows if r["scenario"] == w_name])
        _print_table(f"Window {w_name} Performance Ranking", window_df, sort_by=["AUC", "Accuracy", "F1"])

    df = pd.DataFrame(rows)
    df.to_csv(out / "performance_comparison_full.csv", index=False)
    _print_table("Full Performance Comparison Table", df, sort_by=["scenario", "AUC", "Accuracy"])
    print("\n[Summary Pivot] Full performance table:")
    print(df.pivot_table(index="model", columns="scenario", values=["Accuracy", "AUC", "F1"]).to_string())
    return df



def run_lomo_ablation(x_seq: np.ndarray, x_stat: np.ndarray, y: np.ndarray, modules: np.ndarray, out: Path, seed: int) -> pd.DataFrame:
    x_tab = extract_features(x_seq, x_stat)
    models = {
        "Full": FullFusionModel,
        "NoFusion": StaticFusionModel,
        "NoEntropy": NoEntropyModel,
        "LSTM-only": LSTMOnlyModel,
    }
    rows = []
    for mod in np.unique(modules):
        tr = modules != mod
        te = modules == mod
        if tr.sum() < 30 or te.sum() < 10 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        _print_block(f"LOMO Module: {mod}")
        for name, cls in models.items():
            m = cls()
            x_tr, x_te = x_tab[tr], x_tab[te]
            if name == "NoEntropy":
                x_tr = NoEntropyModel.remove_entropy(x_tr)
                x_te = NoEntropyModel.remove_entropy(x_te)
            m.fit(x_tr, y[tr])
            prob = m.predict_proba(x_te)
            met = evaluate(y[te], prob)
            rows.append({"module": mod, "model": name, **met})
            print(
                f"[Ablation] {name:<10} "
                f"Acc={met['Accuracy']:.4f} AUC={met['AUC']:.4f} "
                f"Precision={met['Precision']:.4f} Recall={met['Recall']:.4f} F1={met['F1']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out / "lomo_ablation_results.csv", index=False)
    if not df.empty:
        _print_table("LOMO Ablation Detailed Results", df, sort_by=["AUC", "Accuracy", "F1"])
    return df



def run_shap_analysis(x_seq: np.ndarray, x_stat: np.ndarray, y: np.ndarray, out: Path, seed: int) -> pd.DataFrame:
    x_tab = extract_features(x_seq, x_stat)
    feature_names = [f"f{i}" for i in range(x_tab.shape[1])]
    model = RandomForestClassifier(n_estimators=300, random_state=seed)
    model.fit(x_tab, y)

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_tab)
        if isinstance(shap_values, list):
            raw = np.asarray(shap_values[-1])
        else:
            raw = np.asarray(shap_values)
        while raw.ndim > 2:
            raw = raw.mean(axis=0)
        vals = np.abs(raw).mean(axis=0)
    except Exception:
        vals = np.asarray(model.feature_importances_)

    vals = np.asarray(vals).reshape(-1)
    if len(vals) < len(feature_names):
        vals = np.pad(vals, (0, len(feature_names) - len(vals)), constant_values=0.0)
    if len(vals) > len(feature_names):
        vals = vals[: len(feature_names)]
    df = pd.DataFrame({"feature": feature_names, "SHAP_value": vals}).sort_values("SHAP_value", ascending=False)
    df.to_csv(out / "shap_importance.csv", index=False)

    top = df.head(15)
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"][::-1], top["SHAP_value"][::-1])
    plt.title("Top SHAP Importance")
    plt.tight_layout()
    plt.savefig(out / "shap_importance.png", dpi=220)
    plt.close()

    _print_table("Top SHAP Features", df.head(10))
    return df



def main() -> None:
    parser = argparse.ArgumentParser(description="Unified experiment launcher: comparison + early window + LOMO ablation + SHAP")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/unified_experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[run_experiments] data_dir={Path(args.data_dir).resolve()}")
    print(f"[run_experiments] output_dir={out.resolve()}")
    print(f"[run_experiments] seed={args.seed}, epochs={args.epochs}")

    tables = load_oulad_data(args.data_dir)
    x_seq, x_stat, _, _, y, modules, student_ids = extract_time_series(tables, max_weeks=16)
    print(f"[run_experiments] dataset seq={x_seq.shape}, stat={x_stat.shape}, n={len(y)}, modules={len(np.unique(modules))}")

    perf_df = run_main_comparison(x_seq, x_stat, y, student_ids, out, args.seed, args.epochs)
    early_df = perf_df[perf_df["scenario"].isin(["4w", "8w"])].copy()
    early_df.to_csv(out / "early_window_performance.csv", index=False)
    _print_table("Early Window Comparison", early_df, sort_by=["scenario", "AUC", "Accuracy"])
    print("\n[Early Window Pivot]")
    print(early_df.pivot_table(index="model", columns="scenario", values=["Accuracy", "AUC", "F1"]).to_string())

    lomo_df = run_lomo_ablation(x_seq, x_stat, y, modules, out, args.seed)
    if not lomo_df.empty:
        lomo_summary = lomo_df.groupby("model", as_index=False)[["Accuracy", "AUC", "F1"]].mean().sort_values("AUC", ascending=False)
        _print_table("LOMO Ablation Summary", lomo_summary, sort_by=["AUC", "Accuracy", "F1"])

    run_shap_analysis(x_seq, x_stat, y, out, args.seed)

    latex = perf_df.pivot_table(index="model", columns="scenario", values=["Accuracy", "AUC", "F1"]).to_latex(float_format="%.4f")
    (out / "performance_comparison.tex").write_text(latex, encoding="utf-8")

    print(f"\nAll experiments completed. Results saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
