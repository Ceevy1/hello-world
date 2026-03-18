from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_loader import extract_time_series, load_junyi_data, load_oulad_data, make_split_loaders
from experiment.ablation.ablation_models import FullFusionModel, LSTMOnlyModel, NoEntropyModel, StaticFusionModel
from experiment.actionable.intervention import generate_intervention
from experiment.actionable.risk_stratification import stratify_predictions
from experiment.evaluation.intervention_metrics import intervention_success_rate
from experiment.evaluation.metrics import evaluate
from experiment.evaluation.predictive_metrics import predictive_metrics
from experiment.explainable.counterfactual import generate_counterfactual
from experiment.explainable.shap_analysis import summarize_shap
from experiment.model_interface import SklearnPIAModel
from experiment.predictive.confidence import compute_confidence
from experiment.predictive.stability import compute_stability
from experiment.predictive.temporal_prediction import slice_temporal_data
from feature_engineering import extract_features
from model import DynamicFusionEnhanced
from trainer import TrainConfig, export_predictions, fit

WINDOWS = {"4w": 4, "8w": 8, "full": 16}
PIA_WINDOWS = [4, 8, "full"]
DEFAULT_PIA_SEEDS = [42, 52, 62, 72, 82]
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



def _print_metric_line(prefix: str, metrics: dict[str, float], extra: dict[str, float] | None = None) -> None:
    payload = {
        "Accuracy": metrics.get("Accuracy", float("nan")),
        "AUC": metrics.get("AUC", float("nan")),
        "Precision": metrics.get("Precision", float("nan")),
        "Recall": metrics.get("Recall", float("nan")),
        "F1": metrics.get("F1", float("nan")),
    }
    if extra:
        payload.update(extra)
    formatted = " ".join(f"{key}={value:.4f}" for key, value in payload.items())
    print(f"{prefix} {formatted}")



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



def _train_test_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    return next(splitter.split(np.zeros(len(y)), y))



def _feature_names(x_stat: np.ndarray) -> list[str]:
    return [f"stat_{idx}" for idx in range(x_stat.shape[1])] + [
        "total_clicks",
        "avg_clicks",
        "click_std",
        "click_entropy",
        "active_week_ratio",
        "active_week_count",
    ]



def _build_pia_model(seed: int) -> SklearnPIAModel:
    estimator = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=3,
        random_state=seed,
    )
    return SklearnPIAModel(estimator=estimator)



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
    print(f"[DynamicFusion][train] seed={seed} window_dir={out_dir.name} epochs={epochs}")
    node_idx = np.arange(len(y), dtype=np.int64)
    tr, va, te, graph = make_split_loaders(x_seq, x_stat, node_idx, week_idx, y, random_state=seed)
    model = DynamicFusionEnhanced(seq_input_dim=x_seq.shape[-1], stat_input_dim=x_stat.shape[-1], graph_input_dim=16)
    fit(model, tr, va, graph, TrainConfig(epochs=epochs), output_dir=str(out_dir))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(splitter.split(np.zeros(len(y)), y))
    metrics = export_predictions(model, te, graph, output_dir=str(out_dir), student_ids=student_ids[test_idx])
    pred = pd.read_csv(out_dir / "predictions.csv")
    _print_metric_line("[DynamicFusion][test]", metrics)
    return metrics, pred["y_pred_prob"].to_numpy()



def run_main_comparison(
    x_seq: np.ndarray,
    x_stat: np.ndarray,
    y: np.ndarray,
    student_ids: np.ndarray,
    out: Path,
    seed: int,
    epochs: int,
    skip_dynamic: bool = False,
) -> pd.DataFrame:
    rows = []
    for w_name, w in WINDOWS.items():
        xw, week_idx = _window_seq(x_seq, w)
        x_tab = extract_features(xw, x_stat)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, te_idx = next(sss.split(x_tab, y))
        x_tr, x_te = x_tab[tr_idx], x_tab[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        _print_block(f"Main Comparison Window: {w_name}")
        for name, clf in BASELINES.items():
            model = clone(clf)
            model.fit(x_tr, y_tr)
            prob = _predict_proba(model, x_te)
            met = evaluate(y_te, prob)
            met["AUPRC"] = float(average_precision_score(y_te, prob))
            rows.append({"scenario": w_name, "model": name, **met})
            _print_metric_line(f"[Baseline][{w_name}][{name}]", met, {"AUPRC": met["AUPRC"]})

        if not skip_dynamic:
            df_dir = out / f"dynamic_{w_name}"
            df_dir.mkdir(parents=True, exist_ok=True)
            dm, prob = _run_dynamicfusion(xw, x_stat, y, week_idx, student_ids, df_dir, seed, epochs)
            dm["AUPRC"] = float(average_precision_score(y[te_idx], prob))
            rows.append({"scenario": w_name, "model": "DynamicFusion-Enhanced", **dm})
            _print_metric_line(f"[DynamicFusion][{w_name}]", dm, {"AUPRC": dm["AUPRC"]})

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
            _print_metric_line(f"[LOMO][{mod}][{name}]", met)

    df = pd.DataFrame(rows)
    df.to_csv(out / "lomo_ablation_results.csv", index=False)
    if not df.empty:
        _print_table("LOMO Ablation Detailed Results", df, sort_by=["AUC", "Accuracy", "F1"])
    return df



def run_shap_analysis(x_seq: np.ndarray, x_stat: np.ndarray, y: np.ndarray, out: Path, seed: int) -> pd.DataFrame:
    x_tab = extract_features(x_seq, x_stat)
    feature_names = _feature_names(x_stat)
    model = RandomForestClassifier(n_estimators=300, random_state=seed)
    model.fit(x_tab, y)

    shap_result = summarize_shap(model, x_tab, feature_names, save_dir=out / "shap")
    df = shap_result["importance"].rename(columns={"mean_abs_shap": "SHAP_value"})
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



def _run_pia_seed(
    x_seq: np.ndarray,
    x_stat: np.ndarray,
    y: np.ndarray,
    modules: np.ndarray,
    out_dir: Path,
    seed: int,
) -> dict[str, object]:
    feature_names = _feature_names(x_stat)
    train_idx, test_idx = _train_test_indices(y, seed)
    temporal_probs: dict[object, np.ndarray] = {}
    predictive_rows: list[dict[str, object]] = []
    intervention_rows: list[dict[str, object]] = []
    risk_tables: list[pd.DataFrame] = []

    x_full_all = None
    full_model: SklearnPIAModel | None = None
    full_x_test = None

    for week in PIA_WINDOWS:
        _, x_features, _ = slice_temporal_data(x_seq, x_stat, week)
        x_train = x_features[train_idx]
        x_test = x_features[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = _build_pia_model(seed)
        print(f"[PIA][train] seed={seed} window={week} samples={len(x_train)}")
        model.fit(x_train, y_train)
        probs = model.predict(x_test)
        temporal_probs[week] = probs

        metrics = evaluate(y_test, probs)
        confidence = compute_confidence(probs)
        predictive_rows.append(
            {
                "seed": seed,
                "window": week,
                **metrics,
                "Confidence": float(np.mean(confidence)),
            }
        )
        _print_metric_line(
            f"[PIA][eval] seed={seed} window={week}",
            metrics,
            {"Confidence": float(np.mean(confidence))},
        )

        if week == "full":
            x_full_all = x_features
            full_model = model
            full_x_test = x_test
            risk_df = stratify_predictions(probs, y_test)
            risk_df.insert(0, "seed", seed)
            risk_tables.append(risk_df)
            print(f"[PIA][risk] seed={seed}\n{risk_df.to_string(index=False)}")

            shap_dir = out_dir / f"seed_{seed}" / "pia_shap"
            shap_result = model.explain(x_test, feature_names, save_dir=shap_dir)
            top_shap = shap_result["importance"].head(10).copy()
            top_shap.insert(0, "seed", seed)
            top_shap.to_csv(out_dir / f"seed_{seed}" / "pia_shap_top10.csv", index=False)
            print(f"[PIA][shap] seed={seed}\n{top_shap.to_string(index=False)}")

            failing_mask = probs < 0.5
            for row_id, sample in enumerate(x_test[failing_mask][:10]):
                counterfactuals = generate_counterfactual(model, sample, feature_names)
                recommendations = generate_intervention(counterfactuals)
                intervention_rows.append(
                    {
                        "seed": seed,
                        "student_rank": row_id,
                        "counterfactuals": counterfactuals,
                        "recommendations": recommendations,
                    }
                )
                print(
                    f"[PIA][intervention] seed={seed} student_rank={row_id} "
                    f"counterfactuals={counterfactuals} recommendations={recommendations}"
                )

    stability = compute_stability(temporal_probs)
    for row in predictive_rows:
        row["Stability"] = stability
    print(f"[PIA][stability] seed={seed} Stability={stability:.4f}")

    generalization_rows = []
    if x_full_all is not None:
        for module in np.unique(modules):
            train_mask = modules != module
            test_mask = modules == module
            if train_mask.sum() < 30 or test_mask.sum() < 10 or len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
                continue
            model = _build_pia_model(seed)
            model.fit(x_full_all[train_mask], y[train_mask])
            probs = model.predict(x_full_all[test_mask])
            metrics = evaluate(y[test_mask], probs)
            generalization_rows.append({"seed": seed, "module": module, **metrics})
            _print_metric_line(f"[PIA][generalization] seed={seed} module={module}", metrics)

    intervention_rate = 0.0
    if full_model is not None and full_x_test is not None:
        intervention_rate = intervention_success_rate(full_model, full_x_test, feature_names)
    print(f"[PIA][intervention_success] seed={seed} rate={intervention_rate:.4f}")

    return {
        "predictive": pd.DataFrame(predictive_rows),
        "risk": pd.concat(risk_tables, ignore_index=True) if risk_tables else pd.DataFrame(),
        "generalization": pd.DataFrame(generalization_rows),
        "interventions": intervention_rows,
        "intervention_success": {"seed": seed, "Intervention Success": intervention_rate},
    }



def run_pia_suite(
    x_seq: np.ndarray,
    x_stat: np.ndarray,
    y: np.ndarray,
    modules: np.ndarray,
    out: Path,
    seeds: list[int],
) -> dict[str, pd.DataFrame]:
    _print_block("PIA Framework Suite")
    predictive_frames = []
    risk_frames = []
    generalization_frames = []
    intervention_rows = []
    intervention_summary = []

    for seed in seeds:
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        outputs = _run_pia_seed(x_seq, x_stat, y, modules, out, seed)
        predictive_frames.append(outputs["predictive"])
        if not outputs["risk"].empty:
            risk_frames.append(outputs["risk"])
        if not outputs["generalization"].empty:
            generalization_frames.append(outputs["generalization"])
        intervention_rows.extend(outputs["interventions"])
        intervention_summary.append(outputs["intervention_success"])

    predictive_df = pd.concat(predictive_frames, ignore_index=True)
    predictive_df.to_csv(out / "pia_predictive_results.csv", index=False)
    _print_table("PIA Predictive Results", predictive_df, sort_by=["AUC", "Accuracy", "F1"])

    stability_df = predictive_df[["seed", "window", "Stability"]].drop_duplicates().sort_values(["seed", "window"])
    stability_df.to_csv(out / "pia_stability_comparison.csv", index=False)
    _print_table("PIA Stability Comparison", stability_df)

    confidence_vs_accuracy = (
        predictive_df.groupby("window", as_index=False)[["Confidence", "Accuracy"]]
        .mean()
        .sort_values("window", key=lambda s: s.map({4: 0, 8: 1, "full": 2}))
    )
    confidence_vs_accuracy.to_csv(out / "pia_confidence_vs_accuracy.csv", index=False)
    _print_table("PIA Confidence vs Accuracy", confidence_vs_accuracy)

    risk_df = pd.concat(risk_frames, ignore_index=True) if risk_frames else pd.DataFrame(columns=["seed", "Risk Level", "Count", "Accuracy"])
    risk_df.to_csv(out / "pia_risk_stratification.csv", index=False)

    generalization_df = pd.concat(generalization_frames, ignore_index=True) if generalization_frames else pd.DataFrame()
    if not generalization_df.empty:
        generalization_df.to_csv(out / "pia_module_generalization.csv", index=False)
        _print_table("PIA Module Generalization", generalization_df, sort_by=["AUC", "Accuracy", "F1"])

    intervention_df = pd.DataFrame(
        [
            {
                "seed": row["seed"],
                "student_rank": row["student_rank"],
                "counterfactuals": json.dumps(row["counterfactuals"], ensure_ascii=False),
                "recommendations": json.dumps(row["recommendations"], ensure_ascii=False),
            }
            for row in intervention_rows
        ]
    )
    intervention_df.to_csv(out / "pia_intervention_recommendations.csv", index=False)

    intervention_summary_df = pd.DataFrame(intervention_summary)
    intervention_summary_df.to_csv(out / "pia_intervention_success.csv", index=False)
    _print_table("PIA Intervention Success", intervention_summary_df)

    x_seq_j, x_stat_j, _, _, y_j, _, _ = load_junyi_data(None, max_weeks=16)
    _, x_full_o, _ = slice_temporal_data(x_seq, x_stat, "full")
    _, x_full_j, _ = slice_temporal_data(x_seq_j, x_stat_j, "full")
    width = min(x_full_o.shape[1], x_full_j.shape[1])
    x_full_o = x_full_o[:, :width]
    x_full_j = x_full_j[:, :width]

    cross_rows = []
    seed0 = seeds[0]
    model_oj = _build_pia_model(seed0)
    model_oj.fit(x_full_o, y)
    pred_oj = model_oj.predict(x_full_j)
    met_oj = evaluate(y_j, pred_oj)
    _print_metric_line("[PIA][cross_dataset] OULAD->Junyi", met_oj)
    cross_rows.append({"seed": seed0, "train_dataset": "OULAD", "test_dataset": "Junyi", **met_oj})

    model_jo = _build_pia_model(seed0)
    model_jo.fit(x_full_j, y_j)
    pred_jo = model_jo.predict(x_full_o)
    met_jo = evaluate(y, pred_jo)
    _print_metric_line("[PIA][cross_dataset] Junyi->OULAD", met_jo)
    cross_rows.append({"seed": seed0, "train_dataset": "Junyi", "test_dataset": "OULAD", **met_jo})

    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(out / "pia_cross_dataset_generalization.csv", index=False)

    metric_catalog = pd.DataFrame({"Predictive Metrics": predictive_metrics()})
    metric_catalog.to_csv(out / "pia_predictive_metric_catalog.csv", index=False)

    return {
        "predictive": predictive_df,
        "stability": stability_df,
        "confidence_vs_accuracy": confidence_vs_accuracy,
        "risk": risk_df,
        "generalization": generalization_df,
        "intervention": intervention_df,
        "intervention_success": intervention_summary_df,
        "cross_dataset": cross_df,
        "metric_catalog": metric_catalog,
    }



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified experiment launcher: main comparison + LOMO ablation + SHAP + PIA Framework"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/unified_experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_PIA_SEEDS)))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument("--skip-main-comparison", action="store_true")
    parser.add_argument("--skip-lomo", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-pia", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pia_seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    print(f"[run_experiments] data_dir={Path(args.data_dir).resolve()}")
    print(f"[run_experiments] output_dir={out.resolve()}")
    print(f"[run_experiments] seed={args.seed}, pia_seeds={pia_seeds}, epochs={args.epochs}")
    print(
        f"[run_experiments] skip_main={args.skip_main_comparison} skip_lomo={args.skip_lomo} "
        f"skip_shap={args.skip_shap} skip_pia={args.skip_pia} skip_dynamic={args.skip_dynamic}"
    )

    tables = load_oulad_data(args.data_dir)
    x_seq, x_stat, _, _, y, modules, student_ids = extract_time_series(tables, max_weeks=16)
    print(f"[run_experiments] dataset seq={x_seq.shape}, stat={x_stat.shape}, n={len(y)}, modules={len(np.unique(modules))}")

    if not args.skip_main_comparison:
        perf_df = run_main_comparison(x_seq, x_stat, y, student_ids, out, args.seed, args.epochs, skip_dynamic=args.skip_dynamic)
        early_df = perf_df[perf_df["scenario"].isin(["4w", "8w"])].copy()
        early_df.to_csv(out / "early_window_performance.csv", index=False)
        _print_table("Early Window Comparison", early_df, sort_by=["scenario", "AUC", "Accuracy"])
        print("\n[Early Window Pivot]")
        print(early_df.pivot_table(index="model", columns="scenario", values=["Accuracy", "AUC", "F1"]).to_string())

        latex = perf_df.pivot_table(index="model", columns="scenario", values=["Accuracy", "AUC", "F1"]).to_latex(float_format="%.4f")
        (out / "performance_comparison.tex").write_text(latex, encoding="utf-8")

    if not args.skip_lomo:
        lomo_df = run_lomo_ablation(x_seq, x_stat, y, modules, out, args.seed)
        if not lomo_df.empty:
            lomo_summary = lomo_df.groupby("model", as_index=False)[["Accuracy", "AUC", "F1"]].mean().sort_values("AUC", ascending=False)
            lomo_summary.to_csv(out / "lomo_ablation_summary.csv", index=False)
            _print_table("LOMO Ablation Summary", lomo_summary, sort_by=["AUC", "Accuracy", "F1"])

    if not args.skip_shap:
        run_shap_analysis(x_seq, x_stat, y, out, args.seed)

    if not args.skip_pia:
        run_pia_suite(x_seq, x_stat, y, modules, out, pia_seeds)

    print(f"\nAll experiments completed. Results saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
