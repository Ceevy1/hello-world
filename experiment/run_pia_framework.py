from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import extract_time_series, load_junyi_data, load_oulad_data
from experiment.actionable import generate_intervention, stratify_predictions
from experiment.evaluation.intervention_metrics import intervention_success_rate
from experiment.evaluation.metrics import evaluate
from experiment.evaluation.predictive_metrics import predictive_metrics
from experiment.explainable import generate_counterfactual
from experiment.model_interface import SklearnPIAModel
from experiment.predictive import compute_confidence, compute_stability, slice_temporal_data, temporal_predict

SEEDS = [42, 52, 62, 72, 82]
WINDOWS = [4, 8, "full"]



def _feature_names(x_stat: np.ndarray) -> list[str]:
    stat_names = [f"stat_{i}" for i in range(x_stat.shape[1])]
    engineered = [
        "total_clicks",
        "avg_clicks",
        "click_std",
        "click_entropy",
        "active_week_ratio",
        "active_week_count",
    ]
    return stat_names + engineered



def _build_model(seed: int) -> SklearnPIAModel:
    from sklearn.ensemble import RandomForestClassifier

    estimator = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=3,
        random_state=seed,
    )
    return SklearnPIAModel(estimator=estimator)



def _train_test_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    return next(splitter.split(np.zeros(len(y)), y))



def _collect_seed_outputs(x_seq: np.ndarray, x_stat: np.ndarray, y: np.ndarray, modules: np.ndarray, seed: int, out_dir: Path) -> dict[str, object]:
    train_idx, test_idx = _train_test_indices(y, seed)
    feature_names = _feature_names(x_stat)

    seed_rows: list[dict[str, object]] = []
    risk_tables: list[pd.DataFrame] = []
    interventions: list[dict[str, object]] = []

    temporal_probs: dict[object, np.ndarray] = {}
    full_model: SklearnPIAModel | None = None
    full_x_test: np.ndarray | None = None

    for week in WINDOWS:
        _, x_features, _ = slice_temporal_data(x_seq, x_stat, week)
        x_train = x_features[train_idx]
        x_test = x_features[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = _build_model(seed)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        temporal_probs[week] = preds

        confidence = compute_confidence(preds)
        metrics = evaluate(y_test, preds)
        seed_rows.append(
            {
                "seed": seed,
                "window": week,
                **metrics,
                "Confidence": float(np.mean(confidence)),
            }
        )

        if week == "full":
            full_model = model
            full_x_test = x_test
            risk_df = stratify_predictions(preds, y_test)
            risk_df.insert(0, "seed", seed)
            risk_tables.append(risk_df)

            explain_dir = out_dir / f"seed_{seed}" / "shap"
            shap_result = model.explain(x_test, feature_names, save_dir=explain_dir)
            top_shap = shap_result["importance"].head(10).copy()
            top_shap.insert(0, "seed", seed)
            top_shap.to_csv(out_dir / f"seed_{seed}" / "shap_top10.csv", index=False)

            low_conf_mask = preds < 0.5
            for row_id, sample in enumerate(x_test[low_conf_mask][:10]):
                cf = generate_counterfactual(model, sample, feature_names)
                recs = generate_intervention(cf)
                interventions.append(
                    {
                        "seed": seed,
                        "student_rank": row_id,
                        "counterfactuals": cf,
                        "recommendations": recs,
                    }
                )

    stability = compute_stability(temporal_probs)
    for row in seed_rows:
        row["Stability"] = stability

    generalization_rows = []
    _, x_full, _ = slice_temporal_data(x_seq, x_stat, "full")
    for module in np.unique(modules):
        train_mask = modules != module
        test_mask = modules == module
        if train_mask.sum() < 30 or test_mask.sum() < 10 or len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
            continue
        model = _build_model(seed)
        model.fit(x_full[train_mask], y[train_mask])
        probs = model.predict(x_full[test_mask])
        met = evaluate(y[test_mask], probs)
        generalization_rows.append({"seed": seed, "module": module, **met})

    intervention_rate = 0.0
    if full_model is not None and full_x_test is not None:
        intervention_rate = intervention_success_rate(full_model, full_x_test, feature_names)

    return {
        "predictive": pd.DataFrame(seed_rows),
        "risk": pd.concat(risk_tables, ignore_index=True) if risk_tables else pd.DataFrame(),
        "interventions": interventions,
        "generalization": pd.DataFrame(generalization_rows),
        "intervention_success_rate": intervention_rate,
    }



def _cross_dataset_generalization(seed: int, out_dir: Path) -> pd.DataFrame:
    x_seq_o, x_stat_o, _, _, y_o, _, _ = extract_time_series(load_oulad_data("data"), max_weeks=16)
    x_seq_j, x_stat_j, _, _, y_j, _, _ = load_junyi_data(None, max_weeks=16)
    _, x_full_o, _ = slice_temporal_data(x_seq_o, x_stat_o, "full")
    _, x_full_j, _ = slice_temporal_data(x_seq_j, x_stat_j, "full")

    width = min(x_full_o.shape[1], x_full_j.shape[1])
    x_full_o = x_full_o[:, :width]
    x_full_j = x_full_j[:, :width]

    model_oj = _build_model(seed)
    model_oj.fit(x_full_o, y_o)
    pred_oj = model_oj.predict(x_full_j)

    model_jo = _build_model(seed)
    model_jo.fit(x_full_j, y_j)
    pred_jo = model_jo.predict(x_full_o)

    rows = [
        {"seed": seed, "train_dataset": "OULAD", "test_dataset": "Junyi", **evaluate(y_j, pred_oj)},
        {"seed": seed, "train_dataset": "Junyi", "test_dataset": "OULAD", **evaluate(y_o, pred_jo)},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cross_dataset_generalization.csv", index=False)
    return df



def run_pia_framework(data_dir: str, output_dir: str, seeds: list[int] | None = None) -> dict[str, pd.DataFrame | list[dict[str, object]]]:
    seeds = seeds or SEEDS
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_seq, x_stat, _, _, y, modules, student_ids = extract_time_series(load_oulad_data(data_dir), max_weeks=16)
    del student_ids  # reserved for future exports

    predictive_tables = []
    risk_tables = []
    generalization_tables = []
    intervention_rows = []
    intervention_summary = []

    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        outputs = _collect_seed_outputs(x_seq, x_stat, y, modules, seed, out_dir)
        predictive_tables.append(outputs["predictive"])
        if not outputs["risk"].empty:
            risk_tables.append(outputs["risk"])
        if not outputs["generalization"].empty:
            generalization_tables.append(outputs["generalization"])
        intervention_rows.extend(outputs["interventions"])
        intervention_summary.append({"seed": seed, "Intervention Success": outputs["intervention_success_rate"]})

    predictive_df = pd.concat(predictive_tables, ignore_index=True)
    predictive_df.to_csv(out_dir / "predictive_results.csv", index=False)

    stability_df = (
        predictive_df[["seed", "window", "Stability"]]
        .drop_duplicates()
        .sort_values(["seed", "window"])
    )
    stability_df.to_csv(out_dir / "stability_comparison.csv", index=False)

    confidence_vs_accuracy = (
        predictive_df.groupby("window", as_index=False)[["Confidence", "Accuracy"]]
        .mean()
        .sort_values("window", key=lambda s: s.map({4: 0, 8: 1, "full": 2}))
    )
    confidence_vs_accuracy.to_csv(out_dir / "confidence_vs_accuracy.csv", index=False)

    risk_df = pd.concat(risk_tables, ignore_index=True) if risk_tables else pd.DataFrame(columns=["seed", "Risk Level", "Count", "Accuracy"])
    risk_df.to_csv(out_dir / "risk_stratification.csv", index=False)

    generalization_df = pd.concat(generalization_tables, ignore_index=True) if generalization_tables else pd.DataFrame()
    if not generalization_df.empty:
        generalization_df.to_csv(out_dir / "module_generalization.csv", index=False)

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
    intervention_df.to_csv(out_dir / "intervention_recommendations.csv", index=False)

    intervention_summary_df = pd.DataFrame(intervention_summary)
    intervention_summary_df.to_csv(out_dir / "intervention_success.csv", index=False)

    cross_dataset_df = _cross_dataset_generalization(seeds[0], out_dir)

    summary = pd.DataFrame(
        {
            "Predictive Metrics": predictive_metrics(),
        }
    )
    summary.to_csv(out_dir / "predictive_metric_catalog.csv", index=False)

    return {
        "predictive": predictive_df,
        "risk": risk_df,
        "generalization": generalization_df,
        "intervention": intervention_df,
        "intervention_summary": intervention_summary_df,
        "cross_dataset": cross_dataset_df,
        "confidence_vs_accuracy": confidence_vs_accuracy,
        "stability": stability_df,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PIA Framework experiment suite.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/pia_framework")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    outputs = run_pia_framework(args.data_dir, args.output_dir, seeds=seeds)

    print("PIA Framework completed.")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    for name, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            print(f"[{name}] rows={len(value)}")


if __name__ == "__main__":
    main()
