from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_loader import OULADDataset, extract_time_series, load_oulad_data, make_split_loaders
from evaluation.metrics import classification_metrics
from evaluation.statistics import significance_tests
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OULAD comprehensive experiments")
    parser.add_argument("--data-dir", default="data", help="Directory containing OULAD CSV files")
    parser.add_argument("--output-dir", default="outputs", help="Directory to store result artifacts")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs for DynamicFusion-Enhanced")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _predict_proba(clf, x: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x)[:, 1]
    score = clf.decision_function(x)
    return 1 / (1 + np.exp(-score))


def _fit_baselines(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> dict[str, np.ndarray]:
    preds = {}
    for name, clf in BASELINES.items():
        clf.fit(x_train, y_train)
        preds[name] = _predict_proba(clf, x_test)
    return preds


def _window_data(x_seq: np.ndarray, window_len: int, full_len: int) -> tuple[np.ndarray, np.ndarray]:
    xw = np.copy(x_seq)
    xw[:, window_len:, :] = 0.0
    week_idx = np.full(x_seq.shape[0], min(window_len - 1, full_len - 1), dtype=np.int64)
    return xw, week_idx


def _run_dynamic_fusion(
    x_seq: np.ndarray,
    x_stat: np.ndarray,
    y: np.ndarray,
    week_idx: np.ndarray,
    student_ids: np.ndarray,
    output_dir: Path,
    epochs: int,
    random_state: int,
) -> tuple[dict[str, float], np.ndarray]:
    node_idx = np.arange(len(y), dtype=np.int64)
    train_loader, val_loader, test_loader, graph = make_split_loaders(
        x_seq=x_seq,
        x_stat=x_stat,
        node_idx=node_idx,
        week_idx=week_idx,
        y=y,
        random_state=random_state,
    )
    model = DynamicFusionEnhanced(seq_input_dim=x_seq.shape[-1], stat_input_dim=x_stat.shape[-1], graph_input_dim=16)
    fit(model, train_loader, val_loader, graph, TrainConfig(epochs=epochs), output_dir=str(output_dir))

    # align student IDs to test split order
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    _, test_idx = next(splitter.split(np.zeros(len(y)), y))
    test_student_ids = student_ids[test_idx]

    metrics = export_predictions(model, test_loader, graph, output_dir=str(output_dir), student_ids=test_student_ids)
    pred_df = pd.read_csv(output_dir / "predictions.csv")
    return metrics, pred_df["y_pred_prob"].to_numpy()


def _save_shap_like_values(x_tab: np.ndarray, y: np.ndarray, output_dir: Path) -> None:
    feature_names = [f"f{i}" for i in range(x_tab.shape[1])]
    try:
        import shap

        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(x_tab, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_tab)
        if isinstance(shap_values, list):
            vals = np.abs(shap_values[-1]).mean(axis=0)
        else:
            vals = np.abs(shap_values).mean(axis=0)
        pd.DataFrame({"feature": feature_names, "shap_value": vals}).to_csv(output_dir / "shap_values.csv", index=False)
    except Exception:
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(x_tab, y)
        vals = model.feature_importances_
        pd.DataFrame({"feature": feature_names, "shap_value": vals}).to_csv(output_dir / "shap_values.csv", index=False)


def run(data_dir: str = "data", output_dir: str = "outputs", epochs: int = 12, seed: int = 42) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[run_experiments] data_dir={Path(data_dir).resolve()}")
    print(f"[run_experiments] output_dir={out.resolve()}")

    x_seq_full, x_stat, _, _, y, modules, student_ids = extract_time_series(load_oulad_data(data_dir), max_weeks=WINDOWS["full"])
    x_tab_full = extract_features(x_seq_full, x_stat)

    print(f"[run_experiments] n={len(y)}, seq={x_seq_full.shape}, tab={x_tab_full.shape}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(x_tab_full, y))

    all_rows: list[dict[str, float | str]] = []
    full_baseline_probs: dict[str, np.ndarray] = {}
    full_dynamic_prob: np.ndarray | None = None

    for w_name, w_len in WINDOWS.items():
        print(f"[run_experiments] window={w_name}")
        x_seq_w, week_idx_w = _window_data(x_seq_full, w_len, WINDOWS["full"])
        x_tab_w = extract_features(x_seq_w, x_stat)

        x_train, x_test = x_tab_w[train_idx], x_tab_w[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        baseline_pred = _fit_baselines(x_train, y_train, x_test)
        for model_name, p in baseline_pred.items():
            m = classification_metrics(y_test, p)
            m["AUPRC"] = float(average_precision_score(y_test, p))
            all_rows.append({"model": model_name, "window": w_name, **m})
            print(f"  baseline {model_name}: AUC={m['AUC']:.4f}, Acc={m['Accuracy']:.4f}")

        wf_out = out / w_name
        wf_out.mkdir(exist_ok=True)
        dyn_metrics, dyn_prob = _run_dynamic_fusion(
            x_seq=x_seq_w,
            x_stat=x_stat,
            y=y,
            week_idx=week_idx_w,
            student_ids=student_ids,
            output_dir=wf_out,
            epochs=epochs,
            random_state=seed,
        )
        dyn_metrics["AUPRC"] = float(average_precision_score(y_test, dyn_prob))
        all_rows.append({"model": "DynamicFusion-Enhanced", "window": w_name, **dyn_metrics})
        print(f"  dynamic fusion: AUC={dyn_metrics['AUC']:.4f}, Acc={dyn_metrics['Accuracy']:.4f}")

        if w_name == "full":
            full_baseline_probs = baseline_pred
            full_dynamic_prob = dyn_prob
            # copy required canonical artifacts to root output
            for name in [
                "model_best.pth",
                "predictions.csv",
                "confusion_matrix.npz",
                "roc_curve_data.csv",
                "loss_curve.csv",
                "weights_trajectory.csv",
                "representation_distances.npy",
            ]:
                src = wf_out / name
                if src.exists():
                    (out / name).write_bytes(src.read_bytes())

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out / "experiment_results.csv", index=False)

    # significance tests (full window, ours vs each baseline)
    sig_rows = []
    y_test = y[test_idx]
    if full_dynamic_prob is not None:
        for name, p in full_baseline_probs.items():
            st = significance_tests(y_test, full_dynamic_prob, p)
            sig_rows.append({"comparison": f"DynamicFusion-Enhanced vs {name}", **st})
    pd.DataFrame(sig_rows).to_csv(out / "significance_tests.csv", index=False)

    # LOMO generalization (quick protocol with LogisticRegression)
    lomo_rows = []
    unique_modules = np.unique(modules)
    x_tab_lomo = extract_features(x_seq_full, x_stat)
    for mod in unique_modules:
        tr = modules != mod
        te = modules == mod
        if tr.sum() < 20 or te.sum() < 10 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=500)
        clf.fit(x_tab_lomo[tr], y[tr])
        prob = clf.predict_proba(x_tab_lomo[te])[:, 1]
        met = classification_metrics(y[te], prob)
        lomo_rows.append({"held_out_module": mod, **met})
    pd.DataFrame(lomo_rows).to_csv(out / "lomo_results.csv", index=False)

    _save_shap_like_values(x_tab_full[train_idx], y[train_idx], out)

    with (out / "experiment_log.json").open("w", encoding="utf-8") as f:
        json.dump({"epochs": epochs, "seed": seed, "windows": WINDOWS, "n_samples": int(len(y))}, f, ensure_ascii=False, indent=2)

    print("[run_experiments] done")


if __name__ == "__main__":
    args = parse_args()
    run(data_dir=args.data_dir, output_dir=args.output_dir, epochs=args.epochs, seed=args.seed)
