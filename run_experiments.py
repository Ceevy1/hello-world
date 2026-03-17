from __future__ import annotations

import argparse
from pathlib import Path

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

from analysis import (
    compute_group_representation_distances,
    compute_weight_analysis,
    plot_roc_curves,
    plot_tsne_embeddings,
    plot_weight_trajectories,
    stat_tests_from_summary,
    write_markdown_report,
)
from data_loader import extract_time_series, load_junyi_data, load_oulad_data, make_split_loaders
from evaluation.metrics import classification_metrics
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
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=3)
    return p.parse_args()


def _window_data(x_seq: np.ndarray, window_len: int, full_len: int) -> tuple[np.ndarray, np.ndarray]:
    xw = np.copy(x_seq)
    xw[:, window_len:, :] = 0.0
    week_idx = np.full(x_seq.shape[0], min(window_len - 1, full_len - 1), dtype=np.int64)
    return xw, week_idx


def _predict_proba(clf, x: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x)[:, 1]
    score = clf.decision_function(x)
    return 1.0 / (1.0 + np.exp(-score))


def _train_dynamicfusion_for_dataset(
    x_seq: np.ndarray,
    x_stat: np.ndarray,
    y: np.ndarray,
    week_idx: np.ndarray,
    student_ids: np.ndarray,
    out_dir: Path,
    seed: int,
    epochs: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_idx = np.arange(len(y), dtype=np.int64)
    tr, va, te, graph = make_split_loaders(x_seq, x_stat, node_idx, week_idx, y, random_state=seed)
    model = DynamicFusionEnhanced(seq_input_dim=x_seq.shape[-1], stat_input_dim=x_stat.shape[-1], graph_input_dim=16)
    fit(model, tr, va, graph, TrainConfig(epochs=epochs), output_dir=str(out_dir))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(splitter.split(np.zeros(len(y)), y))
    test_student_ids = student_ids[test_idx]

    metrics = export_predictions(model, te, graph, output_dir=str(out_dir), student_ids=test_student_ids)
    pred_df = pd.read_csv(out_dir / "predictions.csv")
    probs = pred_df["y_pred_prob"].to_numpy()
    y_true = pred_df["y_true"].to_numpy()
    w = np.load(out_dir / "fusion_weights.npy") if (out_dir / "fusion_weights.npy").exists() else np.empty((0, 3))
    r = np.load(out_dir / "representations.npy") if (out_dir / "representations.npy").exists() else np.empty((0, 128))
    return metrics, probs, y_true, w, r


def _run_baselines(x_tab: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(sss.split(x_tab, y))
    x_tr, x_te = x_tab[tr_idx], x_tab[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    preds = {}
    for name, clf in BASELINES.items():
        clf.fit(x_tr, y_tr)
        preds[name] = _predict_proba(clf, x_te)
    return y_te, te_idx, preds


def run(data_dir: str, output_dir: str, epochs: int, seed: int, repeats: int) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[run_experiments] output_dir={out.resolve()}")

    # ---------- OULAD base data ----------
    x_seq_full, x_stat, _, _, y, modules, student_ids = extract_time_series(load_oulad_data(data_dir), max_weeks=WINDOWS["full"])

    rows = []
    roc_curves = {}
    weight_map: dict[str, np.ndarray] = {}
    repr_map: dict[str, np.ndarray] = {}

    # ---------- 4w / 8w / full with repeated seeds ----------
    for rep in range(repeats):
        rs = seed + rep
        for window_name, window_len in WINDOWS.items():
            x_seq_w, week_idx_w = _window_data(x_seq_full, window_len, WINDOWS["full"])
            x_tab_w = extract_features(x_seq_w, x_stat)

            y_test, te_idx, base_probs = _run_baselines(x_tab_w, y, rs)
            for model_name, prob in base_probs.items():
                met = classification_metrics(y_test, prob)
                met["AUPRC"] = float(average_precision_score(y_test, prob))
                rows.append({"scenario": f"OULAD_{window_name}", "seed": rs, "model": model_name, **met})

            scen_dir = out / f"oulad_{window_name}_seed{rs}"
            scen_dir.mkdir(exist_ok=True)
            dyn_m, dyn_prob, dyn_true, dyn_w, dyn_r = _train_dynamicfusion_for_dataset(
                x_seq_w, x_stat, y, week_idx_w, student_ids, scen_dir, rs, epochs
            )
            dyn_m["AUPRC"] = float(average_precision_score(dyn_true, dyn_prob))
            rows.append({"scenario": f"OULAD_{window_name}", "seed": rs, "model": "DynamicFusion-Enhanced", **dyn_m})

            fpr = np.linspace(0, 1, 101)
            # quick ROC interpolation
            order = np.argsort(dyn_prob)
            y_sorted = dyn_true[order]
            tpr = np.cumsum(y_sorted[::-1]) / max(1, int((y_sorted == 1).sum()))
            x = np.linspace(0, 1, len(tpr))
            roc_curves[f"{window_name}_seed{rs}"] = (fpr, np.interp(fpr, x, tpr))
            weight_map[f"OULAD_{window_name}_seed{rs}"] = dyn_w
            repr_map[f"OULAD_{window_name}_seed{rs}"] = dyn_r

    perf = pd.DataFrame(rows)
    perf.to_csv(out / "performance_summary.csv", index=False)

    # ---------- Module-level table (mean/std across seeds) ----------
    mod_rows = []
    for mod in np.unique(modules):
        idx = modules == mod
        if idx.sum() < 20:
            continue
        for w_name, w_len in WINDOWS.items():
            scores_acc, scores_auc = [], []
            for rep in range(repeats):
                rs = seed + rep
                x_seq_w, _ = _window_data(x_seq_full, w_len, WINDOWS["full"])
                x_tab = extract_features(x_seq_w[idx], x_stat[idx])
                yy = y[idx]
                y_te, _, probs = _run_baselines(x_tab, yy, rs)
                p = probs["LogisticRegression"]
                met = classification_metrics(y_te, p)
                scores_acc.append(met["Accuracy"])
                scores_auc.append(met["AUC"])
            mod_rows.append(
                {
                    "module": mod,
                    "samples": int(idx.sum()),
                    f"{w_name}_Acc_mean": float(np.mean(scores_acc)),
                    f"{w_name}_Acc_std": float(np.std(scores_acc)),
                    f"{w_name}_AUC_mean": float(np.mean(scores_auc)),
                    f"{w_name}_AUC_std": float(np.std(scores_auc)),
                }
            )
    pd.DataFrame(mod_rows).to_csv(out / "module_level_results.csv", index=False)

    # ---------- LOMO / LOPO ----------
    x_tab_full = extract_features(x_seq_full, x_stat)
    lomo_rows = []
    for mod in np.unique(modules):
        tr = modules != mod
        te = modules == mod
        if tr.sum() < 20 or te.sum() < 10 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=500)
        clf.fit(x_tab_full[tr], y[tr])
        prob = clf.predict_proba(x_tab_full[te])[:, 1]
        lomo_rows.append({"模块ID": mod, **classification_metrics(y[te], prob)})
    pd.DataFrame(lomo_rows).to_csv(out / "oulad_LOMO_results.csv", index=False)

    # LOPO via presentation parsed from module string suffix if available
    pres = np.array([m[-5:] if len(m) >= 5 else m for m in modules])
    lopo_rows = []
    for pr in np.unique(pres):
        tr = pres != pr
        te = pres == pr
        if tr.sum() < 20 or te.sum() < 10 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=500)
        clf.fit(x_tab_full[tr], y[tr])
        prob = clf.predict_proba(x_tab_full[te])[:, 1]
        lopo_rows.append({"模块ID": pr, **classification_metrics(y[te], prob)})
    pd.DataFrame(lopo_rows).to_csv(out / "oulad_LOPO_results.csv", index=False)

    # ---------- Cross-dataset (OULAD <-> Junyi) ----------
    j_seq, j_stat, _, j_week, j_y, j_mod, j_sid = load_junyi_data(None, max_weeks=WINDOWS["full"])
    j_tab = extract_features(j_seq, j_stat)
    # OULAD -> Junyi
    src_dir = out / "cross_oulad_to_junyi"
    src_dir.mkdir(exist_ok=True)
    _, _, _, _, _ = _train_dynamicfusion_for_dataset(x_seq_full, x_stat, y, np.full(len(y), WINDOWS['full']-1), student_ids, src_dir, seed, epochs)
    clf = LogisticRegression(max_iter=500)
    clf.fit(x_tab_full, y)
    prob_oj = clf.predict_proba(j_tab)[:, 1]
    met_oj = classification_metrics(j_y, prob_oj)

    # Junyi -> OULAD
    clf2 = LogisticRegression(max_iter=500)
    clf2.fit(j_tab, j_y)
    prob_jo = clf2.predict_proba(x_tab_full)[:, 1]
    met_jo = classification_metrics(y, prob_jo)

    cross_df = pd.DataFrame([
        {"训练数据": "OULAD", "测试数据": "Junyi", **met_oj},
        {"训练数据": "Junyi", "测试数据": "OULAD", **met_jo},
    ])
    cross_df.to_csv(out / "cross_dataset_results.csv", index=False)

    # ---------- Weight analysis / representation distances ----------
    weight_df = compute_weight_analysis(weight_map)
    weight_df.to_csv(out / "weight_analysis.csv", index=False)

    repr_df = compute_group_representation_distances(repr_map)
    repr_df.to_csv(out / "representation_distances.csv", index=False)

    # ---------- SHAP importance by module ----------
    shap_rows = []
    feat_names = [f"f{i}" for i in range(x_tab_full.shape[1])]
    for mod in np.unique(modules):
        idx = modules == mod
        if idx.sum() < 20 or len(np.unique(y[idx])) < 2:
            continue
        model = RandomForestClassifier(n_estimators=200, random_state=seed)
        model.fit(x_tab_full[idx], y[idx])
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:8]
        for i in top_idx:
            shap_rows.append({"模块": mod, "特征": feat_names[i], "SHAP值": float(imp[i])})
    pd.DataFrame(shap_rows).to_csv(out / "shap_importance.csv", index=False)

    # ---------- Stat tests ----------
    stat_df = stat_tests_from_summary(perf)
    stat_df.to_csv(out / "stat_tests.csv", index=False)

    # ---------- Visualizations ----------
    plot_roc_curves(roc_curves, out / "roc_curves.png")
    plot_weight_trajectories(weight_map, out / "weight_trajectories.png")
    plot_tsne_embeddings(repr_map, out / "tsne_embeddings.png")


    # canonical root artifacts from full-window experiment
    full_dir = out / f"oulad_full_seed{seed}"
    for name in ["model_best.pth", "predictions.csv", "confusion_matrix.npz", "roc_curve_data.csv", "loss_curve.csv", "weights_trajectory.csv", "representation_distances.npy"]:
        src = full_dir / name
        if src.exists():
            (out / name).write_bytes(src.read_bytes())

    # ---------- report ----------
    write_markdown_report(perf, stat_df, out / "experiment_report.md")

    print("[run_experiments] complete")


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.output_dir, args.epochs, args.seed, args.repeats)
