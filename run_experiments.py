from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_loader import build_dataloaders, extract_time_series, load_oulad_data
from feature_engineering import extract_features
from model import DynamicFusionEnhanced
from trainer import TrainConfig, export_predictions, fit


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
    "NaiveLogRegSmall": LogisticRegression(C=0.5, max_iter=300),
}


def eval_prob(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y_hat = (prob >= 0.5).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "Accuracy": float(accuracy_score(y_true, y_hat)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
    }


def run(data_dir: str = "data", output_dir: str = "outputs") -> None:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader, graph = build_dataloaders(data_dir)

    tables = load_oulad_data(data_dir)
    x_seq, x_stat, _, _, y = extract_time_series(tables)
    x_tab = extract_features(x_seq, x_stat)
    x_train, x_test, y_train, y_test = train_test_split(x_tab, y, test_size=0.2, random_state=42, stratify=y)

    baseline_rows = []
    for name, clf in BASELINES.items():
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(x_test)
        baseline_rows.append({"model": name, **eval_prob(y_test, prob)})

    model = DynamicFusionEnhanced(seq_input_dim=x_seq.shape[-1], stat_input_dim=x_stat.shape[-1], graph_input_dim=16)
    fit(model, train_loader, val_loader, graph, TrainConfig(), output_dir=output_dir)
    ours = export_predictions(model, test_loader, graph, output_dir=output_dir)
    baseline_rows.append({"model": "DynamicFusion-Enhanced", "AUC": ours["AUC"], "Accuracy": ours["Accuracy"], "F1": np.nan})

    pd.DataFrame(baseline_rows).to_csv(out / "experiment_results.csv", index=False)


if __name__ == "__main__":
    run()
