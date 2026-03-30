from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVR, SVC

from evaluate import classification_metrics, regression_metrics
from preprocessing import build_classification_labels, preprocess_scores
from importlib.util import module_from_spec, spec_from_file_location
import sys

_train_spec = spec_from_file_location("self_train", "train.py")
_train_mod = module_from_spec(_train_spec)
assert _train_spec and _train_spec.loader
sys.modules[_train_spec.name] = _train_mod
_train_spec.loader.exec_module(_train_mod)
TrainConfig = _train_mod.TrainConfig
train_and_evaluate = _train_mod.train_and_evaluate


def _build_feature_matrix(df: pd.DataFrame, exercise_cols: list[str], lab_cols: list[str], static_cols: list[str]) -> np.ndarray:
    return df[exercise_cols + lab_cols + static_cols].to_numpy(dtype=np.float32)


def _to_probabilities(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        z = scores - np.max(scores, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)
    pred = model.predict(x).astype(int)
    n_classes = int(pred.max()) + 1
    prob = np.zeros((len(pred), n_classes), dtype=np.float32)
    prob[np.arange(len(pred)), pred] = 1.0
    return prob


def compare_regression_baselines(csv_path: str = "data/student_scores.csv") -> pd.DataFrame:
    prepared = preprocess_scores(csv_path)
    x_train = _build_feature_matrix(prepared.train_df, prepared.exercise_cols, prepared.lab_cols, prepared.static_cols)
    y_train = prepared.train_df[prepared.target_col].to_numpy(dtype=np.float32)
    x_test = _build_feature_matrix(prepared.test_df, prepared.exercise_cols, prepared.lab_cols, prepared.static_cols)
    y_test = prepared.test_df[prepared.target_col].to_numpy(dtype=np.float32)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "SVR": SVR(C=1.0, epsilon=0.1),
        "KNNRegressor": KNeighborsRegressor(n_neighbors=1),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }

    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        m = regression_metrics(y_test, pred)
        rows.append({"Model": name, **m})

    _, dyn_metrics = train_and_evaluate(TrainConfig(csv_path=csv_path, task="regression"))
    rows.append({"Model": "DynamicFusion", **dyn_metrics})

    out = pd.DataFrame(rows).sort_values(by="RMSE", key=lambda s: s.fillna(np.inf)).reset_index(drop=True)
    return out


def compare_classification_baselines(csv_path: str = "data/student_scores.csv") -> pd.DataFrame:
    prepared = preprocess_scores(csv_path)
    train_df, _ = build_classification_labels(prepared.train_df, prepared.target_col)
    test_df, _ = build_classification_labels(prepared.test_df, prepared.target_col)

    x_train = _build_feature_matrix(train_df, prepared.exercise_cols, prepared.lab_cols, prepared.static_cols)
    y_train = train_df["label_cls"].to_numpy(dtype=np.int64)
    x_test = _build_feature_matrix(test_df, prepared.exercise_cols, prepared.lab_cols, prepared.static_cols)
    y_test = test_df["label_cls"].to_numpy(dtype=np.int64)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "LinearSVC": LinearSVC(),
        "SVC": SVC(probability=True),
        "KNNClassifier": KNeighborsClassifier(n_neighbors=1),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
        "GaussianNB": GaussianNB(),
    }

    rows = []
    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
            prob = _to_probabilities(model, x_test)
            m = classification_metrics(y_test, prob)
            rows.append({"Model": name, **m})
        except Exception as e:  # noqa: BLE001
            rows.append({"Model": name, "Accuracy": np.nan, "F1": np.nan, "AUC": np.nan, "Error": str(e)})

    _, dyn_metrics = train_and_evaluate(TrainConfig(csv_path=csv_path, task="classification"))
    rows.append({"Model": "DynamicFusion", **dyn_metrics})

    out = pd.DataFrame(rows).sort_values(by="F1", key=lambda s: s.fillna(-1), ascending=False).reset_index(drop=True)
    return out


def run_and_save(csv_path: str = "data/student_scores.csv", out_dir: str = "outputs") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    reg = compare_regression_baselines(csv_path)
    cls = compare_classification_baselines(csv_path)

    reg_path = Path(out_dir) / "self_dataset_regression_baselines.csv"
    cls_path = Path(out_dir) / "self_dataset_classification_baselines.csv"
    reg.to_csv(reg_path, index=False, encoding="utf-8-sig")
    cls.to_csv(cls_path, index=False, encoding="utf-8-sig")

    print("Saved:", reg_path)
    print(reg)
    print("Saved:", cls_path)
    print(cls)


if __name__ == "__main__":
    run_and_save()
