from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

XGBClassifier = None
XGBRegressor = None

from data_preprocessing import OULADDatasetBuilder


SELF_REQUIRED_COLUMNS = [
    "考勤",
    "练习1",
    "练习2",
    "练习3",
    "总平时 成绩",
    "实验1",
    "实验2",
    "实验3",
    "实验4",
    "实验5",
    "实验6",
    "实验7",
    "报告",
    "总实验 成绩",
    "平时成绩",
    "总期末成绩",
    "总评成绩",
]

UNIFIED_FEATURES = [
    "performance_assessment",
    "engagement_activity",
    "behavior_stability",
    "practice_mastery",
    "report_quality",
    "consistency_index",
]


@dataclass
class DomainData:
    name: str
    X: pd.DataFrame
    y_reg: np.ndarray
    y_cls: np.ndarray


class DynamicFusionRegressor:
    """Simple modality-weighted regressor used as dynamic fusion baseline."""

    def __init__(self) -> None:
        self.weights_: np.ndarray | None = None
        self.model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DynamicFusionRegressor":
        w = []
        for cols in ([0, 3], [1, 4], [2, 5]):
            lr = LinearRegression().fit(X[:, cols], y)
            pred = lr.predict(X[:, cols])
            rmse = float(np.sqrt(mean_squared_error(y, pred)))
            w.append(1.0 / max(rmse, 1e-6))
        self.weights_ = np.array(w, dtype=np.float32)
        self.weights_ = self.weights_ / self.weights_.sum()
        X_fused = self._fuse(X)
        self.model.fit(X_fused, y)
        return self

    def _fuse(self, X: np.ndarray) -> np.ndarray:
        assert self.weights_ is not None
        perf = X[:, [0, 3]] * self.weights_[0]
        eng = X[:, [1, 4]] * self.weights_[1]
        beh = X[:, [2, 5]] * self.weights_[2]
        return np.concatenate([perf, eng, beh], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._fuse(X))


class DynamicFusionClassifier:
    """Simple modality-weighted classifier used as dynamic fusion baseline."""

    def __init__(self) -> None:
        self.reg = DynamicFusionRegressor()
        self.clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DynamicFusionClassifier":
        self.reg.fit(X, y.astype(np.float32))
        self.clf.fit(self.reg._fuse(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.reg._fuse(X))


def _minmax(series: pd.Series) -> pd.Series:
    vmin = float(series.min())
    vmax = float(series.max())
    if vmax - vmin < 1e-12:
        return pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)
    return (series - vmin) / (vmax - vmin)


def load_self_domain(self_scores_path: str) -> DomainData:
    df = pd.read_csv(self_scores_path, encoding="utf-8-sig")
    missing = [c for c in SELF_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"students_scores.csv 缺失列: {missing}")

    practice_mean = df[["练习1", "练习2", "练习3"]].mean(axis=1)
    lab_mean = df[[f"实验{i}" for i in range(1, 8)]].mean(axis=1)
    practice_std = df[["练习1", "练习2", "练习3"]].std(axis=1).fillna(0)
    lab_std = df[[f"实验{i}" for i in range(1, 8)]].std(axis=1).fillna(0)

    aligned = pd.DataFrame(
        {
            "performance_assessment": _minmax((practice_mean + df["总平时 成绩"] + df["总实验 成绩"]) / 3.0),
            "engagement_activity": _minmax((df["考勤"] + df["平时成绩"]) / 2.0),
            "behavior_stability": _minmax(1.0 / (1.0 + practice_std + lab_std)),
            "practice_mastery": _minmax((practice_mean + lab_mean) / 2.0),
            "report_quality": _minmax((df["报告"] + df["总期末成绩"]) / 2.0),
            "consistency_index": _minmax(df["平时成绩"] - df["总期末成绩"]).abs().rsub(1.0),
        }
    )

    y_reg = df["总评成绩"].to_numpy(dtype=np.float32)
    y_cls = (y_reg >= 60).astype(np.int64)
    return DomainData(name="SelfDataset", X=aligned[UNIFIED_FEATURES], y_reg=y_reg, y_cls=y_cls)


def load_oulad_domain(data_dir: str) -> DomainData:
    ds = OULADDatasetBuilder(data_dir=data_dir).build()
    tab = pd.DataFrame(ds["tabular"], columns=ds["tab_feature_names"])

    def pick(col: str) -> pd.Series:
        return tab[col] if col in tab.columns else pd.Series(np.zeros(len(tab)), index=tab.index)

    aligned = pd.DataFrame(
        {
            "performance_assessment": _minmax(0.5 * pick("mean_clicks") + 0.3 * pick("early_click_ratio") + 0.2 * pick("max_weekly_clicks")),
            "engagement_activity": _minmax(0.6 * pick("total_clicks") + 0.4 * pick("active_weeks")),
            "behavior_stability": _minmax(1.0 / (1.0 + np.abs(pick("click_cv")) + np.abs(pick("growth_rate")))),
            "practice_mastery": _minmax(0.7 * pick("mean_clicks") + 0.3 * pick("studied_credits")),
            "report_quality": _minmax(0.6 * pick("early_click_ratio") + 0.4 * pick("behavior_entropy")),
            "consistency_index": _minmax(1.0 / (1.0 + np.abs(pick("std_clicks")))),
        }
    )

    y_reg = ds["y_reg"].astype(np.float32)
    y_cls = (y_reg >= 60).astype(np.int64)
    return DomainData(name="OULAD", X=aligned[UNIFIED_FEATURES], y_reg=y_reg, y_cls=y_cls)


def build_models(task: str) -> list[tuple[str, object]]:
    if task == "regression":
        models: list[tuple[str, object]] = [
            ("RandomForest", RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)),
            ("MLP", make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42))),
            ("DynamicFusion", DynamicFusionRegressor()),
        ]
        if XGBRegressor is not None:
            models.insert(1, ("XGBoost", XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, objective="reg:squarederror", random_state=42)))
        return models

    models = [
        ("RandomForest", RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)),
        ("MLP", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42))),
        ("DynamicFusion", DynamicFusionClassifier()),
    ]
    if XGBClassifier is not None:
        models.insert(1, ("XGBoost", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")))
    return models


def evaluate_transfer(train_domain: DomainData, test_domain: DomainData) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    X_train = train_domain.X.to_numpy(dtype=np.float32)
    X_test = test_domain.X.to_numpy(dtype=np.float32)

    for model_name, model in build_models("regression"):
        model.fit(X_train, train_domain.y_reg)
        pred_reg = model.predict(X_test)
        mae = mean_absolute_error(test_domain.y_reg, pred_reg)
        rmse = float(np.sqrt(mean_squared_error(test_domain.y_reg, pred_reg)))

        train_classes = np.unique(train_domain.y_cls)
        if train_classes.size < 2:
            pred_cls = np.full(len(X_test), train_classes[0], dtype=np.int64)
        else:
            clf_model = dict(build_models("classification"))[model_name]
            try:
                clf_model.fit(X_train, train_domain.y_cls)
                pred_cls = clf_model.predict(X_test)
            except Exception:  # noqa: BLE001
                majority = int(np.bincount(train_domain.y_cls).argmax())
                pred_cls = np.full(len(X_test), majority, dtype=np.int64)
        acc = accuracy_score(test_domain.y_cls, pred_cls)

        rows.append(
            {
                "train_domain": train_domain.name,
                "test_domain": test_domain.name,
                "model": model_name,
                "accuracy": round(float(acc), 4),
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
            }
        )

    return pd.DataFrame(rows)


def run(self_scores_path: str, data_dir: str, output_path: str) -> pd.DataFrame:
    self_domain = load_self_domain(self_scores_path)
    oulad_domain = load_oulad_domain(data_dir)

    all_results = pd.concat(
        [
            evaluate_transfer(self_domain, self_domain),
            evaluate_transfer(self_domain, oulad_domain),
            evaluate_transfer(oulad_domain, self_domain),
            evaluate_transfer(oulad_domain, oulad_domain),
        ],
        ignore_index=True,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(out, index=False, encoding="utf-8-sig")
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset feature mapping + transfer evaluation")
    parser.add_argument("--self_scores", default="data/student_scores.csv", help="Path to students_scores.csv")
    parser.add_argument("--oulad_data_dir", default="data", help="Directory containing OULAD CSV tables")
    parser.add_argument("--output", default="outputs/cross_dataset_mapping_metrics.csv", help="Result CSV path")
    args = parser.parse_args()

    result = run(self_scores_path=args.self_scores, data_dir=args.oulad_data_dir, output_path=args.output)
    print(result)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
