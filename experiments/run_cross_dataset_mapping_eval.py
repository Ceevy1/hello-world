from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

# 统一语义空间（跨数据集共享）
COMMON_FEATURES = ["engagement", "performance", "behavior", "background"]


@dataclass
class DomainData:
    name: str
    X: pd.DataFrame
    y_reg: np.ndarray
    y_cls: np.ndarray


class DynamicFusionRegressor:
    """基于语义特征组权重的轻量动态融合回归模型。"""

    def __init__(self) -> None:
        self.weights_: np.ndarray | None = None
        self.model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32, 16), random_state=42, max_iter=300),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DynamicFusionRegressor":
        # 用单特征 Ridge 拟合误差反推特征组权重（可解释 + 稳定）
        w = []
        for i in range(X.shape[1]):
            reg = Ridge(alpha=1.0).fit(X[:, [i]], y)
            pred = reg.predict(X[:, [i]])
            rmse = float(np.sqrt(mean_squared_error(y, pred)))
            w.append(1.0 / max(rmse, 1e-6))
        self.weights_ = np.asarray(w, dtype=np.float32)
        self.weights_ /= self.weights_.sum()
        self.model.fit(self._fuse(X), y)
        return self

    def _fuse(self, X: np.ndarray) -> np.ndarray:
        assert self.weights_ is not None
        return X * self.weights_[None, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._fuse(X))


class DynamicFusionClassifier:
    """复用动态融合权重的分类模型。"""

    def __init__(self) -> None:
        self.fusion = DynamicFusionRegressor()
        self.clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), random_state=42, max_iter=300),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DynamicFusionClassifier":
        self.fusion.fit(X, y.astype(np.float32))
        self.clf.fit(self.fusion._fuse(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.fusion._fuse(X))


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """标准化列名并自动修复重复列冲突。"""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if out.columns.duplicated().any():
        # 自动修复重复列：同名列聚合求均值
        out = out.T.groupby(level=0).mean().T
    return out


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """安全取列：缺失补零，重复列聚合，统一返回数值 Series。"""
    if col not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)

    values = df.loc[:, col]
    if isinstance(values, pd.DataFrame):
        values = values.apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        values = pd.to_numeric(values, errors="coerce")
    return values.fillna(0.0).astype(np.float32)


def minmax(series: pd.Series | np.ndarray) -> pd.Series:
    x = pd.Series(series).astype(np.float32)
    xmin, xmax = float(x.min()), float(x.max())
    if abs(xmax - xmin) < 1e-12:
        return pd.Series(np.zeros(len(x), dtype=np.float32), index=x.index)
    return (x - xmin) / (xmax - xmin)


def build_self_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = clean_columns(raw)
    missing = [c for c in SELF_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"students_scores.csv 缺失必要列: {missing}")

    exercise_mean = df[["练习1", "练习2", "练习3"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    lab_cols = [f"实验{i}" for i in range(1, 8)]
    lab_values = df[lab_cols].apply(pd.to_numeric, errors="coerce")
    lab_mean = lab_values.mean(axis=1)
    lab_std = lab_values.std(axis=1).fillna(0.0)

    features = pd.DataFrame(index=df.index)
    features["engagement"] = minmax((safe_series(df, "考勤") + safe_series(df, "平时成绩")) / 2.0)
    features["performance"] = minmax((exercise_mean + safe_series(df, "总平时 成绩") + safe_series(df, "总实验 成绩")) / 3.0)
    features["behavior"] = minmax(1.0 / (1.0 + lab_std))
    features["background"] = minmax((safe_series(df, "报告") + safe_series(df, "总期末成绩") + lab_mean) / 3.0)

    y_reg = safe_series(df, "总评成绩").to_numpy(dtype=np.float32)
    y_cls = (y_reg >= 60).astype(np.int64)
    return features[COMMON_FEATURES], y_reg, y_cls


def build_oulad_features(tab: pd.DataFrame) -> pd.DataFrame:
    """按语义映射构造 OULAD 公共特征。"""
    df = clean_columns(tab)
    features = pd.DataFrame(index=df.index)

    # engagement: 交互活跃度
    features["engagement"] = minmax(
        0.7 * safe_series(df, "total_clicks").to_numpy() + 0.3 * safe_series(df, "active_weeks").to_numpy()
    )
    # performance: 学习表现代理（点击质量 + 早期投入）
    features["performance"] = minmax(
        0.5 * safe_series(df, "mean_clicks").to_numpy()
        + 0.3 * safe_series(df, "early_click_ratio").to_numpy()
        + 0.2 * safe_series(df, "max_weekly_clicks").to_numpy()
    )
    # behavior: 稳定行为（波动越小越好）
    features["behavior"] = minmax(
        1.0
        / (
            1.0
            + np.abs(safe_series(df, "click_cv").to_numpy())
            + np.abs(safe_series(df, "growth_rate").to_numpy())
        )
    )
    # background: 学习背景/先验准备
    features["background"] = minmax(
        0.6 * safe_series(df, "studied_credits").to_numpy()
        + 0.4 * safe_series(df, "num_of_prev_attempts").to_numpy()
    )
    return features[COMMON_FEATURES]


def load_self_domain(self_scores_path: str) -> DomainData:
    raw = pd.read_csv(self_scores_path, encoding="utf-8-sig")
    X, y_reg, y_cls = build_self_features(raw)
    return DomainData(name="SelfDataset", X=X, y_reg=y_reg, y_cls=y_cls)


def load_oulad_domain(data_dir: str) -> DomainData:
    ds = OULADDatasetBuilder(data_dir=data_dir).build()
    tab = pd.DataFrame(ds["tabular"], columns=ds["tab_feature_names"])
    X = build_oulad_features(tab)
    y_reg = ds["y_reg"].astype(np.float32)
    y_cls = (y_reg >= 60).astype(np.int64)
    return DomainData(name="OULAD", X=X, y_reg=y_reg, y_cls=y_cls)


def coral_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """CORAL: source covariance alignment to target covariance."""
    eps = 1e-5
    cs = np.cov(source, rowvar=False) + eps * np.eye(source.shape[1])
    ct = np.cov(target, rowvar=False) + eps * np.eye(target.shape[1])

    vals_s, vecs_s = np.linalg.eigh(cs)
    vals_t, vecs_t = np.linalg.eigh(ct)
    cs_inv_sqrt = vecs_s @ np.diag(1.0 / np.sqrt(np.clip(vals_s, eps, None))) @ vecs_s.T
    ct_sqrt = vecs_t @ np.diag(np.sqrt(np.clip(vals_t, eps, None))) @ vecs_t.T
    return (source - source.mean(axis=0)) @ cs_inv_sqrt @ ct_sqrt + target.mean(axis=0)


def build_regressors() -> list[tuple[str, object]]:
    return [
        ("RandomForest", RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)),
        ("MLP", make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42, max_iter=250))),
        ("DynamicFusion", DynamicFusionRegressor()),
    ]


def build_classifiers() -> dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=300)),
        "DynamicFusion": DynamicFusionClassifier(),
    }


def evaluate_transfer(train_domain: DomainData, test_domain: DomainData, adaptation: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    x_train = train_domain.X.to_numpy(dtype=np.float32)
    x_test = test_domain.X.to_numpy(dtype=np.float32)
    y_train_reg, y_test_reg = train_domain.y_reg, test_domain.y_reg
    y_train_cls, y_test_cls = train_domain.y_cls, test_domain.y_cls

    if adaptation == "coral":
        x_train = coral_align(x_train, x_test)

    cls_models = build_classifiers()
    for name, reg_model in build_regressors():
        reg_model.fit(x_train, y_train_reg)
        pred_reg = reg_model.predict(x_test)
        mae = float(mean_absolute_error(y_test_reg, pred_reg))
        rmse = float(np.sqrt(mean_squared_error(y_test_reg, pred_reg)))

        classes = np.unique(y_train_cls)
        if classes.size < 2:
            pred_cls = np.full(len(x_test), classes[0], dtype=np.int64)
        else:
            clf = cls_models[name]
            clf.fit(x_train, y_train_cls)
            pred_cls = clf.predict(x_test)
        acc = float(accuracy_score(y_test_cls, pred_cls))

        rows.append(
            {
                "adaptation": adaptation,
                "train_domain": train_domain.name,
                "test_domain": test_domain.name,
                "model": name,
                "accuracy": round(acc, 4),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
            }
        )
    return pd.DataFrame(rows)


def run(self_scores_path: str, data_dir: str, output_path: str) -> pd.DataFrame:
    self_domain = load_self_domain(self_scores_path)
    oulad_domain = load_oulad_domain(data_dir)

    settings = [
        (self_domain, self_domain),
        (self_domain, oulad_domain),
        (oulad_domain, self_domain),
        (oulad_domain, oulad_domain),
    ]

    all_parts: list[pd.DataFrame] = []
    for adaptation in ("none", "coral"):
        for tr, te in settings:
            all_parts.append(evaluate_transfer(tr, te, adaptation=adaptation))
    result = pd.concat(all_parts, ignore_index=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset mapping + transfer evaluation")
    parser.add_argument("--self_scores", default="data/student_scores.csv")
    parser.add_argument("--oulad_data_dir", default="data")
    parser.add_argument("--output", default="outputs/cross_dataset_mapping_metrics.csv")
    args = parser.parse_args()

    result = run(args.self_scores, args.oulad_data_dir, args.output)
    print(result)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
