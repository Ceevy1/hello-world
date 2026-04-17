from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engineering.behavior_features import build_behavior_features
from preprocess.oulad_preprocess import preprocess_oulad

BASE_FEATURES = [
    "activity_entropy",
    "active_week_ratio",
    "procrastination_index",
    "resource_switch_rate",
    "avg_session_length",
]


@dataclass
class LeaveOneDomainOutConfig:
    module_col: str = "code_module"
    modules: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")
    max_weeks: int = 40
    random_state: int = 42
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC": _safe_auc(y_true, y_prob),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _extract_student_frame(raw: pd.DataFrame, module_col: str) -> pd.DataFrame:
    required = {"student_id", "activity_type", "week", "final_result", module_col}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    data = preprocess_oulad(raw.copy())
    if "elapsed_time" not in data.columns:
        data["elapsed_time"] = 1.0

    features = build_behavior_features(data, student_col="student_id")
    labels = data.groupby("student_id")["pass_label"].max().reset_index()
    modules = data.groupby("student_id")[module_col].agg(lambda x: x.astype(str).mode().iloc[0]).reset_index()

    merged = (
        features.merge(labels, on="student_id", how="inner")
        .merge(modules, on="student_id", how="inner")
        .sort_values("student_id")
        .reset_index(drop=True)
    )
    merged["student_id"] = merged["student_id"].astype(str)
    merged[module_col] = merged[module_col].astype(str)
    return data, merged


def _build_raw_sequences(data: pd.DataFrame, student_order: List[str], max_weeks: int) -> np.ndarray:
    weekly = (
        data.groupby(["student_id", "week"], as_index=False)
        .agg(
            event_count=("activity_type", "size"),
            unique_activity=("activity_type", "nunique"),
            elapsed_mean=("elapsed_time", "mean"),
            elapsed_std=("elapsed_time", "std"),
        )
        .fillna(0.0)
    )
    weekly["student_id"] = weekly["student_id"].astype(str)

    seq_dim = 4
    seq = np.zeros((len(student_order), max_weeks, seq_dim), dtype=np.float32)
    index = {sid: i for i, sid in enumerate(student_order)}

    for row in weekly.itertuples(index=False):
        sid = str(row.student_id)
        if sid not in index:
            continue
        week_idx = int(row.week) - 1
        if week_idx < 0 or week_idx >= max_weeks:
            continue
        i = index[sid]
        seq[i, week_idx, 0] = float(row.event_count)
        seq[i, week_idx, 1] = float(row.unique_activity)
        seq[i, week_idx, 2] = float(row.elapsed_mean)
        seq[i, week_idx, 3] = float(row.elapsed_std)
    return seq


def _split_and_scale(
    merged: pd.DataFrame,
    seq_raw: np.ndarray,
    test_module: str,
    module_col: str,
) -> dict[str, np.ndarray]:
    train_mask = merged[module_col] != test_module
    test_mask = merged[module_col] == test_module

    if test_mask.sum() == 0:
        raise ValueError(f"No samples found for held-out module: {test_module}")

    x_tab = merged[BASE_FEATURES].to_numpy(dtype=np.float32)
    y = merged["pass_label"].to_numpy(dtype=np.int64)

    x_tab_train_raw = x_tab[train_mask]
    x_tab_test_raw = x_tab[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    tab_scaler = StandardScaler()
    x_tab_train = tab_scaler.fit_transform(x_tab_train_raw)
    x_tab_test = tab_scaler.transform(x_tab_test_raw)

    seq_train_raw = seq_raw[train_mask]
    seq_test_raw = seq_raw[test_mask]

    seq_scaler = StandardScaler()
    seq_train_2d = seq_train_raw.reshape(-1, seq_train_raw.shape[-1])
    seq_test_2d = seq_test_raw.reshape(-1, seq_test_raw.shape[-1])
    seq_train = seq_scaler.fit_transform(seq_train_2d).reshape(seq_train_raw.shape).astype(np.float32)
    seq_test = seq_scaler.transform(seq_test_2d).reshape(seq_test_raw.shape).astype(np.float32)

    return {
        "x_tab_train": x_tab_train,
        "x_tab_test": x_tab_test,
        "x_seq_train": seq_train,
        "x_seq_test": seq_test,
        "y_train": y_train,
        "y_test": y_test,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def _evaluate_models(split: dict[str, np.ndarray], cfg: LeaveOneDomainOutConfig) -> List[Dict[str, float | str]]:
    from experiments.train_validate_oulad_models import (
        DynamicFusionOursClassifier,
        LSTMKTClassifier,
        StaticFusionClassifier,
        TrainConfig,
        VanillaTransformerClassifier,
        _run_torch_model,
    )

    x_tab_train = split["x_tab_train"]
    x_tab_test = split["x_tab_test"]
    x_seq_train = split["x_seq_train"]
    x_seq_test = split["x_seq_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]

    rows: List[Dict[str, float | str]] = []

    single_models = {
        "Logistic Regression": Pipeline([("clf", LogisticRegression(max_iter=2000, random_state=cfg.random_state))]),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=cfg.random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=cfg.random_state),
    }

    for name, model in single_models.items():
        clf = clone(model)
        clf.fit(x_tab_train, y_train)
        prob = clf.predict_proba(x_tab_test)[:, 1]
        rows.append({"Model": name, **_compute_metrics(y_test, prob)})

    torch_cfg = TrainConfig(
        test_size=0.2,
        random_state=cfg.random_state,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        max_weeks=cfg.max_weeks,
    )

    deep_models = {
        "LSTM-KT": LSTMKTClassifier(input_dim=x_seq_train.shape[-1]),
        "Transformer": VanillaTransformerClassifier(seq_dim=x_seq_train.shape[-1]),
        "Static Fusion": StaticFusionClassifier(tab_dim=x_tab_train.shape[-1]),
        "Dynamic Fusion (Ours)": DynamicFusionOursClassifier(seq_dim=x_seq_train.shape[-1], tab_dim=x_tab_train.shape[-1]),
    }

    for name, model in deep_models.items():
        if name == "Static Fusion":
            train_prob, test_prob = _run_torch_model(
                model,
                np.zeros_like(x_seq_train, dtype=np.float32),
                np.zeros_like(x_seq_test, dtype=np.float32),
                x_tab_train,
                x_tab_test,
                y_train,
                torch_cfg,
            )
        else:
            train_prob, test_prob = _run_torch_model(
                model,
                x_seq_train,
                x_seq_test,
                x_tab_train,
                x_tab_test,
                y_train,
                torch_cfg,
            )

        _ = train_prob
        rows.append({"Model": name, **_compute_metrics(y_test, test_prob)})

    return rows


def run_leave_one_domain_out(input_csv: str, output_dir: str, cfg: LeaveOneDomainOutConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(input_csv)
    data, merged = _extract_student_frame(raw, cfg.module_col)

    available_modules = sorted(merged[cfg.module_col].unique().tolist())
    test_modules = [m for m in cfg.modules if m in available_modules]
    if not test_modules:
        raise ValueError(f"None of requested modules {cfg.modules} found in column '{cfg.module_col}'.")

    seq_raw = _build_raw_sequences(data, merged["student_id"].tolist(), cfg.max_weeks)

    result_rows: List[Dict[str, float | str]] = []
    domain_rows: List[Dict[str, float | int | str]] = []

    for held_out in test_modules:
        split = _split_and_scale(merged, seq_raw, held_out, cfg.module_col)

        train_rate = float(split["y_train"].mean())
        test_rate = float(split["y_test"].mean())
        domain_rows.append(
            {
                "held_out_module": held_out,
                "n_train": split["n_train"],
                "n_test": split["n_test"],
                "train_positive_rate": train_rate,
                "test_positive_rate": test_rate,
                "positive_rate_shift": test_rate - train_rate,
            }
        )

        model_rows = _evaluate_models(split, cfg)
        for row in model_rows:
            row["HeldOutModule"] = held_out
            row["TrainModules"] = "+".join([m for m in test_modules if m != held_out])
            result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    summary_df = (
        result_df.groupby("Model", as_index=False)[["AUC", "Accuracy", "F1-Score"]]
        .mean()
        .sort_values("AUC", ascending=False)
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out / "cross_domain_loo_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out / "cross_domain_loo_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(domain_rows).to_csv(out / "cross_domain_domain_shift.csv", index=False, encoding="utf-8-sig")

    return result_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-One-Domain-Out generalization test on OULAD modules.")
    parser.add_argument(
        "--input",
        default="data/oulad_interactions.csv",
        help="Path to OULAD interaction CSV. Default: data/oulad_interactions.csv",
    )
    parser.add_argument("--output-dir", default="outputs/cross_domain", help="Directory to save cross-domain results.")
    parser.add_argument("--module-col", default="code_module", help="Module column name (e.g., code_module).")
    parser.add_argument("--modules", nargs="+", default=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"], help="Target module set for LODO loop.")
    parser.add_argument("--max-weeks", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Default input file not found: {input_path}. "
            "Please pass a valid OULAD csv path via --input."
        )
    config = LeaveOneDomainOutConfig(
        module_col=args.module_col,
        modules=tuple(args.modules),
        max_weeks=args.max_weeks,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    detail, summary = run_leave_one_domain_out(args.input, args.output_dir, config)
    print("Saved detailed metrics:", len(detail))
    print(summary)
