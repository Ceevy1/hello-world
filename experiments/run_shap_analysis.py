from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


def _binary_shap_to_2d(values: np.ndarray | list[np.ndarray]) -> np.ndarray:
    if isinstance(values, list):
        if len(values) == 0:
            return np.empty((0, 0))
        if len(values) == 1:
            return np.asarray(values[0], dtype=float)
        return np.asarray(values[-1], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3 and arr.shape[0] == 2:
        return arr[1]
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[..., 1]
    return arr


def _ensure_dirs(output_dir: Path, figure_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)


def _train_tree_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> tuple[XGBClassifier, CatBoostClassifier]:
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
    )
    xgb.fit(x_train, y_train)

    cat = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=False,
        random_seed=seed,
    )
    cat.fit(x_train, y_train)
    return xgb, cat


def _save_global_shap(
    model,
    model_name: str,
    x_eval: pd.DataFrame,
    output_dir: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(x_eval)
    shap_values = _binary_shap_to_2d(raw_values)

    np.save(output_dir / f"{model_name}_shap_values.npy", shap_values)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x_eval, show=False)
    plt.tight_layout()
    plt.savefig(figure_dir / f"shap_summary_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    ranking = pd.DataFrame(
        {
            "feature": x_eval.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "model": model_name,
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return ranking


def _save_local_case(
    model,
    x_eval: pd.DataFrame,
    y_pred: np.ndarray,
    fail_label: int,
    sample_id_col: str,
    df_eval: pd.DataFrame,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    fail_idx = np.where(y_pred == fail_label)[0]
    case_idx = int(fail_idx[0]) if fail_idx.size > 0 else 0

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(x_eval)
    shap_values = _binary_shap_to_2d(raw_values)

    base = explainer.expected_value
    if isinstance(base, list):
        base_value = float(base[-1])
    else:
        base_arr = np.asarray(base)
        base_value = float(base_arr[-1] if base_arr.ndim > 0 else base_arr)

    contrib = pd.DataFrame(
        {
            "feature": x_eval.columns,
            "feature_value": x_eval.iloc[case_idx].values,
            "shap_value": shap_values[case_idx],
            "abs_shap": np.abs(shap_values[case_idx]),
        }
    ).sort_values("abs_shap", ascending=False, ignore_index=True)

    sample_identifier = int(df_eval.iloc[case_idx][sample_id_col]) if sample_id_col in df_eval.columns else case_idx
    contrib.insert(0, "sample_id", sample_identifier)
    contrib.to_csv(output_dir / "local_case_student_a.csv", index=False)

    plt.figure(figsize=(12, 3.5))
    shap.force_plot(
        base_value=base_value,
        shap_values=shap_values[case_idx],
        features=x_eval.iloc[case_idx],
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(figure_dir / "shap_force_student_a.png", dpi=300, bbox_inches="tight")
    plt.close()


def _resolve_fusion_columns(df: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("tree_weight", "transformer_weight"),
        ("weight_tree", "weight_transformer"),
        ("gate_tree", "gate_transformer"),
    ]
    for tree_col, trans_col in candidates:
        if tree_col in df.columns and trans_col in df.columns:
            return tree_col, trans_col

    if "xgb_weight" in df.columns and "cat_weight" in df.columns and "transformer_weight" in df.columns:
        df["tree_weight"] = df["xgb_weight"] + df["cat_weight"]
        return "tree_weight", "transformer_weight"

    raise ValueError(
        "Cannot find fusion weight columns. Expected one of: "
        "(tree_weight, transformer_weight), (weight_tree, weight_transformer), "
        "(gate_tree, gate_transformer)."
    )


def _save_fusion_plot(
    df: pd.DataFrame,
    week_col: str,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    tree_col, transformer_col = _resolve_fusion_columns(df)

    weekly = (
        df.groupby(week_col, as_index=False)[[tree_col, transformer_col]]
        .mean()
        .rename(columns={tree_col: "tree_weight", transformer_col: "transformer_weight"})
        .sort_values(week_col)
    )
    weekly.to_csv(output_dir / "fusion_weight_by_week.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(weekly[week_col], weekly["tree_weight"], marker="o", label="Tree branch")
    plt.plot(weekly[week_col], weekly["transformer_weight"], marker="o", label="Transformer branch")
    plt.xlabel("Week")
    plt.ylabel("Average fusion weight")
    plt.title("Fusion weight vs week")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "fusion_weight_vs_week.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHAP experiments following experiments/SHAP_EXPERIMENT_DESIGN.md")
    parser.add_argument("--data_csv", type=str, required=True, help="Input csv containing tab_features + target + week + fusion weights")
    parser.add_argument("--target_col", type=str, default="target", help="Binary target column (1=FAIL by default)")
    parser.add_argument("--week_col", type=str, default="week", help="Week index column for fusion plot")
    parser.add_argument("--sample_id_col", type=str, default="student_id", help="Sample identifier column for local explanation")
    parser.add_argument("--fail_label", type=int, default=1, help="Target label treated as FAIL for case study")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="outputs/shap")
    parser.add_argument("--figure_dir", type=str, default="figures")
    parser.add_argument("--save_models", action="store_true", help="Save trained XGB/CAT models into output_dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    _ensure_dirs(output_dir, figure_dir)

    df = pd.read_csv(args.data_csv)
    if args.target_col not in df.columns:
        raise ValueError(f"target column '{args.target_col}' not found in {args.data_csv}")
    if args.week_col not in df.columns:
        raise ValueError(f"week column '{args.week_col}' not found in {args.data_csv}")

    ignored = {
        args.target_col,
        args.week_col,
        args.sample_id_col,
        "tree_weight",
        "transformer_weight",
        "weight_tree",
        "weight_transformer",
        "gate_tree",
        "gate_transformer",
        "xgb_weight",
        "cat_weight",
    }
    feature_cols = [c for c in df.columns if c not in ignored and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("No numeric tab_features found. Please provide numeric feature columns in data_csv.")

    x = df[feature_cols]
    y = df[args.target_col].astype(int)

    x_train, x_test, y_train, _, _, df_test = train_test_split(
        x,
        y,
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    xgb, cat = _train_tree_models(x_train, y_train, seed=args.seed)

    ranking_xgb = _save_global_shap(xgb, "xgb", x_test, output_dir, figure_dir)
    ranking_cat = _save_global_shap(cat, "cat", x_test, output_dir, figure_dir)
    feature_ranking = pd.concat([ranking_xgb, ranking_cat], ignore_index=True)
    feature_ranking.to_csv(output_dir / "feature_ranking.csv", index=False)

    y_pred_xgb = xgb.predict(x_test)
    _save_local_case(
        model=xgb,
        x_eval=x_test,
        y_pred=y_pred_xgb,
        fail_label=args.fail_label,
        sample_id_col=args.sample_id_col,
        df_eval=df_test.reset_index(drop=True),
        output_dir=output_dir,
        figure_dir=figure_dir,
    )

    _save_fusion_plot(df, args.week_col, output_dir, figure_dir)

    if args.save_models:
        with (output_dir / "xgb_model.pkl").open("wb") as f:
            pickle.dump(xgb, f)
        with (output_dir / "cat_model.pkl").open("wb") as f:
            pickle.dump(cat, f)

    print("SHAP analysis completed.")
    print(f"Saved outputs to: {output_dir}")
    print(f"Saved figures to: {figure_dir}")


if __name__ == "__main__":
    main()
