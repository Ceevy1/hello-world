from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing import OULADDatasetBuilder
from evaluation.metrics import classification_metrics
from evaluation.statistics import significance_tests
from experiment.explainable.shap_analysis import summarize_shap
from loss.unified_loss import UnifiedLossConfig
from preprocess.data_builder import truncate_sequence
from train.train_full import train_full_pipeline


@dataclass
class UnifiedOutputConfig:
    data_dir: str = "data"
    out_dir: str = "outputs/unified_oulad"
    seed: int = 42
    epochs: int = 50
    patience: int = 8
    k_fold: int = 5


class TabularMonitorNet(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def _to_prob_by_score(pred_score: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(pred_score - 60.0) / 10.0))


def _train_monitor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: UnifiedOutputConfig,
) -> tuple[pd.DataFrame, Dict[str, float], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMonitorNet(input_dim=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    loss_fn = torch.nn.MSELoss()

    x_train_t = torch.FloatTensor(x_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    x_val_t = torch.FloatTensor(x_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    early_stop_epoch = -1
    stale = 0

    rows: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred_train = model(x_train_t)
        train_loss = loss_fn(pred_train, y_train_t)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_t)
            val_loss = loss_fn(pred_val, y_val_t)

        train_np = pred_train.detach().cpu().numpy()
        val_np = pred_val.detach().cpu().numpy()
        train_m = _reg_metrics(y_train, train_np)
        val_m = _reg_metrics(y_val, val_np)
        lr = float(optimizer.param_groups[0]["lr"])

        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss.item()),
                "train_MAE": train_m["MAE"],
                "train_RMSE": train_m["RMSE"],
                "val_MAE": val_m["MAE"],
                "val_RMSE": val_m["RMSE"],
                "learning_rate": lr,
            }
        )

        scheduler.step(float(val_loss.item()))

        if float(val_loss.item()) < best_val:
            best_val = float(val_loss.item())
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                early_stop_epoch = epoch
                break

    if early_stop_epoch < 0:
        early_stop_epoch = len(rows)

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        final_val_pred = model(x_val_t).cpu().numpy()

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val),
        "early_stop_epoch": int(early_stop_epoch),
    }

    return pd.DataFrame(rows), summary, final_val_pred


def run_unified_outputs(cfg: UnifiedOutputConfig) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = OULADDatasetBuilder(data_dir=cfg.data_dir).build()
    x_seq = dataset["sequence"]
    x_tab = dataset["tabular"]
    y_reg = dataset["y_reg"]
    y_cls = (y_reg >= 60).astype(int)
    if np.unique(y_cls).size < 2:
        y_cls = (y_reg >= np.median(y_reg)).astype(int)
    modules = dataset["module"]
    feature_names = dataset.get("tab_feature_names", [f"f{i}" for i in range(x_tab.shape[1])])

    idx = np.arange(len(y_reg))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=cfg.seed, stratify=y_cls)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=cfg.seed, stratify=y_cls[train_idx])

    x_seq_train, x_seq_val, x_seq_test = x_seq[train_idx], x_seq[val_idx], x_seq[test_idx]
    x_tab_train, x_tab_val, x_tab_test = x_tab[train_idx], x_tab[val_idx], x_tab[test_idx]
    y_train, y_val, y_test = y_reg[train_idx], y_reg[val_idx], y_reg[test_idx]

    # 1) Training trace + early stopping summary
    monitor_df, early_stop_info, _ = _train_monitor(x_tab_train, y_train, x_tab_val, y_val, cfg)
    monitor_df.to_csv(out_dir / "training_curve.csv", index=False)
    with (out_dir / "early_stopping.json").open("w", encoding="utf-8") as f:
        json.dump(early_stop_info, f, ensure_ascii=False, indent=2)

    # 2) Main model and mandatory regression/classification/error outputs
    full_out = train_full_pipeline(
        x_seq_train,
        x_tab_train,
        y_train,
        x_seq_test,
        x_tab_test,
        y_test,
        loss_cfg=UnifiedLossConfig(0.1, 0.1, 0.1),
        modules_train=modules[train_idx],
        hafm_epochs=30,
    )
    y_pred_main = full_out.predictions["HAFM"]

    main_metrics = pd.DataFrame([
        {"Model": "Dynamic-HAFM", **_reg_metrics(y_test, y_pred_main)}
    ])
    main_metrics.to_csv(out_dir / "performance_main.csv", index=False)

    cls_df = pd.DataFrame([
        {
            "Model": "Dynamic-HAFM",
            **classification_metrics(y_cls[test_idx], _to_prob_by_score(y_pred_main), threshold=0.5),
        }
    ])
    cls_df.to_csv(out_dir / "classification_metrics.csv", index=False)

    error_df = pd.DataFrame(
        {
            "sample_id": test_idx,
            "y_true": y_test,
            "y_pred": y_pred_main,
            "error": y_pred_main - y_test,
            "abs_error": np.abs(y_pred_main - y_test),
        }
    )
    error_df.to_csv(out_dir / "error_distribution.csv", index=False)

    # 3) Baseline comparison table (required models)
    baseline_rows: List[Dict[str, float | str]] = []
    lr = LinearRegression().fit(x_tab_train, y_train)
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=cfg.seed, n_jobs=-1).fit(x_tab_train, y_train)

    baseline_rows.append({"Model": "LinearRegression", **_reg_metrics(y_test, lr.predict(x_tab_test))})
    baseline_rows.append({"Model": "RandomForest", **_reg_metrics(y_test, rf_reg.predict(x_tab_test))})
    baseline_rows.append({"Model": "Dynamic-HAFM", **_reg_metrics(y_test, y_pred_main)})

    base_df = pd.DataFrame(baseline_rows)
    best_rmse = float(base_df["RMSE"].min())
    base_df["is_best_RMSE"] = base_df["RMSE"].eq(best_rmse)
    base_df["improve_vs_lr_pct"] = (base_df.loc[base_df["Model"] == "LinearRegression", "RMSE"].iloc[0] - base_df["RMSE"]) / base_df.loc[base_df["Model"] == "LinearRegression", "RMSE"].iloc[0] * 100.0
    base_df.to_csv(out_dir / "baseline_comparison.csv", index=False)

    gbc = GradientBoostingClassifier(random_state=cfg.seed).fit(x_tab_train, y_cls[train_idx])
    rf_cls = RandomForestClassifier(n_estimators=200, random_state=cfg.seed, n_jobs=-1).fit(x_tab_train, y_cls[train_idx])
    cls_baselines = pd.DataFrame(
        [
            {"Model": "GradientBoostingClassifier", **classification_metrics(y_cls[test_idx], gbc.predict_proba(x_tab_test)[:, 1])},
            {"Model": "RandomForest", **classification_metrics(y_cls[test_idx], rf_cls.predict_proba(x_tab_test)[:, 1])},
            {"Model": "Dynamic-HAFM", **classification_metrics(y_cls[test_idx], _to_prob_by_score(y_pred_main))},
        ]
    )
    cls_baselines.to_csv(out_dir / "classification_baselines.csv", index=False)

    # 4) Generalization / stability outputs
    split_reg = RandomForestRegressor(n_estimators=200, random_state=cfg.seed, n_jobs=-1).fit(x_tab_train, y_train)
    split_rows = [
        {"split": "train", **_reg_metrics(y_train, split_reg.predict(x_tab_train))},
        {"split": "val", **_reg_metrics(y_val, split_reg.predict(x_tab_val))},
        {"split": "test", **_reg_metrics(y_test, split_reg.predict(x_tab_test))},
    ]
    pd.DataFrame(split_rows).to_csv(out_dir / "split_performance.csv", index=False)

    kf = KFold(n_splits=cfg.k_fold, shuffle=True, random_state=cfg.seed)
    fold_rows = []
    for fold, (tr, te) in enumerate(kf.split(x_tab), start=1):
        rf_fold = RandomForestRegressor(n_estimators=120, random_state=cfg.seed + fold, n_jobs=-1)
        rf_fold.fit(x_tab[tr], y_reg[tr])
        pred_fold = rf_fold.predict(x_tab[te])
        fold_rows.append({"fold": fold, **_reg_metrics(y_reg[te], pred_fold)})
    kfold_df = pd.DataFrame(fold_rows)
    kfold_df.to_csv(out_dir / "kfold_metrics.csv", index=False)
    pd.DataFrame(
        [{"Model": "RandomForest", "MAE_mean": kfold_df["MAE"].mean(), "MAE_std": kfold_df["MAE"].std(ddof=1)}]
    ).to_csv(out_dir / "kfold_summary.csv", index=False)

    # 5) Ablation
    no_transfer = train_full_pipeline(
        x_seq_train,
        x_tab_train,
        y_train,
        x_seq_test,
        x_tab_test,
        y_test,
        loss_cfg=UnifiedLossConfig(0.0, 0.1, 0.1),
        modules_train=modules[train_idx],
        hafm_epochs=30,
    )
    no_diversity = train_full_pipeline(
        x_seq_train,
        x_tab_train,
        y_train,
        x_seq_test,
        x_tab_test,
        y_test,
        loss_cfg=UnifiedLossConfig(0.1, 0.0, 0.1),
        modules_train=modules[train_idx],
        hafm_epochs=30,
    )
    no_stability = train_full_pipeline(
        x_seq_train,
        x_tab_train,
        y_train,
        x_seq_test,
        x_tab_test,
        y_test,
        loss_cfg=UnifiedLossConfig(0.1, 0.1, 0.0),
        modules_train=modules[train_idx],
        hafm_epochs=30,
    )
    ablation_df = pd.DataFrame(
        [
            {"Model Variant": "Full Model", **_reg_metrics(y_test, y_pred_main)},
            {"Model Variant": "-Transfer", **_reg_metrics(y_test, no_transfer.predictions["HAFM"])},
            {"Model Variant": "-Diversity", **_reg_metrics(y_test, no_diversity.predictions["HAFM"])},
            {"Model Variant": "-Stability", **_reg_metrics(y_test, no_stability.predictions["HAFM"])},
        ]
    )
    ablation_df.to_csv(out_dir / "ablation_results.csv", index=False)

    # 6) Significance
    sig_df = pd.DataFrame(
        [
            {"Comparison": "Dynamic-HAFM vs LinearRegression", **significance_tests(y_test, y_pred_main, lr.predict(x_tab_test))},
            {"Comparison": "Dynamic-HAFM vs RandomForest", **significance_tests(y_test, y_pred_main, rf_reg.predict(x_tab_test))},
        ]
    )
    sig_df.to_csv(out_dir / "significance_tests.csv", index=False)

    # 7) SHAP-like interpretability + top10 + individual explanation
    shap_result = summarize_shap(rf_reg, x_tab_test, feature_names=list(feature_names), save_dir=out_dir)
    importance_df = shap_result["importance"]
    top10 = importance_df.head(10)
    top10.to_csv(out_dir / "shap_top10.csv", index=False)

    sample_row = np.abs(error_df["error"]).sort_values(ascending=False).index[0]
    sample_id = int(error_df.loc[sample_row, "sample_id"])
    contrib = []
    centered = x_tab_test[sample_row] - x_tab_test.mean(axis=0)
    for name, value in zip(feature_names, centered):
        contrib.append({"sample_id": sample_id, "feature": name, "feature_contribution": float(value)})
    pd.DataFrame(contrib).sort_values("feature_contribution", key=lambda s: s.abs(), ascending=False).head(15).to_csv(
        out_dir / "individual_explanations.csv", index=False
    )

    # 8) Score-band and early prediction behavior analysis
    bands = [(-np.inf, 60, "<60"), (60, 80, "60-80"), (80, np.inf, ">80")]
    band_rows = []
    for low, high, name in bands:
        mask = (y_test >= low) & (y_test < high)
        if mask.sum() == 0:
            continue
        band_rows.append({"Score Range": name, "MAE": float(mean_absolute_error(y_test[mask], y_pred_main[mask]))})
    pd.DataFrame(band_rows).to_csv(out_dir / "score_range_performance.csv", index=False)

    early_rows = []
    for week in [4, 8, x_seq.shape[1]]:
        x_week = truncate_sequence(x_seq, week if week != x_seq.shape[1] else None)
        model_week = train_full_pipeline(
            x_week[train_idx],
            x_tab_train,
            y_train,
            x_week[test_idx],
            x_tab_test,
            y_test,
            loss_cfg=UnifiedLossConfig(0.1, 0.1, 0.1),
            modules_train=modules[train_idx],
            hafm_epochs=20,
        )
        early_rows.append({"Window": f"Week {week}" if week != x_seq.shape[1] else "Full", **_reg_metrics(y_test, model_week.predictions["HAFM"])})
    pd.DataFrame(early_rows).to_csv(out_dir / "early_prediction_results.csv", index=False)

    # 9) Unified startup manifest
    manifest = {
        "training_curve": "training_curve.csv",
        "early_stop": "early_stopping.json",
        "performance": "performance_main.csv",
        "baseline_comparison": "baseline_comparison.csv",
        "ablation": "ablation_results.csv",
        "kfold": "kfold_metrics.csv",
        "kfold_summary": "kfold_summary.csv",
        "classification": "classification_metrics.csv",
        "classification_baselines": "classification_baselines.csv",
        "errors": "error_distribution.csv",
        "significance": "significance_tests.csv",
        "shap_importance": "shap_importance.csv",
        "shap_top10": "shap_top10.csv",
        "individual_explanations": "individual_explanations.csv",
        "score_range": "score_range_performance.csv",
        "early_prediction": "early_prediction_results.csv",
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified launcher for OULAD experiment outputs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs/unified_oulad")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--k-fold", type=int, default=5)
    args = parser.parse_args()

    run_unified_outputs(
        UnifiedOutputConfig(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            k_fold=args.k_fold,
        )
    )
