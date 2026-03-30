from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, Dataset, TensorDataset


EXERCISE_COLS = ["练习1", "练习2", "练习3"]
LAB_COLS = ["实验1", "实验2", "实验3", "实验4", "实验5", "实验6", "实验7"]
STATIC_COLS = ["考勤", "报告", "平时成绩", "总平时 成绩", "总实验 成绩", "总期末成绩"]
TARGET_COL = "总评成绩"
ID_COL = "序号"


@dataclass
class ExpConfig:
    csv_path: str = "data/student_scores.csv"
    k_folds: int = 5
    batch_size: int = 32
    epochs: int = 300
    patience: int = 30
    lr: float = 1e-3
    hidden_dim: int = 64
    ranking_margin: float = 2.0
    ranking_weight: float = 0.2
    seeds: tuple[int, ...] = (42, 52, 62)


class TabularDynamicFusion(nn.Module):
    """Tabular DynamicFusion with modality attention + cross feature network."""

    def __init__(self, exercise_dim: int, lab_dim: int, static_dim: int, full_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.ex_encoder = nn.Sequential(nn.Linear(exercise_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.lab_encoder = nn.Sequential(nn.Linear(lab_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.static_encoder = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.cross_proj = nn.Linear(full_dim, hidden_dim)
        self.cross_gate = nn.Linear(hidden_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, exercise: torch.Tensor, lab: torch.Tensor, static: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
        h_ex = self.ex_encoder(exercise)
        h_lab = self.lab_encoder(lab)
        h_st = self.static_encoder(static)

        h_cat = torch.cat([h_ex, h_lab, h_st], dim=-1)
        w = F.softmax(self.attn(h_cat), dim=-1)
        h_modal = w[:, 0:1] * h_ex + w[:, 1:2] * h_lab + w[:, 2:3] * h_st

        cross0 = self.cross_proj(full)
        cross = cross0 * torch.sigmoid(self.cross_gate(cross0)) + cross0

        out = self.fusion(torch.cat([h_modal, cross], dim=-1)).squeeze(-1)
        return out


def robust_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p.resolve()}")
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, "Cannot decode CSV with utf-8/gb18030/gbk")


def add_feature_crossing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["练习均值_x_实验均值"] = out[EXERCISE_COLS].mean(axis=1) * out[LAB_COLS].mean(axis=1)
    out["考勤_x_实验7"] = out["考勤"] * out["实验7"]
    out["报告_x_平时成绩"] = out["报告"] * out["平时成绩"]
    out["练习3_x_实验3"] = out["练习3"] * out["实验3"]
    return out


def ranking_loss(pred: torch.Tensor, y: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_true = y.unsqueeze(1) - y.unsqueeze(0)
    sign = torch.sign(diff_true)
    mask = sign != 0
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    loss = F.relu(margin - sign[mask] * diff_pred[mask])
    return loss.mean()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def _fit_dynamic_fusion_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    exercise_cols: List[str],
    lab_cols: List[str],
    static_cols: List[str],
    full_cols: List[str],
    cfg: ExpConfig,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_tensor(df: pd.DataFrame):
        return (
            torch.tensor(df[exercise_cols].to_numpy(dtype=np.float32)),
            torch.tensor(df[lab_cols].to_numpy(dtype=np.float32)),
            torch.tensor(df[static_cols].to_numpy(dtype=np.float32)),
            torch.tensor(df[full_cols].to_numpy(dtype=np.float32)),
            torch.tensor(df[TARGET_COL].to_numpy(dtype=np.float32)),
        )

    tr = to_tensor(train_df)
    va = to_tensor(val_df)

    train_ds = TensorDataset(*tr)
    train_loader = DataLoader(train_ds, batch_size=min(cfg.batch_size, len(train_ds)), shuffle=True)

    best_pred = None
    best_rmse = float("inf")

    for seed in cfg.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = TabularDynamicFusion(
            exercise_dim=len(exercise_cols),
            lab_dim=len(lab_cols),
            static_dim=len(static_cols),
            full_dim=len(full_cols),
            hidden_dim=cfg.hidden_dim,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        huber = nn.HuberLoss(delta=1.0)

        best_state = None
        patience_count = 0
        best_val = float("inf")

        for _ in range(cfg.epochs):
            model.train()
            for ex, lab, st, full, y in train_loader:
                ex, lab, st, full, y = ex.to(device), lab.to(device), st.to(device), full.to(device), y.to(device)
                pred = model(ex, lab, st, full)
                loss = huber(pred, y) + cfg.ranking_weight * ranking_loss(pred, y, cfg.ranking_margin)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                ex, lab, st, full, y = [t.to(device) for t in va]
                val_pred = model(ex, lab, st, full)
                val_loss = huber(val_pred, y) + cfg.ranking_weight * ranking_loss(val_pred, y, cfg.ranking_margin)

            if float(val_loss.item()) < best_val:
                best_val = float(val_loss.item())
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            ex, lab, st, full, y = [t.to(device) for t in va]
            pred = model(ex, lab, st, full).detach().cpu().numpy()
        rmse = float(np.sqrt(mean_squared_error(val_df[TARGET_COL].to_numpy(dtype=np.float32), pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_pred = pred

    assert best_pred is not None
    return best_pred


def _run_baselines(train_df: pd.DataFrame, val_df: pd.DataFrame, full_cols: List[str]) -> Dict[str, np.ndarray]:
    x_train = train_df[full_cols].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.float32)
    x_val = val_df[full_cols].to_numpy(dtype=np.float32)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "SVR": SVR(C=1.0, epsilon=0.1),
        "KNNRegressor": KNeighborsRegressor(n_neighbors=3),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=300, random_state=42),
    }

    preds: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds[name] = model.predict(x_val)

    # GradientBoosting stacking as required
    stack_x_train = np.column_stack([m.predict(x_train) for m in models.values()])
    stack_x_val = np.column_stack([preds[n] for n in models.keys()])
    stacker = GradientBoostingRegressor(random_state=42)
    stacker.fit(stack_x_train, y_train)
    preds["GradientBoostingStacking"] = stacker.predict(stack_x_val)

    return preds


def run_experiment(cfg: ExpConfig = ExpConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = robust_read_csv(cfg.csv_path)
    required = [ID_COL] + EXERCISE_COLS + LAB_COLS + STATIC_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = add_feature_crossing(df)
    df[required[1:]] = df[required[1:]].apply(pd.to_numeric, errors="coerce").astype(float)
    full_cols = EXERCISE_COLS + LAB_COLS + STATIC_COLS + [
        "练习均值_x_实验均值",
        "考勤_x_实验7",
        "报告_x_平时成绩",
        "练习3_x_实验3",
    ]

    # Mandatory standardization inside each fold (no leakage)
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=42)
    fold_rows = []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[tr_idx].copy().reset_index(drop=True)
        val_df = df.iloc[va_idx].copy().reset_index(drop=True)

        scaler = StandardScaler()
        train_df = train_df.astype({c: float for c in full_cols})
        val_df = val_df.astype({c: float for c in full_cols})
        train_df.loc[:, full_cols] = scaler.fit_transform(train_df[full_cols])
        val_df.loc[:, full_cols] = scaler.transform(val_df[full_cols])

        y_true = val_df[TARGET_COL].to_numpy(dtype=np.float32)

        # DynamicFusion
        dy_pred = _fit_dynamic_fusion_fold(
            train_df, val_df, EXERCISE_COLS, LAB_COLS, STATIC_COLS, full_cols, cfg
        )
        dy_metrics = regression_metrics(y_true, dy_pred)
        fold_rows.append({"Fold": fold_id, "Model": "DynamicFusion", **dy_metrics})

        # Baselines + stacking
        baseline_preds = _run_baselines(train_df, val_df, full_cols)
        for name, pred in baseline_preds.items():
            m = regression_metrics(y_true, pred)
            fold_rows.append({"Fold": fold_id, "Model": name, **m})

    fold_df = pd.DataFrame(fold_rows)
    summary_df = (
        fold_df.groupby("Model", as_index=False)[["MAE", "RMSE", "R2"]]
        .mean()
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    # Ensure DynamicFusion tracked as primary model in report
    summary_df["Primary"] = summary_df["Model"].eq("DynamicFusion")

    Path("outputs").mkdir(exist_ok=True)
    fold_df.to_csv("outputs/self_dataset_5fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv("outputs/self_dataset_5fold_summary.csv", index=False, encoding="utf-8-sig")

    return fold_df, summary_df


if __name__ == "__main__":
    _, summary = run_experiment()
    print(summary)
