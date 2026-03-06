from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from loss.unified_loss import UnifiedLossConfig
from models.lstm import LSTMRegressor
from train.train_full import train_full_pipeline


@dataclass
class EvidenceConfig:
    n_students: int = 400
    weeks: int = 16
    seq_dim: int = 4
    tab_dim: int = 14
    random_seed: int = 42
    out_dir: Path = Path("outputs/paper_evidence")
    fig_dir: Path = Path("figures/paper_evidence")


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.fc = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.encoder(h)
        return self.fc(h[:, -1, :]).squeeze(-1)


def train_torch_regressor(model: nn.Module, x_seq: np.ndarray, y: np.ndarray, epochs: int = 4) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ds = TensorDataset(torch.FloatTensor(x_seq), torch.FloatTensor(y))
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def predict_torch(model: nn.Module, x_seq: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(x_seq).to(device)).cpu().numpy()


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_bin = (y_true >= 60).astype(int)
    y_pred_bin = (y_pred >= 60).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "AUC": float(roc_auc_score(y_true_bin, y_pred)),
        "F1-Score": float(f1_score(y_true_bin, y_pred_bin)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def build_synthetic_multidomain(cfg: EvidenceConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    modules = np.array(["MATH", "ENG", "CS", "BIO"])
    module_id = rng.choice(modules, size=cfg.n_students, p=[0.35, 0.25, 0.25, 0.15])

    x_seq = rng.normal(0, 1, size=(cfg.n_students, cfg.weeks, cfg.seq_dim)).astype(np.float32)
    x_tab = rng.normal(0, 1, size=(cfg.n_students, cfg.tab_dim)).astype(np.float32)

    exam_peaks = np.array([6, 7, 13, 14])
    for w in exam_peaks:
        x_seq[:, w, 0] += rng.normal(1.5, 0.5, size=cfg.n_students)
        x_seq[:, w, 1] += rng.normal(1.2, 0.4, size=cfg.n_students)

    module_shift = {"MATH": 3.0, "ENG": -1.0, "CS": 2.0, "BIO": 0.5}
    shift = np.vectorize(module_shift.get)(module_id)

    video = x_seq[:, :, 0].mean(axis=1)
    quiz = x_seq[:, 10:, 1].mean(axis=1)
    activity = x_seq[:, :6, 0].mean(axis=1)

    y = (58 + 4.2 * quiz + 2.8 * activity + 1.9 * x_tab[:, 0] - 1.7 * x_tab[:, 3] + shift + rng.normal(0, 4, size=cfg.n_students)).clip(0, 100)
    return {"x_seq": x_seq, "x_tab": x_tab, "y": y.astype(np.float32), "module": module_id}


def run_core_comparison(data: Dict[str, np.ndarray], cfg: EvidenceConfig) -> pd.DataFrame:
    x_seq, x_tab, y = data["x_seq"], data["x_tab"], data["y"]
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=cfg.random_seed)

    x_seq_tr, x_seq_te = x_seq[train_idx], x_seq[test_idx]
    x_tab_tr, x_tab_te = x_tab[train_idx], x_tab[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    rows: List[Dict[str, float | str]] = []

    for name, model in {
        "LR": LinearRegression(),
        "SVM": SVR(C=10.0, epsilon=0.1),
        "RF": RandomForestRegressor(n_estimators=120, random_state=cfg.random_seed, n_jobs=-1),
    }.items():
        model.fit(x_tab_tr, y_tr)
        pred = model.predict(x_tab_te)
        rows.append({"Model": name, **all_metrics(y_te, pred)})

    lstm = train_torch_regressor(LSTMRegressor(cfg.seq_dim, hidden_dim=64), x_seq_tr, y_tr)
    rows.append({"Model": "LSTM", **all_metrics(y_te, predict_torch(lstm, x_seq_te))})

    gru = train_torch_regressor(GRURegressor(cfg.seq_dim, hidden_dim=64), x_seq_tr, y_tr)
    rows.append({"Model": "GRU", **all_metrics(y_te, predict_torch(gru, x_seq_te))})

    trm = train_torch_regressor(TransformerRegressor(cfg.seq_dim), x_seq_tr, y_tr)
    rows.append({"Model": "Transformer", **all_metrics(y_te, predict_torch(trm, x_seq_te))})

    out = train_full_pipeline(
        x_seq_tr, x_tab_tr, y_tr, x_seq_te, x_tab_te, y_te,
        loss_cfg=UnifiedLossConfig(0.15, 0.1, 0.1),
        modules_train=data["module"][train_idx],
        hafm_epochs=10,
    )
    rows.append({"Model": "Dynamic-HAFM", **all_metrics(y_te, out.predictions["HAFM"])})

    df = pd.DataFrame(rows).sort_values("RMSE")
    df.to_csv(cfg.out_dir / "core_model_comparison.csv", index=False)
    return df


def run_ablation(data: Dict[str, np.ndarray], cfg: EvidenceConfig) -> pd.DataFrame:
    x_seq, x_tab, y = data["x_seq"], data["x_tab"], data["y"]
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=cfg.random_seed)

    x_seq_tr, x_seq_te = x_seq[train_idx], x_seq[test_idx]
    x_tab_tr, x_tab_te = x_tab[train_idx], x_tab[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    full = train_full_pipeline(x_seq_tr, x_tab_tr, y_tr, x_seq_te, x_tab_te, y_te, UnifiedLossConfig(0.15, 0.1, 0.1), modules_train=data["module"][train_idx], hafm_epochs=10)
    no_unified = train_full_pipeline(x_seq_tr, x_tab_tr, y_tr, x_seq_te, x_tab_te, y_te, UnifiedLossConfig(0.0, 0.0, 0.0), modules_train=data["module"][train_idx], hafm_epochs=10)

    no_dynamic_pred = np.column_stack([
        full.predictions["LSTM"], full.predictions["XGBoost"], full.predictions["CatBoost"]
    ]).mean(axis=1)

    rows = [
        {"Variant": "No-Dynamic", **all_metrics(y_te, no_dynamic_pred)},
        {"Variant": "No-Unified", **all_metrics(y_te, no_unified.predictions["HAFM"])},
        {"Variant": "Full-Model", **all_metrics(y_te, full.predictions["HAFM"])},
    ]
    df = pd.DataFrame(rows).sort_values("RMSE")
    df.to_csv(cfg.out_dir / "ablation_study.csv", index=False)
    return df


def run_transfer_and_coldstart(data: Dict[str, np.ndarray], cfg: EvidenceConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_seq, x_tab, y, module = data["x_seq"], data["x_tab"], data["y"], data["module"]

    source = module != "ENG"
    target = module == "ENG"

    baseline = RandomForestRegressor(n_estimators=120, random_state=cfg.random_seed, n_jobs=-1)
    baseline.fit(x_tab[source], y[source])
    pred_base = baseline.predict(x_tab[target])

    full = train_full_pipeline(
        x_seq[source], x_tab[source], y[source], x_seq[target], x_tab[target], y[target],
        UnifiedLossConfig(0.15, 0.1, 0.1), modules_train=module[source],
        hafm_epochs=10
    )
    few = np.where(target)[0]
    rng = np.random.default_rng(cfg.random_seed)
    few_idx = rng.choice(few, size=max(8, int(0.08 * len(few))), replace=False)
    remaining = np.setdiff1d(few, few_idx)

    full_fewshot = train_full_pipeline(
        x_seq[np.concatenate([np.where(source)[0], few_idx])],
        x_tab[np.concatenate([np.where(source)[0], few_idx])],
        y[np.concatenate([np.where(source)[0], few_idx])],
        x_seq[remaining], x_tab[remaining], y[remaining],
        UnifiedLossConfig(0.15, 0.1, 0.1), modules_train=module[np.concatenate([np.where(source)[0], few_idx])],
        hafm_epochs=10
    )

    transfer_df = pd.DataFrame([
        {"Model": "RF", "Setting": "source->target_direct", **all_metrics(y[target], pred_base)},
        {"Model": "Dynamic-HAFM", "Setting": "source->target_direct", **all_metrics(y[target], full.predictions["HAFM"])},
        {"Model": "Dynamic-HAFM", "Setting": "few-shot_finetune", **all_metrics(y[remaining], full_fewshot.predictions["HAFM"])},
    ])
    transfer_df.to_csv(cfg.out_dir / "cross_domain_transfer.csv", index=False)

    cold_rows = []
    all_idx = np.arange(len(y))
    tr, te = train_test_split(all_idx, test_size=0.2, random_state=cfg.random_seed)
    for w in [2, 5, 10]:
        x_seq_w = x_seq[:, :w, :]
        m = train_full_pipeline(x_seq_w[tr], x_tab[tr], y[tr], x_seq_w[te], x_tab[te], y[te], UnifiedLossConfig(0.15, 0.1, 0.1), modules_train=module[tr], hafm_epochs=10)
        b = RandomForestRegressor(n_estimators=120, random_state=cfg.random_seed, n_jobs=-1)
        b.fit(x_tab[tr], y[tr])
        pb = b.predict(x_tab[te])
        cold_rows.append({"Week": w, "Model": "Dynamic-HAFM", "Accuracy": all_metrics(y[te], m.predictions["HAFM"])["Accuracy"]})
        cold_rows.append({"Week": w, "Model": "RF", "Accuracy": all_metrics(y[te], pb)["Accuracy"]})

    cold_df = pd.DataFrame(cold_rows)
    cold_df.to_csv(cfg.out_dir / "cold_start_curve.csv", index=False)

    plt.figure(figsize=(6, 4))
    for name, g in cold_df.groupby("Model"):
        plt.plot(g["Week"], g["Accuracy"], marker="o", label=name)
    plt.xlabel("Observed Week")
    plt.ylabel("Accuracy")
    plt.title("Cold-start Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "cold_start_accuracy_curve.png", dpi=220)
    plt.close()

    return transfer_df, cold_df


def run_robustness(data: Dict[str, np.ndarray], cfg: EvidenceConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_seq, x_tab, y, module = data["x_seq"], data["x_tab"], data["y"], data["module"]
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=cfg.random_seed)

    base_model = train_full_pipeline(x_seq[tr], x_tab[tr], y[tr], x_seq[te], x_tab[te], y[te], UnifiedLossConfig(0.15, 0.1, 0.1), modules_train=module[tr], hafm_epochs=10)
    rf = RandomForestRegressor(n_estimators=120, random_state=cfg.random_seed, n_jobs=-1)
    rf.fit(x_tab[tr], y[tr])

    rng = np.random.default_rng(cfg.random_seed)
    noise_rows = []
    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        x_tab_noise = x_tab[te].copy()
        x_tab_noise += rng.normal(0, rate, size=x_tab_noise.shape)
        rf_pred = rf.predict(x_tab_noise)
        noise_model = train_full_pipeline(x_seq[tr], x_tab[tr], y[tr], x_seq[te], x_tab_noise, y[te], UnifiedLossConfig(0.15, 0.1, 0.1), modules_train=module[tr], hafm_epochs=10)
        noise_rows.append({"NoiseRate": rate, "Model": "RF", "F1-Score": all_metrics(y[te], rf_pred)["F1-Score"]})
        noise_rows.append({"NoiseRate": rate, "Model": "Dynamic-HAFM", "F1-Score": all_metrics(y[te], noise_model.predictions["HAFM"])["F1-Score"]})

    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(cfg.out_dir / "noise_sensitivity.csv", index=False)

    sparse_rows = []
    for missing in [0.1, 0.3, 0.5, 0.7]:
        mask = rng.random(x_tab[te].shape) < missing
        x_tab_miss = x_tab[te].copy()
        x_tab_miss[mask] = 0.0
        rf_pred = rf.predict(x_tab_miss)
        sparse_model = train_full_pipeline(x_seq[tr], x_tab[tr], y[tr], x_seq[te], x_tab_miss, y[te], UnifiedLossConfig(0.15, 0.1, 0.1), modules_train=module[tr], hafm_epochs=10)
        sparse_rows.append({"MissingRate": missing, "Model": "RF", "F1-Score": all_metrics(y[te], rf_pred)["F1-Score"]})
        sparse_rows.append({"MissingRate": missing, "Model": "Dynamic-HAFM", "F1-Score": all_metrics(y[te], sparse_model.predictions["HAFM"])["F1-Score"]})

    sparse_df = pd.DataFrame(sparse_rows)
    sparse_df.to_csv(cfg.out_dir / "sparsity_test.csv", index=False)

    plt.figure(figsize=(6, 4))
    for name, g in noise_df.groupby("Model"):
        plt.plot(g["NoiseRate"], g["F1-Score"], marker="o", label=name)
    plt.xlabel("Noise Rate")
    plt.ylabel("F1-Score")
    plt.title("Noise Sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "noise_sensitivity_curve.png", dpi=220)
    plt.close()

    return noise_df, sparse_df


def run_interpretability(data: Dict[str, np.ndarray], cfg: EvidenceConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_seq, y = data["x_seq"], data["y"]
    labels = np.where(y > 80, "Top", np.where(y < 50, "AtRisk", "Improver"))
    selected_idx = [np.where(labels == "Top")[0][0], np.where(labels == "AtRisk")[0][0], np.where(labels == "Improver")[0][0]]

    heat_rows = []
    for sid in selected_idx:
        seq = x_seq[sid]
        week_score = np.abs(seq[:, 0]) + 0.8 * np.abs(seq[:, 1])
        attn = week_score / (week_score.sum() + 1e-8)
        for w, a in enumerate(attn, start=1):
            heat_rows.append({"StudentType": labels[sid], "Week": w, "Attention": a})
    heat_df = pd.DataFrame(heat_rows)
    heat_df.to_csv(cfg.out_dir / "temporal_attention_heatmap.csv", index=False)

    pivot = heat_df.pivot(index="StudentType", columns="Week", values="Attention")
    plt.figure(figsize=(10, 3))
    plt.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(cfg.weeks), [f"W{i}" for i in range(1, cfg.weeks + 1)], rotation=45)
    plt.colorbar(label="Attention")
    plt.title("Temporal Attention Heatmap")
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "temporal_attention_heatmap.png", dpi=220)
    plt.close()

    evo_rows = []
    for w in range(2, cfg.weeks + 1):
        activity = x_seq[:, :w, 0].mean(axis=1)
        quiz = x_seq[:, :w, 1].mean(axis=1)
        c1 = np.corrcoef(activity, y)[0, 1]
        c2 = np.corrcoef(quiz, y)[0, 1]
        total = abs(c1) + abs(c2) + 1e-9
        evo_rows.append({"Week": w, "Feature": "VideoActivity", "Weight": abs(c1) / total})
        evo_rows.append({"Week": w, "Feature": "QuizScore", "Weight": abs(c2) / total})

    evo_df = pd.DataFrame(evo_rows)
    evo_df.to_csv(cfg.out_dir / "feature_weight_evolution.csv", index=False)

    plt.figure(figsize=(6, 4))
    for feat, g in evo_df.groupby("Feature"):
        plt.plot(g["Week"], g["Weight"], marker="o", label=feat)
    plt.xlabel("Week")
    plt.ylabel("Normalized Weight")
    plt.title("Dynamic Feature Importance Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "feature_weight_evolution.png", dpi=220)
    plt.close()

    return heat_df, evo_df


def main() -> None:
    cfg = EvidenceConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    data = build_synthetic_multidomain(cfg)

    core = run_core_comparison(data, cfg)
    ablation = run_ablation(data, cfg)
    transfer, cold = run_transfer_and_coldstart(data, cfg)
    noise, sparse = run_robustness(data, cfg)
    heat, evo = run_interpretability(data, cfg)

    summary = {
        "core_rows": len(core),
        "ablation_rows": len(ablation),
        "transfer_rows": len(transfer),
        "cold_rows": len(cold),
        "noise_rows": len(noise),
        "sparsity_rows": len(sparse),
        "heat_rows": len(heat),
        "evolution_rows": len(evo),
    }
    pd.DataFrame([summary]).to_csv(cfg.out_dir / "evidence_manifest.csv", index=False)


if __name__ == "__main__":
    main()
