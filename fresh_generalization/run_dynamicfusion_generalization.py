import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Config:
    self_data_path: str
    oulad_data_path: str | None
    output_dir: str
    epochs_encoder: int
    epochs_model: int
    epochs_finetune: int
    batch_size: int
    lr: float
    latent_dim: int
    hidden_dim: int
    test_size: float
    min_samples_for_training: int


class FeatureAligner:
    """Map heterogeneous columns to a shared semantic space."""

    MANUAL_MAP_SELF = {
        "考勤": "engagement_attendance",
        "练习1": "engagement_quiz1",
        "练习2": "engagement_quiz2",
        "练习3": "engagement_quiz3",
        "总平时 成绩": "performance_coursework_total",
        "实验1": "lab_1",
        "实验2": "lab_2",
        "实验3": "lab_3",
        "实验4": "lab_4",
        "实验5": "lab_5",
        "实验6": "lab_6",
        "实验7": "lab_7",
        "报告": "lab_report",
        "总实验 成绩": "lab_total",
        "平时成绩": "performance_regular",
        "总期末成绩": "performance_final_exam",
        "总评成绩": "target_final",
    }

    MANUAL_MAP_OULAD = {
        "studied_credits": "background_credits",
        "num_of_prev_attempts": "background_prev_attempts",
        "final_result": "target_final",
        "score": "performance_final_exam",
        "sum_click": "engagement_attendance",
    }

    BRANCHES = {
        "engagement": [
            "engagement_attendance",
            "engagement_quiz1",
            "engagement_quiz2",
            "engagement_quiz3",
        ],
        "behavior": [
            "lab_1",
            "lab_2",
            "lab_3",
            "lab_4",
            "lab_5",
            "lab_6",
            "lab_7",
            "lab_report",
            "lab_total",
        ],
        "performance": [
            "performance_coursework_total",
            "performance_regular",
            "performance_final_exam",
        ],
    }

    def transform(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        mapped = self.MANUAL_MAP_SELF if domain == "self" else self.MANUAL_MAP_OULAD
        out = pd.DataFrame(index=df.index)
        for src, tgt in mapped.items():
            if src in df.columns:
                out[tgt] = pd.to_numeric(df[src], errors="coerce")

        # Fill missing branch columns with 0 for semantic alignment
        for cols in self.BRANCHES.values():
            for col in cols:
                if col not in out.columns:
                    out[col] = 0.0

        if "target_final" not in out.columns:
            raise ValueError("target_final label is required after feature alignment")

        out = out.fillna(out.median(numeric_only=True)).fillna(0.0)
        return out

    def split_branches(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_eng = df[self.BRANCHES["engagement"]].to_numpy(dtype=np.float32)
        x_beh = df[self.BRANCHES["behavior"]].to_numpy(dtype=np.float32)
        x_perf = df[self.BRANCHES["performance"]].to_numpy(dtype=np.float32)
        y = df["target_final"].to_numpy(dtype=np.float32)
        return x_eng, x_beh, x_perf, y


def maybe_augment_small_dataset(df: pd.DataFrame, min_samples: int, target_col: str = "target_final") -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if len(df) >= min_samples:
        return df
    needed = min_samples - len(df)
    rows = []
    numeric_cols = [c for c in df.columns if c != target_col]
    for _ in range(needed):
        row = df.sample(1, replace=True, random_state=np.random.randint(0, 10_000)).iloc[0].astype(float).copy()
        for c in numeric_cols:
            std = max(df[c].std(ddof=0), 1.0)
            row[c] = float(row[c]) + np.random.normal(0, 0.05 * std)
        row[target_col] = float(np.clip(row[target_col] + np.random.normal(0, 2.0), 0, 100))
        rows.append(row)
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    d = source.shape[1]
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    c_s = (source_centered.T @ source_centered) / max(source.shape[0] - 1, 1)
    c_t = (target_centered.T @ target_centered) / max(target.shape[0] - 1, 1)
    return torch.sum((c_s - c_t) ** 2) / (4 * d * d)


class DynamicFusion(nn.Module):
    def __init__(self, d_eng: int, d_beh: int, d_perf: int, hidden_dim: int) -> None:
        super().__init__()
        self.e_eng = nn.Linear(d_eng, hidden_dim)
        self.e_beh = nn.Linear(d_beh, hidden_dim)
        self.e_perf = nn.Linear(d_perf, hidden_dim)
        self.attn = nn.Sequential(nn.Linear(hidden_dim * 3, 3), nn.Softmax(dim=1))
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x_eng: torch.Tensor, x_beh: torch.Tensor, x_perf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_eng = torch.relu(self.e_eng(x_eng))
        h_beh = torch.relu(self.e_beh(x_beh))
        h_perf = torch.relu(self.e_perf(x_perf))
        h = torch.cat([h_eng, h_beh, h_perf], dim=1)
        w = self.attn(h)
        fused = w[:, 0:1] * h_eng + w[:, 1:2] * h_beh + w[:, 2:3] * h_perf
        pred = self.head(fused).squeeze(1)
        return pred, w


def make_loader(x_eng, x_beh, x_perf, y, batch_size: int, shuffle: bool = True):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_eng),
        torch.tensor(x_beh),
        torch.tensor(x_perf),
        torch.tensor(y),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle)


def pretrain_encoder(encoder: SharedEncoder, x_source: np.ndarray, x_target: np.ndarray, epochs: int, lr: float, batch_size: int):
    optim = torch.optim.Adam(encoder.parameters(), lr=lr)
    src_tensor = torch.tensor(x_source)
    tgt_tensor = torch.tensor(x_target)
    for _ in range(epochs):
        idx_src = torch.randint(0, len(src_tensor), (min(batch_size, len(src_tensor)),))
        idx_tgt = torch.randint(0, len(tgt_tensor), (min(batch_size, len(tgt_tensor)),))
        src_batch = src_tensor[idx_src]
        tgt_batch = tgt_tensor[idx_tgt]
        z_s, recon_s = encoder(src_batch)
        z_t, recon_t = encoder(tgt_batch)
        loss_recon = F.mse_loss(recon_s, src_batch) + F.mse_loss(recon_t, tgt_batch)
        loss_align = coral_loss(z_s, z_t)
        loss = loss_recon + 0.5 * loss_align
        optim.zero_grad()
        loss.backward()
        optim.step()


def fit_dynamic_fusion(model, train_loader, val_loader, epochs: int, lr: float):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        for x1, x2, x3, y in train_loader:
            pred, _ = model(x1, x2, x3)
            loss = F.mse_loss(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for x1, x2, x3, y in val_loader:
                pred, _ = model(x1, x2, x3)
                losses.append(F.mse_loss(pred, y).item())
        val_loss = float(np.mean(losses)) if losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)


def evaluate_model(model, x_eng, x_beh, x_perf, y_true):
    model.eval()
    with torch.no_grad():
        pred, attn = model(torch.tensor(x_eng), torch.tensor(x_beh), torch.tensor(x_perf))
    y_pred = pred.numpy()
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
    return metrics, y_pred, attn.numpy()


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000) -> Tuple[float, float]:
    boots = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        boots.append(np.mean(sample))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_distribution_shift(source_df: pd.DataFrame, target_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        rows.append({
            "feature": c,
            "wasserstein": float(wasserstein_distance(source_df[c], target_df[c])),
        })
    return pd.DataFrame(rows).sort_values("wasserstein", ascending=False)


def build_source_from_self(aligned_self: pd.DataFrame) -> pd.DataFrame:
    src = aligned_self.copy()
    for c in src.columns:
        if c == "target_final":
            continue
        src[c] = src[c] * np.random.uniform(0.85, 1.15) + np.random.normal(0, max(src[c].std(ddof=0), 1.0) * 0.1, size=len(src))
    src["target_final"] = np.clip(src["target_final"] + np.random.normal(0, 4.0, size=len(src)), 0, 100)
    return src


def run_experiment(cfg: Config) -> Dict:
    set_seed()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aligner = FeatureAligner()

    raw_self = pd.read_csv(cfg.self_data_path)
    self_aligned = aligner.transform(raw_self, domain="self")
    self_aligned = maybe_augment_small_dataset(self_aligned, cfg.min_samples_for_training)

    if cfg.oulad_data_path and os.path.exists(cfg.oulad_data_path):
        raw_oulad = pd.read_csv(cfg.oulad_data_path)
        source_aligned = aligner.transform(raw_oulad, domain="oulad")
        source_note = "real_oulad"
    else:
        source_aligned = build_source_from_self(self_aligned)
        source_note = "synthetic_proxy_from_self"

    # standardize feature space consistently
    feature_cols = [c for c in self_aligned.columns if c != "target_final"]
    scaler = StandardScaler()
    scaler.fit(pd.concat([source_aligned[feature_cols], self_aligned[feature_cols]], axis=0))
    source_aligned[feature_cols] = scaler.transform(source_aligned[feature_cols])
    self_aligned[feature_cols] = scaler.transform(self_aligned[feature_cols])

    # split target domain
    tgt_train_df, tgt_test_df = train_test_split(self_aligned, test_size=cfg.test_size, random_state=SEED)
    tgt_train_df, tgt_val_df = train_test_split(tgt_train_df, test_size=0.2, random_state=SEED)
    src_train_df, src_val_df = train_test_split(source_aligned, test_size=0.2, random_state=SEED)

    # pretrain shared encoder on concatenated features
    enc_input = len(feature_cols)
    encoder = SharedEncoder(enc_input, cfg.latent_dim, cfg.hidden_dim)
    pretrain_encoder(
        encoder,
        src_train_df[feature_cols].to_numpy(dtype=np.float32),
        tgt_train_df[feature_cols].to_numpy(dtype=np.float32),
        epochs=cfg.epochs_encoder,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
    )

    # Prepare branch-wise tensors for DynamicFusion
    x_eng_src_tr, x_beh_src_tr, x_perf_src_tr, y_src_tr = aligner.split_branches(src_train_df)
    x_eng_src_va, x_beh_src_va, x_perf_src_va, y_src_va = aligner.split_branches(src_val_df)

    x_eng_tgt_tr, x_beh_tgt_tr, x_perf_tgt_tr, y_tgt_tr = aligner.split_branches(tgt_train_df)
    x_eng_tgt_te, x_beh_tgt_te, x_perf_tgt_te, y_tgt_te = aligner.split_branches(tgt_test_df)

    model = DynamicFusion(
        d_eng=x_eng_src_tr.shape[1],
        d_beh=x_beh_src_tr.shape[1],
        d_perf=x_perf_src_tr.shape[1],
        hidden_dim=cfg.hidden_dim,
    )

    src_train_loader = make_loader(x_eng_src_tr, x_beh_src_tr, x_perf_src_tr, y_src_tr, cfg.batch_size)
    src_val_loader = make_loader(x_eng_src_va, x_beh_src_va, x_perf_src_va, y_src_va, cfg.batch_size, shuffle=False)
    fit_dynamic_fusion(model, src_train_loader, src_val_loader, cfg.epochs_model, cfg.lr)

    zero_shot_metrics, _, zero_attn = evaluate_model(model, x_eng_tgt_te, x_beh_tgt_te, x_perf_tgt_te, y_tgt_te)

    # Fine-tune on target train
    tgt_train_loader = make_loader(x_eng_tgt_tr, x_beh_tgt_tr, x_perf_tgt_tr, y_tgt_tr, cfg.batch_size)
    x_eng_tgt_va, x_beh_tgt_va, x_perf_tgt_va, y_tgt_va = aligner.split_branches(tgt_val_df)
    tgt_val_loader = make_loader(x_eng_tgt_va, x_beh_tgt_va, x_perf_tgt_va, y_tgt_va, cfg.batch_size, shuffle=False)
    fit_dynamic_fusion(model, tgt_train_loader, tgt_val_loader, cfg.epochs_finetune, cfg.lr * 0.5)
    finetuned_metrics, y_pred_tgt, ft_attn = evaluate_model(model, x_eng_tgt_te, x_beh_tgt_te, x_perf_tgt_te, y_tgt_te)

    # Target-only training for reference
    model_target_only = DynamicFusion(
        d_eng=x_eng_src_tr.shape[1],
        d_beh=x_beh_src_tr.shape[1],
        d_perf=x_perf_src_tr.shape[1],
        hidden_dim=cfg.hidden_dim,
    )
    fit_dynamic_fusion(model_target_only, tgt_train_loader, tgt_val_loader, cfg.epochs_model, cfg.lr)
    target_only_metrics, _, _ = evaluate_model(model_target_only, x_eng_tgt_te, x_beh_tgt_te, x_perf_tgt_te, y_tgt_te)

    # Baselines
    def baseline_eval(reg):
        reg.fit(src_train_df[feature_cols], src_train_df["target_final"])
        pred = reg.predict(tgt_test_df[feature_cols])
        return {
            "RMSE": float(np.sqrt(mean_squared_error(y_tgt_te, pred))),
            "MAE": float(mean_absolute_error(y_tgt_te, pred)),
            "R2": float(r2_score(y_tgt_te, pred)),
        }

    baseline_linear = baseline_eval(LinearRegression())
    baseline_rf = baseline_eval(RandomForestRegressor(n_estimators=100, random_state=SEED))

    # distribution shift
    dist_df = compute_distribution_shift(src_train_df, tgt_train_df, feature_cols)

    # intervention on attendance
    tgt_te_copy = tgt_test_df.copy()
    if "engagement_attendance" in tgt_te_copy.columns:
        idx = feature_cols.index("engagement_attendance")
        bump = 0.1
        x_alt = tgt_te_copy[feature_cols].to_numpy(dtype=np.float32)
        x_alt[:, idx] = x_alt[:, idx] + bump
        alt_df = pd.DataFrame(x_alt, columns=feature_cols)
        alt_df["target_final"] = tgt_te_copy["target_final"].values
        x1a, x2a, x3a, _ = aligner.split_branches(alt_df)
        _, y_alt, _ = evaluate_model(model, x1a, x2a, x3a, y_tgt_te)
        cf_delta = float(np.mean(y_alt - y_pred_tgt))
    else:
        cf_delta = float("nan")

    # generalization drop and CI
    in_domain_rmse = float(np.sqrt(mean_squared_error(y_tgt_tr, model(torch.tensor(x_eng_tgt_tr), torch.tensor(x_beh_tgt_tr), torch.tensor(x_perf_tgt_tr))[0].detach().numpy())))
    cross_domain_rmse = finetuned_metrics["RMSE"]
    generalization_drop = (cross_domain_rmse - in_domain_rmse) / max(in_domain_rmse, 1e-8)

    abs_errors = np.abs(y_tgt_te - y_pred_tgt)
    ci_low, ci_high = bootstrap_ci(abs_errors)

    result = {
        "data_note": source_note,
        "n_source": int(len(source_aligned)),
        "n_target": int(len(self_aligned)),
        "zero_shot": zero_shot_metrics,
        "fine_tuned": finetuned_metrics,
        "baseline_linear": baseline_linear,
        "baseline_rf": baseline_rf,
        "dynamicfusion_target_only": target_only_metrics,
        "generalization_drop": float(generalization_drop),
        "error_ci_95": [ci_low, ci_high],
        "counterfactual_attendance_plus0.1_delta": cf_delta,
        "attention_mean_zero_shot": zero_attn.mean(axis=0).tolist(),
        "attention_mean_finetuned": ft_attn.mean(axis=0).tolist(),
    }

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    dist_df.to_csv(out_dir / "distribution_shift.csv", index=False)

    perf_table = pd.DataFrame([
        {"model": "DynamicFusion_zero_shot", **zero_shot_metrics},
        {"model": "DynamicFusion_finetuned", **finetuned_metrics},
        {"model": "LinearRegression_transfer", **baseline_linear},
        {"model": "RandomForest_transfer", **baseline_rf},
        {"model": "DynamicFusion_target_only", **target_only_metrics},
    ])
    perf_table.to_csv(out_dir / "performance_table.csv", index=False)

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Fresh DynamicFusion generalization framework")
    parser.add_argument("--self-data", default="/data/student_scores.csv")
    parser.add_argument("--oulad-data", default="")
    parser.add_argument("--output-dir", default="fresh_generalization/outputs")
    parser.add_argument("--epochs-encoder", type=int, default=200)
    parser.add_argument("--epochs-model", type=int, default=300)
    parser.add_argument("--epochs-finetune", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--min-samples-for-training", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    self_path = args.self_data
    if not os.path.exists(self_path):
        fallback = "data/student_scores.csv"
        if os.path.exists(fallback):
            self_path = fallback
        else:
            raise FileNotFoundError(f"Cannot find self dataset at {args.self_data} or {fallback}")

    cfg = Config(
        self_data_path=self_path,
        oulad_data_path=args.oulad_data if args.oulad_data else None,
        output_dir=args.output_dir,
        epochs_encoder=args.epochs_encoder,
        epochs_model=args.epochs_model,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        test_size=args.test_size,
        min_samples_for_training=args.min_samples_for_training,
    )

    results = run_experiment(cfg)
    print(json.dumps(results, ensure_ascii=False, indent=2))
