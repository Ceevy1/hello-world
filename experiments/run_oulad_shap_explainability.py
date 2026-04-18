from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_cross_domain_generalization import _build_oulad_student_level, load_oulad_events
from models.catboost_model import CatBoostClassifierModel
from models.xgboost_model import XGBoostClassifierModel


@dataclass
class ExplainConfig:
    max_weeks: int = 40
    test_size: float = 0.2
    random_state: int = 42
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3


class WeekAwareFusion(nn.Module):
    """Fuse tree branch and transformer branch with interpretable dynamic weights."""

    def __init__(self, seq_dim: int, tab_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(seq_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.trans_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + tab_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        seq: torch.Tensor,
        tab: torch.Tensor,
        tree_prob: torch.Tensor,
        week_ratio: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(self.proj(seq)).mean(dim=1)
        trans_prob = torch.sigmoid(self.trans_head(z).squeeze(-1))

        gate_in = torch.cat([z, tab, week_ratio.unsqueeze(-1)], dim=1)
        tree_weight = self.gate(gate_in).squeeze(-1)
        trans_weight = 1.0 - tree_weight

        y_prob = tree_weight * tree_prob + trans_weight * trans_prob
        if return_weights:
            return y_prob, tree_weight, trans_weight
        return y_prob


def _prepare_data(oulad_input: str, cfg: ExplainConfig):
    seq, tab, y, domain = _build_oulad_student_level(load_oulad_events(oulad_input), cfg.max_weeks)
    week_counts = np.clip((seq[:, :, 0] > 0).sum(axis=1), 1, cfg.max_weeks)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    return {
        "seq_train": seq[train_idx],
        "seq_test": seq[test_idx],
        "tab_train": tab[train_idx],
        "tab_test": tab[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "week_train": week_counts[train_idx].astype(np.float32),
        "week_test": week_counts[test_idx].astype(np.float32),
        "domain_train": domain[train_idx],
        "domain_test": domain[test_idx],
        "feature_names": ["activity_entropy", "active_week_ratio", "procrastination_index", "resource_switch_rate", "avg_session_length"],
    }


def _safe_shap_values(explainer, x: np.ndarray) -> np.ndarray:
    sv = explainer.shap_values(x)
    if isinstance(sv, list):
        if len(sv) == 2:
            return np.array(sv[1])
        return np.array(sv[0])
    sv = np.array(sv)
    if sv.ndim == 3 and sv.shape[-1] == 2:
        return sv[:, :, 1]
    return sv


def _train_tree_models(tab_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    xgb = XGBoostClassifierModel(random_state=42)
    xgb.fit(tab_train, y_train)

    cat = CatBoostClassifierModel(random_state=42)
    cat.fit(tab_train, y_train)

    return {"XGB": xgb, "CAT": cat}


def _plot_shap_and_ranking(
    model_name: str,
    model_obj,
    x_eval: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    import matplotlib.pyplot as plt
    import shap

    if hasattr(model_obj, "model"):
        model = model_obj.model
    else:
        model = model_obj

    explainer = shap.TreeExplainer(model)
    shap_values = _safe_shap_values(explainer, x_eval)

    shap.summary_plot(shap_values, x_eval, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_summary_{model_name}.png", dpi=300)
    plt.close()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranking = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(out_dir / f"feature_ranking_{model_name}.csv", index=False, encoding="utf-8-sig")

    return ranking, shap_values, explainer.expected_value


def _single_student_explanation(
    shap_values: np.ndarray,
    expected_value,
    x_eval: np.ndarray,
    y_prob: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
) -> Dict[str, object]:
    import matplotlib.pyplot as plt
    import shap

    fail_idx = int(np.argmin(y_prob))
    sample_shap = shap_values[fail_idx]
    sample_x = x_eval[fail_idx]

    shap.force_plot(
        expected_value if np.isscalar(expected_value) else expected_value[1],
        sample_shap,
        sample_x,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_force_studentA.png", dpi=300)
    plt.close()

    contributions = sorted(zip(feature_names, sample_shap.tolist()), key=lambda kv: kv[1])
    negative_top = [{"feature": f, "shap": float(v)} for f, v in contributions[:3]]

    explanation = {
        "student_index_in_test": fail_idx,
        "predicted_fail_probability": float(1.0 - y_prob[fail_idx]),
        "predicted_pass_probability": float(y_prob[fail_idx]),
        "message": "Student A predicted FAIL because top negative SHAP features are low-impact toward pass.",
        "top_negative_contributors": negative_top,
    }
    with open(out_dir / "single_student_explanation.json", "w", encoding="utf-8") as f:
        json.dump(explanation, f, ensure_ascii=False, indent=2)
    return explanation


def _train_fusion_and_plot(
    seq_train: np.ndarray,
    tab_train: np.ndarray,
    y_train: np.ndarray,
    week_train: np.ndarray,
    seq_test: np.ndarray,
    tab_test: np.ndarray,
    week_test: np.ndarray,
    tree_train_prob: np.ndarray,
    cfg: ExplainConfig,
    out_dir: Path,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    scaler_tab = StandardScaler()
    tab_train_n = scaler_tab.fit_transform(tab_train)
    tab_test_n = scaler_tab.transform(tab_test)

    flat = seq_train.reshape(-1, seq_train.shape[-1])
    seq_mean = flat.mean(axis=0, keepdims=True)
    seq_std = flat.std(axis=0, keepdims=True) + 1e-6
    seq_train_n = (seq_train - seq_mean) / seq_std
    seq_test_n = (seq_test - seq_mean) / seq_std

    week_ratio_train = week_train / float(cfg.max_weeks)
    week_ratio_test = week_test / float(cfg.max_weeks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeekAwareFusion(seq_dim=seq_train.shape[-1], tab_dim=tab_train.shape[-1]).to(device)

    ds = TensorDataset(
        torch.tensor(seq_train_n, dtype=torch.float32),
        torch.tensor(tab_train_n, dtype=torch.float32),
        torch.tensor(tree_train_prob, dtype=torch.float32),
        torch.tensor(week_ratio_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    loss_fn = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    model.train()
    for _ in range(cfg.epochs):
        for seq_b, tab_b, tree_b, week_b, y_b in loader:
            seq_b, tab_b = seq_b.to(device), tab_b.to(device)
            tree_b, week_b, y_b = tree_b.to(device), week_b.to(device), y_b.to(device)
            prob = model(seq_b, tab_b, tree_b, week_b)
            loss = loss_fn(prob, y_b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    week_rows = []
    with torch.no_grad():
        tree_test_prob = torch.tensor(tree_train_prob.mean() * np.ones(len(seq_test_n)), dtype=torch.float32, device=device)
        for week in range(1, cfg.max_weeks + 1):
            seq_cut = seq_test_n.copy()
            seq_cut[:, week:, :] = 0.0
            week_ratio = torch.full((len(seq_cut),), float(week / cfg.max_weeks), dtype=torch.float32, device=device)
            _, w_tree, w_trans = model(
                torch.tensor(seq_cut, dtype=torch.float32, device=device),
                torch.tensor(tab_test_n, dtype=torch.float32, device=device),
                tree_test_prob,
                week_ratio,
                return_weights=True,
            )
            week_rows.append(
                {
                    "week": week,
                    "tree_weight": float(w_tree.mean().cpu().item()),
                    "transformer_weight": float(w_trans.mean().cpu().item()),
                }
            )

    wdf = pd.DataFrame(week_rows)
    wdf.to_csv(out_dir / "fusion_weight_by_week.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 4))
    plt.plot(wdf["week"], wdf["tree_weight"], label="Tree branch weight", linewidth=2)
    plt.plot(wdf["week"], wdf["transformer_weight"], label="Transformer branch weight", linewidth=2)
    plt.xlabel("Week")
    plt.ylabel("Average fusion weight")
    plt.title("Fusion weight vs week")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fusion_weight_vs_week.png", dpi=300)
    plt.close()

    return wdf


def run_experiment(oulad_input: str, output_dir: str, cfg: ExplainConfig) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _prepare_data(oulad_input, cfg)
    models = _train_tree_models(data["tab_train"], data["y_train"])

    xgb_train_prob = models["XGB"].predict_proba(data["tab_train"])
    xgb_test_prob = models["XGB"].predict_proba(data["tab_test"])
    cat_train_prob = models["CAT"].predict_proba(data["tab_train"])

    # Tree SHAP (global + feature ranking) for both XGB/CAT
    xgb_ranking, xgb_shap, xgb_expected = _plot_shap_and_ranking(
        "XGB", models["XGB"], data["tab_test"], data["feature_names"], out_dir
    )
    cat_ranking, _, _ = _plot_shap_and_ranking(
        "CAT", models["CAT"], data["tab_test"], data["feature_names"], out_dir
    )

    # Single student explanation with XGB
    single_exp = _single_student_explanation(
        shap_values=xgb_shap,
        expected_value=xgb_expected,
        x_eval=data["tab_test"],
        y_prob=xgb_test_prob,
        feature_names=data["feature_names"],
        out_dir=out_dir,
    )

    # Fusion weight dynamics (early tree / late transformer trend)
    wdf = _train_fusion_and_plot(
        seq_train=data["seq_train"],
        tab_train=data["tab_train"],
        y_train=data["y_train"].astype(np.float32),
        week_train=data["week_train"],
        seq_test=data["seq_test"],
        tab_test=data["tab_test"],
        week_test=data["week_test"],
        tree_train_prob=((xgb_train_prob + cat_train_prob) / 2.0).astype(np.float32),
        cfg=cfg,
        out_dir=out_dir,
    )

    report = {
        "shap_summary_xgb": str(out_dir / "shap_summary_XGB.png"),
        "shap_summary_cat": str(out_dir / "shap_summary_CAT.png"),
        "feature_ranking_xgb": str(out_dir / "feature_ranking_XGB.csv"),
        "feature_ranking_cat": str(out_dir / "feature_ranking_CAT.csv"),
        "single_student_force_plot": str(out_dir / "shap_force_studentA.png"),
        "single_student_explanation": str(out_dir / "single_student_explanation.json"),
        "fusion_weight_curve": str(out_dir / "fusion_weight_vs_week.png"),
        "fusion_weight_table": str(out_dir / "fusion_weight_by_week.csv"),
    }

    pd.DataFrame([report]).to_csv(out_dir / "artifact_index.csv", index=False, encoding="utf-8-sig")

    print("Top XGB features:\n", xgb_ranking.head(10))
    print("Top CAT features:\n", cat_ranking.head(10))
    print("Single student explanation:", json.dumps(single_exp, ensure_ascii=False, indent=2))
    print("Fusion weights (head):\n", wdf.head())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OULAD SHAP explainability + fusion-weight analysis experiment.")
    parser.add_argument("--oulad-input", default="/data", help="Path to raw OULAD folder (studentVle/studentInfo/vle).")
    parser.add_argument("--output-dir", default="outputs/oulad_shap_explainability")
    parser.add_argument("--max-weeks", type=int, default=40)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExplainConfig(
        max_weeks=args.max_weeks,
        test_size=args.test_size,
        random_state=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)

    artifacts = run_experiment(args.oulad_input, args.output_dir, cfg)
    print("Saved artifacts:")
    for k, v in artifacts.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
