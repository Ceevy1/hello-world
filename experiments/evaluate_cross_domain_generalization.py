from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engineering.behavior_features import build_behavior_features
from preprocess.oulad_preprocess import preprocess_oulad
from preprocessing import LAB_COLS, STATIC_COLS, TARGET_COL, EXERCISE_COLS, preprocess_scores


DEFAULT_OULAD_DIR = Path("/data")
DEFAULT_SCORES = Path("data/student_scores.csv")


@dataclass
class EvalConfig:
    max_weeks: int = 40
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    random_state: int = 42


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC": _safe_auc(y_true, y_prob),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1-score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _read_csv_ci(base_dir: Path, file_name: str) -> pd.DataFrame:
    p = base_dir / file_name
    if p.exists():
        return pd.read_csv(p)
    wanted = file_name.lower()
    for c in base_dir.glob("*.csv"):
        if c.name.lower() == wanted:
            return pd.read_csv(c)
    raise FileNotFoundError(f"Missing {file_name} under {base_dir}")


def load_oulad_events(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    base = path if path.is_dir() else path.parent

    student_vle = _read_csv_ci(base, "studentVle.csv")
    student_info = _read_csv_ci(base, "studentInfo.csv")
    vle = _read_csv_ci(base, "vle.csv")

    merged = student_vle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    merged = merged.merge(
        student_info[["id_student", "code_module", "code_presentation", "final_result"]],
        on=["id_student", "code_module", "code_presentation"],
        how="left",
    )

    out = pd.DataFrame(
        {
            "student_id": merged["id_student"].astype(str),
            "module": merged["code_module"].astype(str),
            "activity_type": merged["activity_type"].fillna("unknown").astype(str),
            "week": (merged["date"].fillna(0).astype(float) // 7 + 1).astype(int),
            "elapsed_time": merged["sum_click"].fillna(0).astype(float),
            "final_result": merged["final_result"].fillna("Withdrawn").astype(str),
        }
    )
    return out


def _build_oulad_student_level(events: pd.DataFrame, max_weeks: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    events = preprocess_oulad(events)
    events = events.copy()
    events["student_key"] = events["student_id"].astype(str) + "__" + events["module"].astype(str)

    feat_input = events.drop(columns=["student_id"]).rename(columns={"student_key": "student_id"})
    feat = build_behavior_features(feat_input, student_col="student_id")
    labels = events.groupby("student_key")["pass_label"].max().rename("label")
    modules = events.groupby("student_key")["module"].first().rename("module")

    merged = feat.merge(labels, left_on="student_id", right_index=True).merge(modules, left_on="student_id", right_index=True)

    tab = merged[["activity_entropy", "active_week_ratio", "procrastination_index", "resource_switch_rate", "avg_session_length"]].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=np.int64)
    domain = merged["module"].to_numpy(dtype=str)

    seq = np.zeros((len(merged), max_weeks, 4), dtype=np.float32)
    idx = {sid: i for i, sid in enumerate(merged["student_id"].tolist())}
    weekly = (
        events.groupby(["student_key", "week"], as_index=False)
        .agg(
            event_count=("activity_type", "size"),
            unique_activity=("activity_type", "nunique"),
            elapsed_mean=("elapsed_time", "mean"),
            elapsed_std=("elapsed_time", "std"),
        )
        .fillna(0.0)
    )
    for row in weekly.itertuples(index=False):
        i = idx.get(row.student_key)
        if i is None:
            continue
        w = int(row.week) - 1
        if 0 <= w < max_weeks:
            seq[i, w, 0] = float(row.event_count)
            seq[i, w, 1] = float(row.unique_activity)
            seq[i, w, 2] = float(row.elapsed_mean)
            seq[i, w, 3] = float(row.elapsed_std)

    return seq, tab, y, domain


def _build_scores_student_level(score_csv: str, max_weeks: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prep = preprocess_scores(score_csv, standardize=False)
    df = pd.concat([prep.train_df, prep.val_df, prep.test_df], ignore_index=True)

    exercise = df[EXERCISE_COLS].to_numpy(dtype=np.float32)
    labs = df[LAB_COLS].to_numpy(dtype=np.float32)
    static = df[STATIC_COLS].to_numpy(dtype=np.float32)

    ex_avg = exercise.mean(axis=1)
    lab_avg = labs.mean(axis=1)
    tab = np.column_stack(
        [
            static[:, 0],
            ex_avg,
            lab_avg,
            static[:, 1],
            static[:, 2],
        ]
    ).astype(np.float32)

    seq_raw = np.concatenate([exercise, labs], axis=1)
    L = seq_raw.shape[1]
    seq = np.zeros((len(df), max_weeks, 4), dtype=np.float32)
    for i in range(len(df)):
        x = seq_raw[i]
        x = np.clip(x, 0, 100) / 100.0
        d = np.concatenate([[0.0], np.diff(x)])
        c = np.cumsum(x) / (np.arange(L) + 1)
        p = np.linspace(0.0, 1.0, L)
        seq[i, :L, 0] = x
        seq[i, :L, 1] = d
        seq[i, :L, 2] = c
        seq[i, :L, 3] = p

    y = (df[TARGET_COL].to_numpy(dtype=np.float32) >= 70.0).astype(np.int64)
    domain = np.array(["SCORES"] * len(df), dtype=str)
    return seq, tab, y, domain


def _normalize_seq_by_train(train_seq: np.ndarray, test_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = train_seq.reshape(-1, train_seq.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-6
    return ((train_seq - mean) / std).astype(np.float32), ((test_seq - mean) / std).astype(np.float32)


class LSTMKT(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, dropout=0.2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, seq: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(seq)
        return self.head(h[-1]).squeeze(-1)


class VanillaTransformer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 2, dropout=0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, seq: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        z = self.enc(self.proj(seq)).mean(dim=1)
        return self.head(z).squeeze(-1)


class StaticFusion(nn.Module):
    def __init__(self, tab_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(tab_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, seq: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        return self.net(tab).squeeze(-1)


class DynamicFusion(nn.Module):
    def __init__(self, seq_dim: int, tab_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.seq_enc = nn.LSTM(seq_dim, hidden, batch_first=True)
        self.tab_enc = nn.Sequential(nn.Linear(tab_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, seq: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.seq_enc(seq)
        s = h[-1]
        t = self.tab_enc(tab)
        a = self.gate(torch.cat([s, t], dim=-1))
        z = a * s + (1 - a) * t
        return self.head(z).squeeze(-1)


def _train_torch_and_predict(
    model: nn.Module,
    train_seq: np.ndarray,
    train_tab: np.ndarray,
    train_y: np.ndarray,
    test_seq: np.ndarray,
    test_tab: np.ndarray,
    cfg: EvalConfig,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ds = TensorDataset(
        torch.tensor(train_seq, dtype=torch.float32),
        torch.tensor(train_tab, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    model.train()
    for _ in range(cfg.epochs):
        for seq_b, tab_b, y_b in loader:
            seq_b, tab_b, y_b = seq_b.to(device), tab_b.to(device), y_b.to(device)
            logits = model(seq_b, tab_b)
            loss = loss_fn(logits, y_b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(
                torch.tensor(test_seq, dtype=torch.float32, device=device),
                torch.tensor(test_tab, dtype=torch.float32, device=device),
            )
        ).cpu().numpy()
    return probs


def evaluate_model_suite(
    train_seq: np.ndarray,
    train_tab: np.ndarray,
    train_y: np.ndarray,
    test_seq: np.ndarray,
    test_tab: np.ndarray,
    test_y: np.ndarray,
    cfg: EvalConfig,
) -> List[Dict[str, float | str]]:
    scaler = StandardScaler()
    train_tab_scaled = scaler.fit_transform(train_tab)
    test_tab_scaled = scaler.transform(test_tab)

    train_seq_norm, test_seq_norm = _normalize_seq_by_train(train_seq, test_seq)

    rows: List[Dict[str, float | str]] = []

    classical = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=cfg.random_state))]),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=cfg.random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=cfg.random_state),
    }
    for name, model in classical.items():
        clf = clone(model)
        clf.fit(train_tab, train_y)
        prob = clf.predict_proba(test_tab)[:, 1]
        rows.append({"Model": name, **_binary_metrics(test_y, prob)})

    deep_models = {
        "LSTM-KT": LSTMKT(in_dim=train_seq.shape[-1]),
        "Transformer": VanillaTransformer(in_dim=train_seq.shape[-1]),
        "Static Fusion": StaticFusion(tab_dim=train_tab.shape[-1]),
        "Dynamic Fusion (Ours)": DynamicFusion(seq_dim=train_seq.shape[-1], tab_dim=train_tab.shape[-1]),
    }

    for name, model in deep_models.items():
        if name == "Static Fusion":
            test_prob = _train_torch_and_predict(
                model,
                np.zeros_like(train_seq_norm),
                train_tab_scaled,
                train_y,
                np.zeros_like(test_seq_norm),
                test_tab_scaled,
                cfg,
            )
        else:
            test_prob = _train_torch_and_predict(
                model,
                train_seq_norm,
                train_tab_scaled,
                train_y,
                test_seq_norm,
                test_tab_scaled,
                cfg,
            )

        rows.append({"Model": name, **_binary_metrics(test_y, test_prob)})

    return rows


def run_leave_one_domain_out(oulad_input: str, cfg: EvalConfig) -> pd.DataFrame:
    seq, tab, y, domain = _build_oulad_student_level(load_oulad_events(oulad_input), cfg.max_weeks)
    modules = sorted(np.unique(domain))
    rows: List[Dict[str, float | str]] = []

    for test_module in modules:
        test_mask = domain == test_module
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
            continue

        metrics_rows = evaluate_model_suite(
            train_seq=seq[train_mask],
            train_tab=tab[train_mask],
            train_y=y[train_mask],
            test_seq=seq[test_mask],
            test_tab=tab[test_mask],
            test_y=y[test_mask],
            cfg=cfg,
        )
        for r in metrics_rows:
            r.update(
                {
                    "Scenario": "Leave-One-Domain-Out",
                    "TrainDomain": "ALL_EXCEPT_" + test_module,
                    "TestDomain": test_module,
                    "TrainSize": int(train_mask.sum()),
                    "TestSize": int(test_mask.sum()),
                }
            )
            rows.append(r)

    return pd.DataFrame(rows)


def run_cross_dataset(oulad_input: str, scores_csv: str, cfg: EvalConfig) -> pd.DataFrame:
    ou_seq, ou_tab, ou_y, _ = _build_oulad_student_level(load_oulad_events(oulad_input), cfg.max_weeks)
    sc_seq, sc_tab, sc_y, _ = _build_scores_student_level(scores_csv, cfg.max_weeks)

    rows: List[Dict[str, float | str]] = []

    directions = [
        ("OULAD", ou_seq, ou_tab, ou_y, "StudentScores", sc_seq, sc_tab, sc_y),
        ("StudentScores", sc_seq, sc_tab, sc_y, "OULAD", ou_seq, ou_tab, ou_y),
    ]

    for train_name, tr_seq, tr_tab, tr_y, test_name, te_seq, te_tab, te_y in directions:
        if len(np.unique(tr_y)) < 2 or len(np.unique(te_y)) < 2:
            continue

        metrics_rows = evaluate_model_suite(
            train_seq=tr_seq,
            train_tab=tr_tab,
            train_y=tr_y,
            test_seq=te_seq,
            test_tab=te_tab,
            test_y=te_y,
            cfg=cfg,
        )
        for r in metrics_rows:
            r.update(
                {
                    "Scenario": "Cross-Dataset",
                    "TrainDomain": train_name,
                    "TestDomain": test_name,
                    "TrainSize": int(len(tr_y)),
                    "TestSize": int(len(te_y)),
                }
            )
            rows.append(r)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-domain generalization evaluation for OULAD-based models.")
    parser.add_argument("--oulad-input", default=str(DEFAULT_OULAD_DIR), help="Raw OULAD directory (contains studentVle/studentInfo/vle CSVs).")
    parser.add_argument("--scores-input", default=str(DEFAULT_SCORES), help="Path to self-built student_scores.csv.")
    parser.add_argument("--output", default="outputs/cross_domain_generalization_metrics.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-weeks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvalConfig(
        max_weeks=args.max_weeks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_state=args.seed,
    )
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)

    lodo_df = run_leave_one_domain_out(args.oulad_input, cfg)
    cross_df = run_cross_dataset(args.oulad_input, args.scores_input, cfg)
    result = pd.concat([lodo_df, cross_df], ignore_index=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")

    print("Saved:", out)
    print(result.head(20))


if __name__ == "__main__":
    main()
