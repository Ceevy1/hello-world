from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import sys

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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

DEFAULT_DATA_DIR = Path("/data")
DEFAULT_OULAD_CANDIDATES = [
    "studentInfo.csv",
    "oulad.csv",
    "oulad_interactions.csv",
    "studentvle.csv",
    "studentVle.csv",
]


def resolve_default_input(input_path: str | None) -> str:
    if input_path:
        return input_path

    if DEFAULT_DATA_DIR.exists() and DEFAULT_DATA_DIR.is_dir():
        csv_files = sorted(DEFAULT_DATA_DIR.glob("*.csv"))
        if csv_files:
            return str(DEFAULT_DATA_DIR)

    for name in DEFAULT_OULAD_CANDIDATES:
        candidate = DEFAULT_DATA_DIR / name
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "No input CSV provided and no OULAD CSV found under /data. "
        "Please place OULAD CSV in /data or pass --input explicitly."
    )


def _read_csv_case_insensitive(base_dir: Path, file_name: str) -> pd.DataFrame:
    exact = base_dir / file_name
    if exact.exists():
        return pd.read_csv(exact)

    lowered = file_name.lower()
    for candidate in base_dir.glob("*.csv"):
        if candidate.name.lower() == lowered:
            return pd.read_csv(candidate)

    raise FileNotFoundError(f"Required OULAD file not found: {file_name} (under {base_dir})")


def _build_from_oulad_raw(base_dir: Path) -> pd.DataFrame:
    student_vle = _read_csv_case_insensitive(base_dir, "studentVle.csv")
    student_info = _read_csv_case_insensitive(base_dir, "studentInfo.csv")
    vle = _read_csv_case_insensitive(base_dir, "vle.csv")

    merged = student_vle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    info_cols = ["id_student", "code_module", "code_presentation", "final_result"]
    merged = merged.merge(student_info[info_cols], on=["id_student", "code_module", "code_presentation"], how="left")

    out = pd.DataFrame(
        {
            "student_id": merged["id_student"],
            "activity_type": merged["activity_type"].fillna("unknown").astype(str),
            "week": (merged["date"].fillna(0).astype(float) // 7 + 1).astype(int),
            "elapsed_time": merged["sum_click"].fillna(0).astype(float),
            "final_result": merged["final_result"].fillna("Withdrawn").astype(str),
        }
    )
    return out


def load_oulad_dataframe(input_path: str) -> pd.DataFrame:
    path = Path(input_path)

    if path.is_dir():
        return _build_from_oulad_raw(path)

    df = pd.read_csv(path)
    normalized = {c.lower(): c for c in df.columns}

    if "final_result" in normalized and "student_id" in normalized:
        return df

    if "id_student" in normalized and "sum_click" in normalized:
        return _build_from_oulad_raw(path.parent)

    missing = {"student_id", "activity_type", "week", "final_result"} - set(df.columns)
    raise ValueError(
        "Unrecognized input format. "
        "Expected either processed OULAD columns "
        "['student_id', 'activity_type', 'week', 'final_result'] "
        "or raw OULAD files under a directory with studentVle.csv + studentInfo.csv + vle.csv. "
        f"Missing columns from provided CSV: {sorted(missing)}"
    )


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_weeks: int = 40


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC": _safe_auc(y_true, y_prob),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _build_student_week_sequence(df: pd.DataFrame, student_order: List[str], max_weeks: int) -> np.ndarray:
    students = df["student_id"].astype(str)
    df = df.copy()
    df["student_id"] = students

    weekly = (
        df.groupby(["student_id", "week"], as_index=False)
        .agg(
            event_count=("activity_type", "size"),
            unique_activity=("activity_type", "nunique"),
            elapsed_mean=("elapsed_time", "mean"),
            elapsed_std=("elapsed_time", "std"),
        )
        .fillna(0.0)
    )

    seq_feature_dim = 4
    seq = np.zeros((len(student_order), max_weeks, seq_feature_dim), dtype=np.float32)
    idx_lookup = {sid: i for i, sid in enumerate(student_order)}

    for row in weekly.itertuples(index=False):
        sid = str(row.student_id)
        if sid not in idx_lookup:
            continue
        week_idx = int(row.week) - 1
        if week_idx < 0 or week_idx >= max_weeks:
            continue
        seq[idx_lookup[sid], week_idx, 0] = float(row.event_count)
        seq[idx_lookup[sid], week_idx, 1] = float(row.unique_activity)
        seq[idx_lookup[sid], week_idx, 2] = float(row.elapsed_mean)
        seq[idx_lookup[sid], week_idx, 3] = float(row.elapsed_std)

    flat = seq.reshape(-1, seq_feature_dim)
    std = flat.std(axis=0, keepdims=True) + 1e-6
    mean = flat.mean(axis=0, keepdims=True)
    return ((seq - mean) / std).astype(np.float32)


def prepare_oulad_data(input_csv: str, cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = load_oulad_dataframe(input_csv)
    data = preprocess_oulad(raw)

    required = {"student_id", "activity_type", "week", "final_result"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    if "elapsed_time" not in data.columns:
        data["elapsed_time"] = 1.0

    features = build_behavior_features(data, student_col="student_id")
    labels = data.groupby("student_id")["pass_label"].max().reset_index()

    merged = features.merge(labels, on="student_id", how="inner").sort_values("student_id").reset_index(drop=True)
    merged["student_id"] = merged["student_id"].astype(str)

    tab = merged[BASE_FEATURES].to_numpy(dtype=np.float32)
    y = merged["pass_label"].to_numpy(dtype=np.int64)
    seq = _build_student_week_sequence(data, merged["student_id"].tolist(), cfg.max_weeks)

    x_seq_train, x_seq_val, x_tab_train, x_tab_val, y_train, y_val = train_test_split(
        seq,
        tab,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    return x_seq_train, x_seq_val, x_tab_train, x_tab_val, y_train, y_val


class LSTMKTClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, seq: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq)
        return self.head(h_n[-1]).squeeze(-1)


class VanillaTransformerClassifier(nn.Module):
    def __init__(self, seq_dim: int, model_dim: int = 64, nhead: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(seq_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,
            dropout=0.2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(model_dim, 1))

    def forward(self, seq: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(seq)
        z = self.encoder(x).mean(dim=1)
        return self.head(z).squeeze(-1)


class StaticFusionClassifier(nn.Module):
    def __init__(self, tab_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        assert tab is not None
        return self.net(tab).squeeze(-1)


class DynamicFusionOursClassifier(nn.Module):
    def __init__(self, seq_dim: int, tab_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.seq_encoder = nn.LSTM(input_size=seq_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.tab_encoder = nn.Sequential(nn.Linear(tab_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, seq: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        assert tab is not None
        _, (h_n, _) = self.seq_encoder(seq)
        seq_repr = h_n[-1]
        tab_repr = self.tab_encoder(tab)
        alpha = self.gate(torch.cat([seq_repr, tab_repr], dim=-1))
        fused = alpha * seq_repr + (1.0 - alpha) * tab_repr
        return self.head(fused).squeeze(-1)


def _run_torch_model(
    model: nn.Module,
    x_seq_train: np.ndarray,
    x_seq_val: np.ndarray,
    x_tab_train: np.ndarray,
    x_tab_val: np.ndarray,
    y_train: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_ds = TensorDataset(
        torch.tensor(x_seq_train, dtype=torch.float32),
        torch.tensor(x_tab_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    model.train()
    for _ in range(cfg.epochs):
        for seq_b, tab_b, y_b in train_loader:
            seq_b = seq_b.to(device)
            tab_b = tab_b.to(device)
            y_b = y_b.to(device)
            logits = model(seq_b, tab_b)
            loss = criterion(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(torch.tensor(x_seq_train, dtype=torch.float32, device=device), torch.tensor(x_tab_train, dtype=torch.float32, device=device))).cpu().numpy()
        val_prob = torch.sigmoid(model(torch.tensor(x_seq_val, dtype=torch.float32, device=device), torch.tensor(x_tab_val, dtype=torch.float32, device=device))).cpu().numpy()
    return train_prob, val_prob


def run_experiment(input_csv: str | None, output_csv: str, cfg: TrainConfig) -> pd.DataFrame:
    resolved_input = resolve_default_input(input_csv)
    x_seq_train, x_seq_val, x_tab_train, x_tab_val, y_train, y_val = prepare_oulad_data(resolved_input, cfg)

    scaler = StandardScaler()
    x_tab_train_scaled = scaler.fit_transform(x_tab_train)
    x_tab_val_scaled = scaler.transform(x_tab_val)

    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=cfg.random_state))]),
        "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=cfg.random_state))]),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=cfg.random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=cfg.random_state),
        "AdaBoost": AdaBoostClassifier(random_state=cfg.random_state),
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "Decision Tree": DecisionTreeClassifier(random_state=cfg.random_state),
    }

    rows: List[Dict[str, float | str]] = []

    for name, model in models.items():
        clf = clone(model)
        clf.fit(x_tab_train, y_train)
        train_prob = clf.predict_proba(x_tab_train)[:, 1]
        val_prob = clf.predict_proba(x_tab_val)[:, 1]

        rows.append({"Model": name, "Split": "Train", **_compute_metrics(y_train, train_prob)})
        rows.append({"Model": name, "Split": "Validation", **_compute_metrics(y_val, val_prob)})

    torch_models = {
        "LSTM-KT": LSTMKTClassifier(input_dim=x_seq_train.shape[-1]),
        "Vanilla Transformer": VanillaTransformerClassifier(seq_dim=x_seq_train.shape[-1]),
        "Static Fusion": StaticFusionClassifier(tab_dim=x_tab_train.shape[-1]),
        "Dynamic Fusion (Ours)": DynamicFusionOursClassifier(seq_dim=x_seq_train.shape[-1], tab_dim=x_tab_train.shape[-1]),
    }

    for name, model in torch_models.items():
        if name == "Static Fusion":
            seq_train_input = np.zeros_like(x_seq_train, dtype=np.float32)
            seq_val_input = np.zeros_like(x_seq_val, dtype=np.float32)
            train_prob, val_prob = _run_torch_model(
                model,
                seq_train_input,
                seq_val_input,
                x_tab_train_scaled,
                x_tab_val_scaled,
                y_train,
                cfg,
            )
        else:
            train_prob, val_prob = _run_torch_model(
                model,
                x_seq_train,
                x_seq_val,
                x_tab_train_scaled,
                x_tab_val_scaled,
                y_train,
                cfg,
            )

        rows.append({"Model": name, "Split": "Train", **_compute_metrics(y_train, train_prob)})
        rows.append({"Model": name, "Split": "Validation", **_compute_metrics(y_val, val_prob)})

    result_df = pd.DataFrame(rows)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate OULAD models with unified metrics output.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to OULAD CSV or raw OULAD directory. Defaults to auto-detect under /data.",
    )
    parser.add_argument("--output", default="outputs/oulad_train_validation_metrics.csv", help="Output CSV for train/validation metrics.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-weeks", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_weeks=args.max_weeks,
    )
    resolved_input = resolve_default_input(args.input)
    print(f"Using input CSV: {resolved_input}")
    df = run_experiment(resolved_input, args.output, config)
    print(df)
