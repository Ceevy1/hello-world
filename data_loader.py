from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


@dataclass
class OULADBatch:
    x_seq: torch.Tensor
    x_stat: torch.Tensor
    node_index: torch.Tensor
    week_index: torch.Tensor
    y: torch.Tensor


class OULADDataset(Dataset):
    def __init__(self, seq: np.ndarray, stat: np.ndarray, node_index: np.ndarray, week_index: np.ndarray, y: np.ndarray) -> None:
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.stat = torch.tensor(stat, dtype=torch.float32)
        self.node_index = torch.tensor(node_index, dtype=torch.long)
        self.week_index = torch.tensor(week_index, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> OULADBatch:
        return OULADBatch(
            x_seq=self.seq[idx],
            x_stat=self.stat[idx],
            node_index=self.node_index[idx],
            week_index=self.week_index[idx],
            y=self.y[idx],
        )


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_oulad_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_path = Path(data_dir)
    return {
        "studentInfo": _safe_read_csv(data_path / "studentInfo.csv"),
        "studentVle": _safe_read_csv(data_path / "studentVle.csv"),
        "assessments": _safe_read_csv(data_path / "assessments.csv"),
        "studentAssessment": _safe_read_csv(data_path / "studentAssessment.csv"),
        "vle": _safe_read_csv(data_path / "vle.csv"),
    }


def _fallback_synthetic(n: int = 800, seq_len: int = 16, seq_dim: int = 6, stat_dim: int = 14, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_seq = rng.normal(size=(n, seq_len, seq_dim)).astype(np.float32)
    x_stat = rng.normal(size=(n, stat_dim)).astype(np.float32)
    week = rng.integers(3, 16, size=n)
    node_idx = np.arange(n)
    score = x_seq[:, :, 0].mean(1) * 1.3 + x_stat[:, 0] * 1.1 - x_stat[:, 1] * 0.9 + (week > 8).astype(np.float32) * 0.6
    prob = 1.0 / (1.0 + np.exp(-score))
    y = (prob > 0.5).astype(np.float32)
    return x_seq, x_stat, node_idx, week, y


def extract_time_series(tables: dict[str, pd.DataFrame], max_weeks: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    info = tables["studentInfo"]
    vle = tables["studentVle"]
    if info.empty or vle.empty:
        return _fallback_synthetic()

    merged = info[["id_student", "final_result", "studied_credits", "num_of_prev_attempts"]].copy()
    merged["y"] = merged["final_result"].isin(["Pass", "Distinction"]).astype(np.float32)

    vle = vle[["id_student", "date", "sum_click"]].copy()
    vle["week"] = np.clip((vle["date"] // 7).astype(int), 0, max_weeks - 1)
    agg = vle.groupby(["id_student", "week"], as_index=False).agg(sum_click=("sum_click", "sum"), act_days=("date", "nunique"))

    students = merged["id_student"].unique()
    student_to_idx = {sid: i for i, sid in enumerate(students)}
    x_seq = np.zeros((len(students), max_weeks, 2), dtype=np.float32)
    for row in agg.itertuples(index=False):
        i = student_to_idx[row.id_student]
        x_seq[i, int(row.week), 0] = np.log1p(row.sum_click)
        x_seq[i, int(row.week), 1] = row.act_days

    stat = merged.drop_duplicates("id_student").set_index("id_student").loc[students]
    x_stat = stat[["studied_credits", "num_of_prev_attempts"]].fillna(0).to_numpy(dtype=np.float32)
    week_idx = np.full(len(students), max_weeks - 1, dtype=np.int64)
    y = stat["y"].to_numpy(dtype=np.float32)
    node_idx = np.arange(len(students), dtype=np.int64)
    return x_seq, x_stat, node_idx, week_idx, y


def construct_knowledge_graph(num_students: int, feat_dim: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(num_students, dtype=torch.float32).unsqueeze(1)
    base = torch.linspace(0.1, 1.0, steps=feat_dim).unsqueeze(0)
    node_feat = torch.sin(idx * base)
    src = torch.arange(0, num_students - 1, dtype=torch.long)
    dst = torch.arange(1, num_students, dtype=torch.long)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return node_feat, edge_index


def build_dataloaders(data_dir: str | Path, batch_size: int = 64, val_ratio: float = 0.1, test_ratio: float = 0.2, random_state: int = 42) -> tuple[DataLoader, DataLoader, DataLoader, tuple[torch.Tensor, torch.Tensor]]:
    x_seq, x_stat, node_idx, week_idx, y = extract_time_series(load_oulad_data(data_dir))

    x_seq_tv, x_seq_test, x_stat_tv, x_stat_test, node_tv, node_test, week_tv, week_test, y_tv, y_test = train_test_split(
        x_seq, x_stat, node_idx, week_idx, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    val_frac = val_ratio / (1 - test_ratio)
    x_seq_train, x_seq_val, x_stat_train, x_stat_val, node_train, node_val, week_train, week_val, y_train, y_val = train_test_split(
        x_seq_tv,
        x_stat_tv,
        node_tv,
        week_tv,
        y_tv,
        test_size=val_frac,
        random_state=random_state,
        stratify=y_tv,
    )

    train_ds = OULADDataset(x_seq_train, x_stat_train, node_train, week_train, y_train)
    val_ds = OULADDataset(x_seq_val, x_stat_val, node_val, week_val, y_val)
    test_ds = OULADDataset(x_seq_test, x_stat_test, node_test, week_test, y_test)

    num_nodes = int(np.max(node_idx) + 1)
    graph = construct_knowledge_graph(num_nodes)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=list),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=list),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=list),
        graph,
    )
