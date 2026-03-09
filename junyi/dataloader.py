from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_LOG_COLUMNS = [
    "user_id",
    "exercise_id",
    "correct",
    "elapsed_time",
    "timestamp",
]


@dataclass
class JunyiConfig:
    max_seq_len: int = 100
    min_history: int = 1
    min_interactions_per_user: int = 20
    elapsed_time_clip_sec: float = 7200.0


class JunyiSequenceDataset(Dataset):
    def __init__(self, samples: Dict[str, np.ndarray]) -> None:
        self.exercise_ids = torch.LongTensor(samples["exercise_ids"])
        self.response_ids = torch.LongTensor(samples["response_ids"])
        self.continuous = torch.FloatTensor(samples["continuous"])
        self.mask = torch.FloatTensor(samples["mask"])
        self.target = torch.FloatTensor(samples["target"])
        self.target_exercise = torch.LongTensor(samples["target_exercise"])

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int):
        return {
            "exercise_ids": self.exercise_ids[idx],
            "response_ids": self.response_ids[idx],
            "continuous": self.continuous[idx],
            "mask": self.mask[idx],
            "target": self.target[idx],
            "target_exercise": self.target_exercise[idx],
        }


class JunyiDataBuilder:
    def __init__(self, cfg: JunyiConfig = JunyiConfig()) -> None:
        self.cfg = cfg
        self.exercise2id: Dict[str, int] = {"<PAD>": 0}

    @staticmethod
    def _normalize_timestamp_seconds(series: pd.Series) -> pd.Series:
        ts_numeric = pd.to_numeric(series, errors="coerce")
        if ts_numeric.notna().any():
            median_abs = float(np.nanmedian(np.abs(ts_numeric.values)))
            # Heuristic by magnitude to avoid wrong unit assumptions.
            if median_abs > 1e14:  # microseconds
                return ts_numeric / 1e6
            if median_abs > 1e11:  # milliseconds
                return ts_numeric / 1e3
            return ts_numeric  # already seconds

        dt_series = pd.to_datetime(series, errors="coerce")
        return pd.Series(
            np.where(dt_series.notna(), dt_series.astype("int64") / 1e9, np.nan),
            index=series.index,
        )

    @staticmethod
    def _safe_standardize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean = float(values.mean())
        std = float(values.std())
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        out = (values - mean) / std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def load_logs(self, log_csv: str | Path) -> pd.DataFrame:
        df = pd.read_csv(log_csv)

        # Junyi original column names -> unified training schema.
        column_map = {
            "exercise": "exercise_id",
            "time_done": "timestamp",
            "time_taken": "elapsed_time",
        }
        for src, tgt in column_map.items():
            if src in df.columns and tgt not in df.columns:
                df[tgt] = df[src]

        if "hint_used" not in df.columns:
            df["hint_used"] = 0

        missing = [c for c in REQUIRED_LOG_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in log file: {missing}")

        df = df[REQUIRED_LOG_COLUMNS + ["hint_used"]].copy()
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int).clip(0, 1)

        elapsed = pd.to_numeric(df["elapsed_time"], errors="coerce")
        elapsed = elapsed.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        if self.cfg.elapsed_time_clip_sec > 0:
            elapsed = elapsed.clip(upper=self.cfg.elapsed_time_clip_sec)
        df["elapsed_time"] = elapsed

        hints = pd.to_numeric(df["hint_used"], errors="coerce")
        df["hint_used"] = hints.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=100.0)

        df["timestamp"] = self._normalize_timestamp_seconds(df["timestamp"])

        df = df.dropna(subset=["timestamp"]).sort_values(["user_id", "timestamp"])
        df = df.groupby("user_id").filter(lambda x: len(x) >= self.cfg.min_interactions_per_user)
        if df.empty:
            raise ValueError(
                "No usable Junyi records after preprocessing. "
                "Please check timestamp/elapsed_time fields or lower min_interactions_per_user."
            )
        return df

    def build_exercise_graph(self, exercise_csv: str | Path) -> Tuple[np.ndarray, Dict[str, int]]:
        ex_df = pd.read_csv(exercise_csv)
        if "name" not in ex_df.columns:
            raise ValueError("junyi_Exercise_table.csv requires column `name`.")
        if "prerequisites" not in ex_df.columns:
            ex_df["prerequisites"] = ""

        for name in ex_df["name"].astype(str).unique():
            if name not in self.exercise2id:
                self.exercise2id[name] = len(self.exercise2id)

        n = len(self.exercise2id)
        adj = np.eye(n, dtype=np.float32)

        for _, row in ex_df.iterrows():
            cur = str(row["name"])
            cur_id = self.exercise2id.get(cur)
            prereq_text = str(row.get("prerequisites", ""))
            prereqs = [p.strip() for p in prereq_text.replace(";", ",").split(",") if p.strip()]
            for p in prereqs:
                if p not in self.exercise2id:
                    self.exercise2id[p] = len(self.exercise2id)
                    new_n = len(self.exercise2id)
                    new_adj = np.eye(new_n, dtype=np.float32)
                    new_adj[: adj.shape[0], : adj.shape[1]] = adj
                    adj = new_adj
                p_id = self.exercise2id[p]
                adj[cur_id, p_id] = 1.0
                adj[p_id, cur_id] = 1.0

        deg = adj.sum(axis=1, keepdims=True)
        adj = adj / np.maximum(deg, 1.0)
        return adj, self.exercise2id

    def _encode_exercise(self, name: str) -> int:
        if name not in self.exercise2id:
            self.exercise2id[name] = len(self.exercise2id)
        return self.exercise2id[name]

    def build_samples(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        elapsed_raw = np.asarray(df["elapsed_time"].values, dtype=np.float64)
        hint_raw = np.asarray(df["hint_used"].values, dtype=np.float64)
        elapsed_scale = np.log1p(np.clip(np.nan_to_num(elapsed_raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1e7))
        hint_scale = np.log1p(np.clip(np.nan_to_num(hint_raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1e4))
        elapsed_norm = self._safe_standardize(elapsed_scale)
        hint_norm = self._safe_standardize(hint_scale)

        rows_ex, rows_resp, rows_cont, rows_mask = [], [], [], []
        rows_y, rows_target_ex = [], []

        # Pre-materialize normalized columns to avoid repeated unstable statistics in group loops.
        work_df = df.copy()
        work_df["_elapsed_norm"] = elapsed_norm
        work_df["_hint_norm"] = hint_norm

        for _, g in work_df.groupby("user_id"):
            g = g.sort_values("timestamp")
            ex = [self._encode_exercise(str(x)) for x in g["exercise_id"].tolist()]
            resp = g["correct"].astype(int).tolist()
            elapsed = np.asarray(g["_elapsed_norm"].values, dtype=np.float32)
            hints = np.asarray(g["_hint_norm"].values, dtype=np.float32)

            for i in range(self.cfg.min_history, len(g)):
                hist_ex = ex[max(0, i - self.cfg.max_seq_len): i]
                hist_resp = resp[max(0, i - self.cfg.max_seq_len): i]
                hist_elapsed = elapsed[max(0, i - self.cfg.max_seq_len): i]
                hist_hints = hints[max(0, i - self.cfg.max_seq_len): i]

                L = len(hist_ex)
                pad = self.cfg.max_seq_len - L

                rows_ex.append(([0] * pad) + hist_ex)
                rows_resp.append(([0] * pad) + hist_resp)
                cont = np.zeros((self.cfg.max_seq_len, 2), dtype=np.float32)
                if L > 0:
                    cont[pad:, 0] = hist_elapsed
                    cont[pad:, 1] = hist_hints
                rows_cont.append(cont)
                m = np.zeros((self.cfg.max_seq_len,), dtype=np.float32)
                if L > 0:
                    m[pad:] = 1.0
                rows_mask.append(m)

                rows_y.append(float(resp[i]))
                rows_target_ex.append(int(ex[i]))

        return {
            "exercise_ids": np.asarray(rows_ex, dtype=np.int64),
            "response_ids": np.asarray(rows_resp, dtype=np.int64),
            "continuous": np.asarray(rows_cont, dtype=np.float32),
            "mask": np.asarray(rows_mask, dtype=np.float32),
            "target": np.asarray(rows_y, dtype=np.float32),
            "target_exercise": np.asarray(rows_target_ex, dtype=np.int64),
        }

    def build_dataset(self, log_csv: str | Path) -> JunyiSequenceDataset:
        df = self.load_logs(log_csv)
        samples = self.build_samples(df)
        return JunyiSequenceDataset(samples)


def split_by_exercise(samples: Dict[str, np.ndarray], train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    ex_ids = np.unique(samples["target_exercise"])
    ex_ids = np.sort(ex_ids)
    cut = max(1, int(len(ex_ids) * train_ratio))
    src = set(ex_ids[:cut].tolist())
    train_idx = np.where(np.isin(samples["target_exercise"], list(src)))[0]
    test_idx = np.where(~np.isin(samples["target_exercise"], list(src)))[0]
    return train_idx, test_idx


def index_samples(samples: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[indices] for k, v in samples.items()}
