from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class ExperimentLogger:
    """Track epoch-wise losses and persist CSV artifacts (text-only)."""

    output_dir: str = "outputs"
    history: List[Dict[str, float]] = field(default_factory=list)

    def log_epoch(self, epoch: int, **losses: float) -> None:
        row = {"epoch": epoch}
        row.update({k: float(v) for k, v in losses.items()})
        self.history.append(row)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def save(self, filename: str = "loss_history.csv") -> pd.DataFrame:
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_frame()
        df.to_csv(out_dir / filename, index=False)
        return df

    def export_loss_series(self, loss_cols: Dict[str, str], curves_dir: str = "loss_curves") -> None:
        """Export one CSV per loss curve (project forbids binary outputs)."""
        out_dir = Path(self.output_dir) / curves_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_frame()
        if df.empty:
            return
        for col, name in loss_cols.items():
            if col not in df.columns:
                continue
            series_df = df[["epoch", col]].copy()
            series_df.to_csv(out_dir / name, index=False)
