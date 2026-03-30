"""Checkpoint save/load utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


def save_checkpoint(obj: Any, output_dir: str, prefix: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = path / f"{prefix}_{ts}.pkl"
    joblib.dump(obj, out)
    return out


def load_checkpoint(path: str) -> Any:
    return joblib.load(path)
