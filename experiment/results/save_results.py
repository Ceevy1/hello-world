from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_results(results, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False, encoding="utf-8")
