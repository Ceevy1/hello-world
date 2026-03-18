from __future__ import annotations

import pandas as pd


def to_latex(df: pd.DataFrame) -> str:
    return df.to_latex(index=False, float_format="%.4f")
