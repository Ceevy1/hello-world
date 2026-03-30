from pathlib import Path

import pandas as pd

from src.data.data_loader import DataLoader


def test_load_custom_dataset_csv(tmp_path: Path):
    f = tmp_path / "sample.csv"
    pd.DataFrame({"考分f": [80], "总评成绩": [85]}).to_csv(f, index=False)
    loader = DataLoader({"exam_score": "考分f", "target": "总评成绩"})
    df = loader.load_custom_dataset(str(f))
    assert "exam_score" in df.columns
    assert "target" in df.columns
