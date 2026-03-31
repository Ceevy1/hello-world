from pathlib import Path
import pandas as pd


REQUIRED_COLS = [
    '考勤', '练习1', '练习2', '练习3',
    '实验1', '实验2', '实验3', '实验4', '实验5', '实验6', '实验7',
    '报告', '总平时 成绩', '总实验 成绩', '平时成绩', '总期末成绩', '总评成绩'
]


def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Dataset not found: {path}')
    if p.suffix.lower() in {'.xlsx', '.xls'}:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    return df.copy()
