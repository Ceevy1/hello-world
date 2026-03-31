from pathlib import Path
import pandas as pd

from src.data.loader import load_dataset
from src.model.dynamic_fusion import LightDynamicFusion


def run(cfg, debug: bool = False):
    df = load_dataset(cfg['data']['path'])
    X = df.drop(columns=[cfg['data']['target_col']])
    y = df[cfg['data']['target_col']].to_numpy(dtype=float)

    model = LightDynamicFusion(stage='T3', random_state=cfg['project']['seed'])
    model.fit(X, y)
    contrib = model.get_source_contributions(X)

    p = Path(cfg['project']['output_dir']) / 'tables' / 'exp4_source_contributions.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    contrib.to_csv(p, index=False)
    return contrib
