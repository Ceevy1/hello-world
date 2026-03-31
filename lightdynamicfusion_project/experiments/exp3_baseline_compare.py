from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error

from src.data.loader import load_dataset


def run(cfg, debug: bool = False):
    df = load_dataset(cfg['data']['path'])
    X = df.drop(columns=[cfg['data']['target_col']])
    y = df[cfg['data']['target_col']].to_numpy(dtype=float)

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=cfg['project']['seed']),
    }

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    rows = []
    for name, model in models.items():
        scores = -cross_val_score(model, X, y, cv=5, scoring=scorer)
        rows.append({'model': name, 'mae_mean': scores.mean(), 'mae_std': scores.std()})
    out = pd.DataFrame(rows)
    p = Path(cfg['project']['output_dir']) / 'tables' / 'exp3_baselines.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out
