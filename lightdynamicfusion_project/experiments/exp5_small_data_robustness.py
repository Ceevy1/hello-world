from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.data.loader import load_dataset
from src.model.dynamic_fusion import LightDynamicFusion


try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


def run(cfg, debug: bool = False):
    df = load_dataset(cfg['data']['path'])
    X = df.drop(columns=[cfg['data']['target_col']]).reset_index(drop=True)
    y = df[cfg['data']['target_col']].to_numpy(dtype=float)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg['project']['seed']
    )
    y_train_full = pd.Series(y_train_full).reset_index(drop=True)
    X_train_full = X_train_full.reset_index(drop=True)

    sizes = [1.0, 0.5, 0.3, 0.1]
    rows = []
    for frac in sizes:
        n = max(2, int(len(X_train_full) * frac))
        replace = n > len(X_train_full)
        subset = X_train_full.sample(n=n, random_state=cfg['project']['seed'], replace=replace)
        y_sub = y_train_full.iloc[subset.index].to_numpy()

        rf = RandomForestRegressor(n_estimators=200, random_state=cfg['project']['seed'])
        rf.fit(subset, y_sub)
        rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf.predict(X_test))))

        if XGBRegressor is not None:
            xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=cfg['project']['seed'])
        else:
            xgb = GradientBoostingRegressor(random_state=cfg['project']['seed'])
        xgb.fit(subset, y_sub)
        xgb_rmse = float(np.sqrt(mean_squared_error(y_test, xgb.predict(X_test))))

        df_model = LightDynamicFusion(stage='T4', random_state=cfg['project']['seed'])
        df_model.fit(subset, y_sub)
        df_rmse = float(np.sqrt(mean_squared_error(y_test, df_model.predict(X_test))))

        rows.append({'data_size': f'{int(frac*100)}%', 'RF': rf_rmse, 'XGB': xgb_rmse, 'DynamicFusion': df_rmse})

    out = pd.DataFrame(rows)
    p = Path(cfg['project']['output_dir']) / 'tables' / 'exp5_small_data_robustness.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out
