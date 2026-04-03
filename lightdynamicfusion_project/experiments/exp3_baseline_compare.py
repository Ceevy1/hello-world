from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.loader import load_dataset
from src.model.dynamic_fusion import LightDynamicFusion


try:
    from xgboost import XGBRegressor
except Exception:  # fallback when xgboost is unavailable
    XGBRegressor = None


def _estimate_params(model) -> int:
    if hasattr(model, 'coefs_') and model.coefs_:
        return int(sum(np.prod(w.shape) for w in model.coefs_) + sum(len(b) for b in model.intercepts_))
    if hasattr(model, 'estimators_'):
        estimators = model.estimators_
        if isinstance(estimators, np.ndarray):
            return int(estimators.size)
        return int(len(estimators))
    if hasattr(model, 'coef_'):
        coef = model.coef_
        return int(np.size(coef) + (1 if not hasattr(model, 'intercept_') else np.size(model.intercept_)))
    return 0


def _cv_eval_sklearn(model, X, y, seed):
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rmses, maes, r2s = [], [], []
    train_time = 0.0
    params = []
    for tr, va in cv.split(X):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X.iloc[tr], y[tr])
        train_time += time.perf_counter() - t0
        pred = m.predict(X.iloc[va])
        rmses.append(np.sqrt(mean_squared_error(y[va], pred)))
        maes.append(mean_absolute_error(y[va], pred))
        r2s.append(0.0 if len(va) < 2 else r2_score(y[va], pred))
        params.append(_estimate_params(m))
    return float(np.mean(rmses)), float(np.mean(maes)), float(np.mean(r2s)), float(train_time / 5), int(np.max(params))


def _cv_eval_df(cfg, X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=cfg['project']['seed'])
    rmses, maes, r2s = [], [], []
    train_time = 0.0
    for tr, va in cv.split(X):
        model = LightDynamicFusion(
            stage='T4',
            base_estimator=cfg['model']['base_estimator'],
            use_data_augmentation=cfg['augmentation']['enabled'],
            aug_noise_std=cfg['augmentation']['noise_std'],
            aug_multiplier=cfg['augmentation']['multiplier'],
            aug_method=cfg['augmentation']['method'],
            random_state=cfg['project']['seed'],
        )
        t0 = time.perf_counter()
        model.fit(X.iloc[tr], y[tr])
        train_time += time.perf_counter() - t0
        pred = model.predict(X.iloc[va])
        rmses.append(np.sqrt(mean_squared_error(y[va], pred)))
        maes.append(mean_absolute_error(y[va], pred))
        r2s.append(0.0 if len(va) < 2 else r2_score(y[va], pred))
    return float(np.mean(rmses)), float(np.mean(maes)), float(np.mean(r2s)), float(train_time / 5), len(model.group_predictors)


def run(cfg, debug: bool = False):
    df = load_dataset(cfg['data']['path'])
    X = df.drop(columns=[cfg['data']['target_col']])
    y = df[cfg['data']['target_col']].to_numpy(dtype=float)

    models = {
        'Linear': LinearRegression(),
        'RF': RandomForestRegressor(n_estimators=200, max_depth=6, random_state=cfg['project']['seed']),
        'MLP': MLPRegressor(hidden_layer_sizes=(32,), max_iter=1000, random_state=cfg['project']['seed']),
    }
    if XGBRegressor is not None:
        models['XGB'] = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=cfg['project']['seed']
        )
    else:
        models['XGB'] = GradientBoostingRegressor(random_state=cfg['project']['seed'])

    rows = []
    for name, model in models.items():
        rmse, mae, r2, train_t, params = _cv_eval_sklearn(model, X, y, cfg['project']['seed'])
        rows.append({'model': name, 'dataset': 'student_scores', 'rmse': rmse, 'mae': mae, 'r2': r2,
                     'params': params, 'train_time_sec': train_t})

    rmse, mae, r2, train_t, params = _cv_eval_df(cfg, X, y)
    rows.append({'model': 'DynamicFusion', 'dataset': 'student_scores', 'rmse': rmse, 'mae': mae, 'r2': r2,
                 'params': params, 'train_time_sec': train_t})

    out = pd.DataFrame(rows).sort_values('rmse').reset_index(drop=True)
    p = Path(cfg['project']['output_dir']) / 'tables' / 'exp3_baselines.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out
