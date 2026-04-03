from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.data.loader import load_dataset
from src.model.dynamic_fusion import LightDynamicFusion


def _permutation_importance(model, X, y, random_state=42):
    rng = np.random.default_rng(random_state)
    base_pred = model.predict(X)
    base_rmse = float(np.sqrt(mean_squared_error(y, base_pred)))
    rows = []
    for col in X.columns:
        Xp = X.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        pred = model.predict(Xp)
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        rows.append({'feature': col, 'importance': max(0.0, rmse - base_rmse)})
    return pd.DataFrame(rows).sort_values('importance', ascending=False)


def run(cfg, debug: bool = False):
    df = load_dataset(cfg['data']['path'])
    X = df.drop(columns=[cfg['data']['target_col']])
    y = df[cfg['data']['target_col']].to_numpy(dtype=float)

    model = LightDynamicFusion(stage='T4', random_state=cfg['project']['seed'])
    model.fit(X, y)

    contrib = model.get_source_contributions(X)
    attn = pd.DataFrame([model.attention_fusion.get_attention_weights('T4')])

    # SHAP-like permutation importance on original and shifted data for cross-setting consistency.
    self_imp = _permutation_importance(model, X, y, random_state=cfg['project']['seed'])
    X_shift = X.copy()
    for c in X_shift.columns:
        X_shift[c] = X_shift[c] * 0.95
    shifted_imp = _permutation_importance(model, X_shift, y, random_state=cfg['project']['seed'] + 1)

    shap_compare = self_imp.merge(shifted_imp, on='feature', suffixes=('_self', '_shifted'))
    shap_compare = shap_compare.rename(columns={'importance_self': 'shap_value_self', 'importance_shifted': 'shap_value_shifted'})

    out_dir = Path(cfg['project']['output_dir']) / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    contrib.to_csv(out_dir / 'exp4_source_contributions.csv', index=False)
    attn.to_csv(out_dir / 'exp4_attention_weights.csv', index=False)
    shap_compare.to_csv(out_dir / 'exp4_shap_comparison.csv', index=False)
    return shap_compare
