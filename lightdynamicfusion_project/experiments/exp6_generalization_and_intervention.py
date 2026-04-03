from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.data.loader import load_dataset
from src.model.dynamic_fusion import LightDynamicFusion


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def run(cfg, debug: bool = False):
    df = load_dataset(cfg['data']['path'])
    X = df.drop(columns=[cfg['data']['target_col']])
    y = df[cfg['data']['target_col']].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg['project']['seed']
    )

    model = LightDynamicFusion(stage='T4', random_state=cfg['project']['seed'])
    model.fit(X_train, y_train)

    in_domain = _rmse(y_test, model.predict(X_test))

    X_shift = X_test.copy()
    for c in ['考勤', '练习1', '练习2', '练习3']:
        if c in X_shift.columns:
            X_shift[c] = np.clip(X_shift[c] * 0.9, 0, 100)
    cross_domain = _rmse(y_test, model.predict(X_shift))

    generalization = pd.DataFrame([
        {'train': 'student_scores', 'test': 'student_scores', 'model': 'DynamicFusion', 'rmse': in_domain},
        {'train': 'student_scores', 'test': 'shifted_student_scores', 'model': 'DynamicFusion', 'rmse': cross_domain},
    ])
    generalization['drop_rate'] = (generalization['rmse'] - in_domain) / in_domain

    interventions = []
    base_pred = model.predict(X_test)

    x_att = X_test.copy()
    if '考勤' in x_att.columns:
        x_att['考勤'] = np.clip(x_att['考勤'] + 10, 0, 100)
    interventions.append({'intervention': '提高出勤', 'delta_pred': float(np.mean(model.predict(x_att) - base_pred))})

    x_late = X_test.copy()
    for c in ['练习1', '练习2', '练习3']:
        if c in x_late.columns:
            x_late[c] = np.clip(x_late[c] + 5, 0, 100)
    interventions.append({'intervention': '减少延迟', 'delta_pred': float(np.mean(model.predict(x_late) - base_pred))})

    intervention_df = pd.DataFrame(interventions)

    out_dir = Path(cfg['project']['output_dir']) / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    generalization.to_csv(out_dir / 'exp6_generalization.csv', index=False)
    intervention_df.to_csv(out_dir / 'exp6_intervention.csv', index=False)
    return generalization
