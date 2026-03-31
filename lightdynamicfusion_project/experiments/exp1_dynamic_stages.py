from pathlib import Path
import pandas as pd

from src.data.loader import load_dataset
from src.model.dynamic_fusion import LightDynamicFusion
from src.training.cv_handler import SmallSampleCVHandler


def run(cfg, debug: bool = False):
    df = load_dataset(cfg['data']['path'])
    y = df[cfg['data']['target_col']].to_numpy(dtype=float)
    X = df.drop(columns=[cfg['data']['target_col']])

    cv = SmallSampleCVHandler(random_state=cfg['project']['seed'])
    rows = []
    for stage in cfg['model']['stages']:
        model = LightDynamicFusion(stage=stage, base_estimator=cfg['model']['base_estimator'],
                                   use_data_augmentation=cfg['augmentation']['enabled'],
                                   aug_noise_std=cfg['augmentation']['noise_std'],
                                   aug_multiplier=cfg['augmentation']['multiplier'],
                                   aug_method=cfg['augmentation']['method'],
                                   random_state=cfg['project']['seed'])
        metrics = cv.run_cv(model, X, y)
        rows.append({'stage': stage, 'mae_mean': metrics['mae']['mean'], 'mae_std': metrics['mae']['std'],
                     'rmse_mean': metrics['rmse']['mean'], 'r2_mean': metrics['r2']['mean']})
    out = pd.DataFrame(rows)
    p = Path(cfg['project']['output_dir']) / 'tables' / 'exp1_dynamic_stages.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out
