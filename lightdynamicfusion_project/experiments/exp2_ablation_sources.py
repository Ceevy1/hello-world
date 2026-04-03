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
    common = dict(
        stage='T4',
        base_estimator=cfg['model']['base_estimator'],
        use_data_augmentation=cfg['augmentation']['enabled'],
        aug_noise_std=cfg['augmentation']['noise_std'],
        aug_multiplier=cfg['augmentation']['multiplier'],
        aug_method=cfg['augmentation']['method'],
        random_state=cfg['project']['seed'],
    )

    variants = [
        ('Full model', {'active_sources': ['source1', 'source2', 'source3'], 'use_attention': True}),
        ('-behavior branch', {'active_sources': ['source2', 'source3'], 'use_attention': True}),
        ('-performance branch', {'active_sources': ['source1', 'source2'], 'use_attention': True}),
        ('-attention', {'active_sources': ['source1', 'source2', 'source3'], 'use_attention': False}),
        ('single branch', {'active_sources': ['source1'], 'use_attention': False}),
    ]

    rows = []
    for name, opts in variants:
        model = LightDynamicFusion(**common, **opts)
        metrics = cv.run_cv(model, X, y)
        rows.append({
            'variant': name,
            'rmse_mean': metrics['rmse']['mean'],
            'mae_mean': metrics['mae']['mean'],
            'r2_mean': metrics['r2']['mean'],
        })

    out = pd.DataFrame(rows).sort_values('rmse_mean').reset_index(drop=True)
    p = Path(cfg['project']['output_dir']) / 'tables' / 'exp2_ablation_results.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out
