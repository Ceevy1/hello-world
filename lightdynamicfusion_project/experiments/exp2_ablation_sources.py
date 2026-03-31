from pathlib import Path
import pandas as pd


def run(cfg, debug: bool = False):
    # Placeholder scaffold: store ablation plan for reproducibility.
    rows = [
        {'group': 'ABL-S1', 'desc': 'source1 only'},
        {'group': 'ABL-S2', 'desc': 'source2 only'},
        {'group': 'ABL-S3', 'desc': 'source3 only'},
        {'group': 'ABL-S12', 'desc': 'source1+2 equal'},
        {'group': 'ABL-S12-Attn', 'desc': 'source1+2 attention'},
        {'group': 'ABL-Full-Equal', 'desc': 'all equal'},
        {'group': 'ABL-Full-Attn', 'desc': 'all attention'},
    ]
    out = pd.DataFrame(rows)
    p = Path(cfg['project']['output_dir']) / 'tables' / 'exp2_ablation_plan.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out
