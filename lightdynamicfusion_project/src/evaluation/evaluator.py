import pandas as pd


def flatten_metrics(name: str, metrics: dict) -> pd.DataFrame:
    row = {'model': name}
    for metric, values in metrics.items():
        row[f'{metric}_mean'] = values['mean']
        row[f'{metric}_std'] = values['std']
    return pd.DataFrame([row])
