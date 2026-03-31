from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, RepeatedStratifiedKFold

from src.evaluation.metrics import compute_metrics


class SmallSampleCVHandler:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _get_stratify_labels(self, y):
        y = np.asarray(y)
        q = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
        return np.digitize(y, q)

    def get_cv_strategy(self, n_samples: int):
        if n_samples < 30:
            return LeaveOneOut()
        if n_samples < 100:
            return RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.random_state)
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

    def run_cv(self, model, X: pd.DataFrame, y: np.ndarray, task: str = 'regression') -> dict:
        cv = self.get_cv_strategy(len(y))
        metrics_all = defaultdict(list)
        strat = self._get_stratify_labels(y)

        for train_idx, val_idx in cv.split(X, strat):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            fold_metrics = compute_metrics(y_val, y_pred, task)
            for k, v in fold_metrics.items():
                metrics_all[k].append(v)

        return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'all_folds': v} for k, v in metrics_all.items()}
