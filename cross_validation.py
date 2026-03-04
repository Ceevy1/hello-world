"""
Cross-Validation & Model Selection Utilities
Supports k-fold CV with full reproducibility.
"""

import logging
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, List, Callable, Tuple
from config import SEED, CV_FOLDS

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    K-Fold cross-validation for any model trainer.
    Ensures strict data leakage prevention and reproducibility.
    """

    def __init__(self, n_splits: int = CV_FOLDS, seed: int = SEED, stratify: bool = False):
        self.n_splits = n_splits
        self.seed = seed
        self.stratify = stratify
        if stratify:
            self.kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_fn: Callable,
        predict_fn: Callable,
        metric_fn: Callable,
        stratify_y: np.ndarray = None,
    ) -> Dict:
        """
        Args:
            X:           Input features (N, ...)
            y:           Target values (N,)
            model_fn:    Callable(X_train, y_train) -> trained model
            predict_fn:  Callable(model, X_test) -> predictions
            metric_fn:   Callable(y_true, y_pred) -> dict of metrics
        Returns:
            Dict with per-fold and aggregate metrics
        """
        all_metrics: List[Dict] = []
        all_preds = np.zeros(len(y))
        
        splitter = self.kf
        split_y = stratify_y if (self.stratify and stratify_y is not None) else y

        for fold, (train_idx, val_idx) in enumerate(splitter.split(X, split_y)):
            logger.info(f"  Fold {fold+1}/{self.n_splits}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            np.random.seed(self.seed + fold)
            model = model_fn(X_train, y_train)
            preds = predict_fn(model, X_val)
            metrics = metric_fn(y_val, preds)
            all_metrics.append(metrics)
            all_preds[val_idx] = preds

            logger.info(f"    Metrics: {metrics}")

        # Aggregate
        agg = {}
        for key in all_metrics[0].keys():
            vals = [m[key] for m in all_metrics]
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        return {
            "fold_metrics": all_metrics,
            "aggregate": agg,
            "oof_predictions": all_preds,
        }

    def format_results(self, cv_result: Dict) -> str:
        """Format CV results as readable string."""
        lines = []
        for k, v in cv_result["aggregate"].items():
            lines.append(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
        return "\n".join(lines)


class HyperparamSearch:
    """
    Simple grid search over model configurations.
    """

    def __init__(self, param_grid: Dict, seed: int = SEED):
        self.param_grid = param_grid
        self.seed = seed

    def _expand_grid(self) -> List[Dict]:
        import itertools
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = list(itertools.product(*values))
        return [dict(zip(keys, c)) for c in combos]

    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_class,
        metric_key: str = "RMSE",
        minimize: bool = True,
    ) -> Tuple[Dict, float]:
        """
        Grid search. Returns best config and best metric.
        """
        from evaluation import compute_regression_metrics

        configs = self._expand_grid()
        logger.info(f"Hyperparameter search: {len(configs)} configurations")

        best_config = None
        best_score = float("inf") if minimize else float("-inf")

        for i, config in enumerate(configs):
            try:
                model = model_class(config)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                metrics = compute_regression_metrics(y_val, preds)
                score = metrics[metric_key]

                is_better = (score < best_score) if minimize else (score > best_score)
                if is_better:
                    best_score = score
                    best_config = config
                    logger.info(f"  Config {i+1}: {metric_key}={score:.4f} [NEW BEST] {config}")
                else:
                    logger.debug(f"  Config {i+1}: {metric_key}={score:.4f}")

            except Exception as e:
                logger.warning(f"Config {i+1} failed: {e}")

        logger.info(f"Best config: {best_config}, best {metric_key}={best_score:.4f}")
        return best_config, best_score


def set_all_seeds(seed: int = SEED):
    """Ensure full reproducibility."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"All seeds set to {seed}")
