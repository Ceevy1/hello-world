import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, cohen_kappa_score


def compute_metrics(y_true, y_pred, task='regression'):
    if task == 'regression':
        return {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
        }

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
    }
