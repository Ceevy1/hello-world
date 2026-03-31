import numpy as np
import pandas as pd


def gaussian_augment(X: pd.DataFrame, y: np.ndarray, std: float = 1.5, multiplier: int = 5, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    X_num = X.copy()
    for c in X_num.columns:
        X_num[c] = pd.to_numeric(X_num[c], errors='coerce').fillna(0.0)

    X_all = [X_num]
    y_all = [y]
    for _ in range(max(0, multiplier - 1)):
        noise = rng.normal(0, std, size=X_num.shape)
        X_all.append(X_num + noise)
        y_all.append(y + rng.normal(0, std * 0.5, size=len(y)))
    return pd.concat(X_all, ignore_index=True), np.concatenate(y_all)


def mixup_augment(X: pd.DataFrame, y: np.ndarray, alpha: float = 0.2, n_new: int | None = None, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    n = len(X)
    n_new = n if n_new is None else n_new
    lam = rng.beta(alpha, alpha, size=n_new)
    i = rng.integers(0, n, size=n_new)
    j = rng.integers(0, n, size=n_new)

    Xv = X.to_numpy(dtype=float)
    X_mix = lam[:, None] * Xv[i] + (1 - lam)[:, None] * Xv[j]
    y_mix = lam * y[i] + (1 - lam) * y[j]

    X_new = pd.concat([X.reset_index(drop=True), pd.DataFrame(X_mix, columns=X.columns)], ignore_index=True)
    y_new = np.concatenate([y, y_mix])
    return X_new, y_new
