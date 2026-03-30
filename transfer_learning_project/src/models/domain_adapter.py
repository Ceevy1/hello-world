"""Domain adaptation components and domain-shift metrics."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import QuantileTransformer


class CORALAdapter:
    """CORAL domain adaptation implementation."""

    def __init__(self, lambda_coral: float = 1.0):
        self.lambda_coral = lambda_coral

    def coral_loss(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        d = source_features.shape[1]
        cs = self._covariance(source_features)
        ct = self._covariance(target_features)
        loss = torch.norm(cs - ct, p="fro") ** 2 / (4 * d * d)
        return self.lambda_coral * loss

    def _covariance(self, features: torch.Tensor) -> torch.Tensor:
        n = features.shape[0]
        centered = features - features.mean(dim=0, keepdim=True)
        return (centered.T @ centered) / (n - 1)

    def align(self, X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        cs = np.cov(X_source.T) + 1e-8 * np.eye(X_source.shape[1])
        ct = np.cov(X_target.T) + 1e-8 * np.eye(X_target.shape[1])
        ws = np.linalg.cholesky(cs)
        wt = np.linalg.cholesky(ct)
        return X_target @ np.linalg.inv(wt).T @ ws.T


class TCAAdapter:
    """Simple TCA-like projection via KernelPCA shared subspace."""

    def __init__(self, n_components: int = 10, kernel: str = "rbf") -> None:
        self.kpca = KernelPCA(n_components=n_components, kernel=kernel)

    def fit_transform(self, X_source: np.ndarray, X_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        combined = np.vstack([X_source, X_target])
        embedded = self.kpca.fit_transform(combined)
        return embedded[: len(X_source)], embedded[len(X_source) :]


class QuantileAligner:
    """Marginal-distribution alignment with quantile transform."""

    def __init__(self, n_quantiles: int = 100):
        self.qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal")

    def fit_transform(self, X_source: np.ndarray, X_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs = self.qt.fit_transform(X_source)
        Xt = self.qt.transform(X_target)
        return Xs, Xt


def compute_mmd(X_source: np.ndarray, X_target: np.ndarray, gamma: float = 1.0) -> float:
    k_ss = rbf_kernel(X_source, X_source, gamma=gamma).mean()
    k_tt = rbf_kernel(X_target, X_target, gamma=gamma).mean()
    k_st = rbf_kernel(X_source, X_target, gamma=gamma).mean()
    return float(k_ss + k_tt - 2 * k_st)


def compute_a_distance(X_source: np.ndarray, X_target: np.ndarray) -> float:
    X = np.vstack([X_source, X_target])
    y = np.array([0] * len(X_source) + [1] * len(X_target))
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    eps = 1 - acc
    return float(2 * (1 - 2 * eps))


def compute_feature_fid(X_source: np.ndarray, X_target: np.ndarray) -> float:
    mu_s, mu_t = X_source.mean(axis=0), X_target.mean(axis=0)
    cov_s, cov_t = np.cov(X_source, rowvar=False), np.cov(X_target, rowvar=False)
    mean_dist = np.sum((mu_s - mu_t) ** 2)
    cov_prod = cov_s @ cov_t
    eigvals = np.linalg.eigvals(cov_prod)
    sqrt_trace = np.sum(np.sqrt(np.maximum(eigvals.real, 0)))
    return float(mean_dist + np.trace(cov_s + cov_t - 2 * sqrt_trace))
