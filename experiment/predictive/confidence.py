from __future__ import annotations

import numpy as np



def compute_confidence(preds: np.ndarray) -> np.ndarray:
    preds = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)
    eps = 1e-8
    entropy = -(preds * np.log(preds + eps) + (1 - preds) * np.log(1 - preds + eps))
    max_entropy = np.log(2.0)
    return 1.0 - entropy / max_entropy
