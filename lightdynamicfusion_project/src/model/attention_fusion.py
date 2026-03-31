import numpy as np
from scipy.optimize import minimize


class AttentionFusion:
    def __init__(self, n_sources: int = 3, method: str = 'learned_stage'):
        self.n_sources = n_sources
        self.method = method
        self.stage_weights = {}
        self.source_order = ['source1', 'source2', 'source3']

    def fit(self, group_preds: dict, y_true: np.ndarray, stage: str):
        active = [k for k in self.source_order if k in group_preds]
        preds_matrix = np.column_stack([group_preds[k] for k in active])

        def objective(weights):
            w = np.exp(weights) / np.sum(np.exp(weights))
            return float(np.mean((preds_matrix @ w - y_true) ** 2))

        result = minimize(objective, x0=np.zeros(len(active)))
        soft = np.exp(result.x) / np.sum(np.exp(result.x))
        full = np.zeros(self.n_sources)
        for i, s in enumerate(active):
            full[self.source_order.index(s)] = soft[i]
        self.stage_weights[stage] = full

    def predict(self, group_preds: dict, stage: str) -> np.ndarray:
        weights = self.stage_weights.get(stage, np.ones(self.n_sources) / self.n_sources)
        preds = []
        for idx, source in enumerate(self.source_order):
            if source in group_preds:
                preds.append(group_preds[source] * weights[idx])
        if not preds:
            raise ValueError('No source predictions provided')
        return np.sum(np.column_stack(preds), axis=1)

    def get_attention_weights(self, stage: str) -> dict:
        w = self.stage_weights.get(stage, np.ones(self.n_sources) / self.n_sources)
        return {s: float(w[i]) for i, s in enumerate(self.source_order)}
