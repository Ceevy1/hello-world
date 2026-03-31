"""Dynamic fusion regressor with branch attention."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _FusionNet(nn.Module):
    def __init__(self, d_perf: int, d_behav: int, d_eng: int, hidden: int = 64) -> None:
        super().__init__()
        self.e_perf = nn.Linear(d_perf, hidden)
        self.e_behav = nn.Linear(d_behav, hidden)
        self.e_eng = nn.Linear(d_eng, hidden)
        self.attn = nn.Sequential(nn.Linear(hidden * 3, 3), nn.Softmax(dim=1))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_perf: torch.Tensor, x_behav: torch.Tensor, x_eng: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = torch.relu(self.e_perf(x_perf))
        h2 = torch.relu(self.e_behav(x_behav))
        h3 = torch.relu(self.e_eng(x_eng))
        weights = self.attn(torch.cat([h1, h2, h3], dim=1))
        fused = weights[:, 0:1] * h1 + weights[:, 1:2] * h2 + weights[:, 2:3] * h3
        pred = self.head(fused).squeeze(1)
        return pred, weights


class DynamicFusionRegressor:
    def __init__(self, input_dims: tuple[int, int, int], lr: float = 1e-3, batch_size: int = 32, epochs: int = 60) -> None:
        self.input_dims = input_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _FusionNet(*input_dims).to(self.device)

    def fit(self, X_perf: np.ndarray, X_behav: np.ndarray, X_eng: np.ndarray, y: np.ndarray) -> "DynamicFusionRegressor":
        ds = TensorDataset(
            torch.tensor(X_perf, dtype=torch.float32),
            torch.tensor(X_behav, dtype=torch.float32),
            torch.tensor(X_eng, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for x1, x2, x3, yt in loader:
                x1, x2, x3, yt = x1.to(self.device), x2.to(self.device), x3.to(self.device), yt.to(self.device)
                pred, _ = self.model(x1, x2, x3)
                loss = loss_fn(pred, yt)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_with_attention(self, X_perf: np.ndarray, X_behav: np.ndarray, X_eng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            x1 = torch.tensor(X_perf, dtype=torch.float32, device=self.device)
            x2 = torch.tensor(X_behav, dtype=torch.float32, device=self.device)
            x3 = torch.tensor(X_eng, dtype=torch.float32, device=self.device)
            pred, weights = self.model(x1, x2, x3)
        return pred.cpu().numpy(), weights.cpu().numpy()

    def predict(self, X_perf: np.ndarray, X_behav: np.ndarray, X_eng: np.ndarray) -> np.ndarray:
        return self.predict_with_attention(X_perf, X_behav, X_eng)[0]
