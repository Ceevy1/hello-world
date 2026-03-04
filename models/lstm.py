from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :])).squeeze(-1)


@dataclass
class LSTMConfig:
    hidden_dim: int = 128
    dropout: float = 0.3
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3


class LSTMTrainer:
    def __init__(self, input_dim: int, config: LSTMConfig = LSTMConfig(), device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMRegressor(input_dim, config.hidden_dim, config.dropout).to(self.device)

    def fit(self, x_seq: np.ndarray, y: np.ndarray) -> None:
        use_cuda = self.device.type == "cuda"
        ds = TensorDataset(torch.FloatTensor(x_seq), torch.FloatTensor(y))
        dl = DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=use_cuda,
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.config.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device, non_blocking=use_cuda)
                yb = yb.to(self.device, non_blocking=use_cuda)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        self.model.eval()
        use_cuda = self.device.type == "cuda"
        with torch.no_grad():
            x = torch.FloatTensor(x_seq).to(self.device, non_blocking=use_cuda)
            return self.model(x).cpu().numpy()
