from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from evaluation.metrics import regression_metrics
from loss.unified_loss import UnifiedLoss, UnifiedLossConfig
from models.cat import CatModel
from models.hafm import HAFM
from models.lstm import LSTMTrainer
from models.xgb import XGBModel


@dataclass
class TrainOutputs:
    predictions: Dict[str, np.ndarray]
    metrics: Dict[str, Dict[str, float]]


def _train_hafm(
    x_tab_train: np.ndarray,
    base_preds_train: np.ndarray,
    y_train: np.ndarray,
    loss_cfg: UnifiedLossConfig,
    epochs: int = 100,
    lr: float = 1e-3,
) -> HAFM:
    model = HAFM(input_dim=x_tab_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    unified = UnifiedLoss(loss_cfg)

    xt = torch.FloatTensor(x_tab_train)
    bt = torch.FloatTensor(base_preds_train)
    yt = torch.FloatTensor(y_train)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        weights = model(xt)
        y_hat = (weights * bt).sum(dim=1)
        loss = unified.total_loss(y_hat, yt, {}, bt, weights)
        loss.backward()
        opt.step()
    return model


def train_full_pipeline(
    x_seq_train: np.ndarray,
    x_tab_train: np.ndarray,
    y_train: np.ndarray,
    x_seq_test: np.ndarray,
    x_tab_test: np.ndarray,
    y_test: np.ndarray,
    loss_cfg: UnifiedLossConfig = UnifiedLossConfig(),
) -> TrainOutputs:
    lstm = LSTMTrainer(input_dim=x_seq_train.shape[2])
    xgb = XGBModel()
    cat = CatModel()

    lstm.fit(x_seq_train, y_train)
    xgb.fit(x_tab_train, y_train)
    cat.fit(x_tab_train, y_train)

    p_lstm_train = lstm.predict(x_seq_train)
    p_xgb_train = xgb.predict(x_tab_train)
    p_cat_train = cat.predict(x_tab_train)

    p_lstm = lstm.predict(x_seq_test)
    p_xgb = xgb.predict(x_tab_test)
    p_cat = cat.predict(x_tab_test)

    base_train = np.column_stack([p_lstm_train, p_xgb_train, p_cat_train])
    base_test = np.column_stack([p_lstm, p_xgb, p_cat])

    hafm = _train_hafm(x_tab_train, base_train, y_train, loss_cfg)
    hafm.eval()
    with torch.no_grad():
        weights = hafm(torch.FloatTensor(x_tab_test))
        p_fusion = (weights * torch.FloatTensor(base_test)).sum(dim=1).numpy()

    preds = {"LSTM": p_lstm, "XGBoost": p_xgb, "CatBoost": p_cat, "HAFM": p_fusion}
    metrics = {k: regression_metrics(y_test, v) for k, v in preds.items()}
    return TrainOutputs(predictions=preds, metrics=metrics)
