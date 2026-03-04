from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from evaluation.metrics import regression_metrics
from loss.unified_loss import UnifiedLoss, UnifiedLossConfig
from models.cat import CatModel
from models.hafm import HAFM, fuse_predictions
from models.lstm import LSTMTrainer
from models.xgb import XGBModel


@dataclass
class TrainOutputs:
    predictions: Dict[str, np.ndarray]
    metrics: Dict[str, Dict[str, float]]


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

    p_lstm = lstm.predict(x_seq_test)
    p_xgb = xgb.predict(x_tab_test)
    p_cat = cat.predict(x_tab_test)

    base = np.column_stack([p_lstm, p_xgb, p_cat])
    hafm = HAFM(input_dim=x_tab_test.shape[1])
    fusion = fuse_predictions(torch.FloatTensor(x_tab_test), torch.FloatTensor(base), hafm)
    p_fusion = fusion.y_pred.detach().numpy()

    unified = UnifiedLoss(loss_cfg)
    _ = unified.total_loss(
        fusion.y_pred,
        torch.FloatTensor(y_test),
        {},
        torch.FloatTensor(base),
        fusion.weights,
    )

    preds = {"LSTM": p_lstm, "XGBoost": p_xgb, "CatBoost": p_cat, "HAFM": p_fusion}
    metrics = {k: regression_metrics(y_test, v) for k, v in preds.items()}
    return TrainOutputs(predictions=preds, metrics=metrics)
