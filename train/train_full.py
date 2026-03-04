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
from train.logger import ExperimentLogger


@dataclass
class TrainOutputs:
    predictions: Dict[str, np.ndarray]
    metrics: Dict[str, Dict[str, float]]
    weights: np.ndarray
    base_predictions: np.ndarray
    hidden_repr: np.ndarray
    loss_history: Dict[str, list[float]]


def _build_hidden_by_module(x_tab_train: np.ndarray, modules: np.ndarray | None) -> Dict[str, torch.Tensor]:
    if modules is None:
        return {}
    out: Dict[str, torch.Tensor] = {}
    for m in np.unique(modules):
        idx = np.where(modules == m)[0]
        if len(idx) > 0:
            out[str(m)] = torch.FloatTensor(x_tab_train[idx])
    return out


def _train_hafm(
    x_tab_train: np.ndarray,
    base_preds_train: np.ndarray,
    y_train: np.ndarray,
    loss_cfg: UnifiedLossConfig,
    modules_train: np.ndarray | None,
    logger: ExperimentLogger | None,
    epochs: int = 100,
    lr: float = 1e-3,
) -> HAFM:
    model = HAFM(input_dim=x_tab_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    unified = UnifiedLoss(loss_cfg)

    xt = torch.FloatTensor(x_tab_train)
    bt = torch.FloatTensor(base_preds_train)
    yt = torch.FloatTensor(y_train)
    hidden_by_module = _build_hidden_by_module(x_tab_train, modules_train)

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        weights = model(xt)
        y_hat = (weights * bt).sum(dim=1)

        l_reg = unified.regression_loss(y_hat, yt)
        l_transfer = unified.transfer_loss(hidden_by_module)
        l_div = unified.diversity_loss(bt)
        l_stab = unified.stability_loss(weights)
        loss = (
            l_reg
            + loss_cfg.lambda_transfer * l_transfer
            + loss_cfg.lambda_diversity * l_div
            + loss_cfg.lambda_stability * l_stab
        )
        loss.backward()
        opt.step()

        if logger is not None:
            logger.log_epoch(
                epoch,
                loss_total=loss.item(),
                loss_reg=l_reg.item(),
                loss_transfer=l_transfer.item(),
                loss_diversity=l_div.item(),
                loss_stability=l_stab.item(),
            )
    return model


def train_full_pipeline(
    x_seq_train: np.ndarray,
    x_tab_train: np.ndarray,
    y_train: np.ndarray,
    x_seq_test: np.ndarray,
    x_tab_test: np.ndarray,
    y_test: np.ndarray,
    loss_cfg: UnifiedLossConfig = UnifiedLossConfig(),
    modules_train: np.ndarray | None = None,
    logger: ExperimentLogger | None = None,
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

    hafm = _train_hafm(x_tab_train, base_train, y_train, loss_cfg, modules_train, logger)
    hafm.eval()
    with torch.no_grad():
        weights = hafm(torch.FloatTensor(x_tab_test))
        p_fusion = (weights * torch.FloatTensor(base_test)).sum(dim=1).numpy()

    preds = {"LSTM": p_lstm, "XGBoost": p_xgb, "CatBoost": p_cat, "HAFM": p_fusion}
    metrics = {k: regression_metrics(y_test, v) for k, v in preds.items()}
    hist = logger.to_frame().to_dict(orient="list") if logger is not None else {}
    return TrainOutputs(
        predictions=preds,
        metrics=metrics,
        weights=weights.numpy(),
        base_predictions=base_test,
        hidden_repr=x_tab_test,
        loss_history=hist,
    )
