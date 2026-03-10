from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam

from models.catboost_model import CatBoostClassifierModel
from models.dynamic_fusion_enhanced import DynamicFusionEnhanced
from models.transformer_encoder import StudentTransformerEncoder
from models.xgboost_model import XGBoostClassifierModel
from training.loss import dynamic_fusion_loss


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    lambda1: float = 0.1
    lambda2: float = 0.2


class DynamicFusionTrainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = StudentTransformerEncoder().to(self.device)
        self.deep_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid()).to(self.device)
        self.fusion = DynamicFusionEnhanced().to(self.device)
        self.xgb = XGBoostClassifierModel()
        self.cat = CatBoostClassifierModel()

    def fit(self, seq_features: np.ndarray, tab_features: np.ndarray, y: np.ndarray, week_idx: np.ndarray) -> None:
        self.xgb.fit(tab_features, y)
        self.cat.fit(tab_features, y)

        x_tr, x_va, tab_tr, tab_va, y_tr, y_va, w_tr, w_va = train_test_split(
            seq_features, tab_features, y, week_idx, test_size=0.2, random_state=42, stratify=y
        )

        opt = Adam(list(self.encoder.parameters()) + list(self.deep_head.parameters()) + list(self.fusion.parameters()), lr=self.cfg.learning_rate)
        y_tr_t = torch.FloatTensor(y_tr).to(self.device)
        for _ in range(self.cfg.epochs):
            self.encoder.train()
            emb = self.encoder(torch.FloatTensor(x_tr).to(self.device))
            deep_pred = self.deep_head(emb).squeeze(-1)

            xgb_p = self.xgb.predict_proba(tab_tr)
            cat_p = self.cat.predict_proba(tab_tr)
            tree_pred = torch.FloatTensor(np.column_stack([xgb_p, cat_p])).to(self.device)

            y_hat, weights = self.fusion(emb, tree_pred, torch.LongTensor(w_tr).to(self.device), deep_pred)
            loss = dynamic_fusion_loss(y_hat, y_tr_t, weights, y_hat, y_hat.detach(), self.cfg.lambda1, self.cfg.lambda2)

            opt.zero_grad()
            loss.backward()
            opt.step()

        self._val_cache = (x_va, tab_va, y_va, w_va)

    def predict_proba(self, seq_features: np.ndarray, tab_features: np.ndarray, week_idx: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        with torch.no_grad():
            emb = self.encoder(torch.FloatTensor(seq_features).to(self.device))
            deep_pred = self.deep_head(emb).squeeze(-1)
            xgb_p = self.xgb.predict_proba(tab_features)
            cat_p = self.cat.predict_proba(tab_features)
            tree_pred = torch.FloatTensor(np.column_stack([xgb_p, cat_p])).to(self.device)
            y_hat, _ = self.fusion(emb, tree_pred, torch.LongTensor(week_idx).to(self.device), deep_pred)
            return y_hat.cpu().numpy()
