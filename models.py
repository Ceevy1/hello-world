"""
Models Module
Implements: LSTM, XGBoost, CatBoost, Dynamic Fusion, MAML
"""

import os
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


# ============================================================
# PyTorch Dataset
# ============================================================
class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray = None):
        self.sequences = torch.FloatTensor(sequences)
        self.y_reg = torch.FloatTensor(y_reg)
        self.y_cls = torch.LongTensor(y_cls) if y_cls is not None else None

    def __len__(self):
        return len(self.y_reg)

    def __getitem__(self, idx):
        if self.y_cls is not None:
            return self.sequences[idx], self.y_reg[idx], self.y_cls[idx]
        return self.sequences[idx], self.y_reg[idx]


class TabularDataset(Dataset):
    def __init__(self, tabular: np.ndarray, y_reg: np.ndarray):
        self.tabular = torch.FloatTensor(tabular)
        self.y_reg = torch.FloatTensor(y_reg)

    def __len__(self):
        return len(self.y_reg)

    def __getitem__(self, idx):
        return self.tabular[idx], self.y_reg[idx]


# ============================================================
# LSTM Model
# ============================================================
class LSTMModel(nn.Module):
    """
    2-layer LSTM for sequential behavioral data.
    Input:  (batch, T, D)
    Output: (batch, 1) regression
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # Last layer hidden state
        h_last = self.dropout(h_last)
        return self.fc(h_last).squeeze(-1)


class LSTMTrainer:
    """Trainer wrapper for LSTM model."""

    def __init__(self, config: Dict, input_dim: int, seed: int = 42, device: str = None):
        self.config = config
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.history = {"train_loss": [], "val_loss": []}

    def fit(
        self,
        X_seq: np.ndarray,
        y_reg: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        train_ds = SequenceDataset(X_seq, y_reg)
        train_loader = DataLoader(
            train_ds, batch_size=self.config["batch_size"], shuffle=True, num_workers=0
        )

        val_loader = None
        if X_val is not None:
            val_ds = SequenceDataset(X_val, y_val)
            val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(self.config["epochs"]):
            # Training
            self.model.train()
            train_losses = []
            for batch in train_loader:
                seq_b, y_b = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(seq_b)
                loss = self.criterion(pred, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            self.history["train_loss"].append(avg_train)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for seq_b, y_b in val_loader:
                        seq_b, y_b = seq_b.to(self.device), y_b.to(self.device)
                        pred = self.model(seq_b)
                        val_losses.append(self.criterion(pred, y_b).item())
                avg_val = np.mean(val_losses)
                self.history["val_loss"].append(avg_val)
                self.scheduler.step(avg_val)

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config["patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(f"[LSTM] Epoch {epoch+1}: train={avg_train:.4f}, val={avg_val:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"[LSTM] Epoch {epoch+1}: train={avg_train:.4f}")

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        ds = SequenceDataset(X_seq, np.zeros(len(X_seq)))
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for seq_b, _ in loader:
                seq_b = seq_b.to(self.device)
                preds.append(self.model(seq_b).cpu().numpy())
        return np.concatenate(preds)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


# ============================================================
# XGBoost Wrapper
# ============================================================
class XGBoostModel:
    """XGBoost regression wrapper."""

    def __init__(self, config: Dict):
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**config)
        except ImportError:
            logger.warning("XGBoost not installed. Using GradientBoostingRegressor.")
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=config.get("n_estimators", 300),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", 0.05),
                random_state=config.get("random_state", 42),
            )
        self.config = config

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        try:
            if eval_set:
                self.model.fit(X, y, eval_set=eval_set, verbose=False)
            else:
                self.model.fit(X, y)
        except TypeError:
            self.model.fit(X, y)
        logger.info("[XGBoost] Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        try:
            return self.model.feature_importances_
        except AttributeError:
            return None

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str):
        import joblib
        self.model = joblib.load(path)


# ============================================================
# CatBoost Wrapper
# ============================================================
class CatBoostModel:
    """CatBoost regression wrapper."""

    def __init__(self, config: Dict):
        try:
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(**config)
            self.use_catboost = True
        except ImportError:
            logger.warning("CatBoost not installed. Using RandomForest as fallback.")
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=config.get("iterations", 300),
                max_depth=config.get("depth", 6),
                random_state=config.get("random_seed", 42),
                n_jobs=-1,
            )
            self.use_catboost = False
        self.config = config

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        if self.use_catboost and X_val is not None:
            self.model.fit(X, y, eval_set=(X_val, y_val))
        else:
            self.model.fit(X, y)
        logger.info("[CatBoost] Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        try:
            return self.model.feature_importances_
        except AttributeError:
            return None

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str):
        import joblib
        self.model = joblib.load(path)


# ============================================================
# Dynamic Fusion Module
# ============================================================
class DynamicWeightFusion(nn.Module):
    """
    Dynamic weight fusion network.
    Generates sample-wise weights for base model predictions.

    Input:  tabular_features (N, F)
    Output: weighted prediction (N, 1)

    Architecture:
        FC(F -> 32) -> ReLU -> FC(32 -> 3) -> Softmax
        y = w1*y_lstm + w2*y_xgb + w3*y_cat
    """

    def __init__(self, tabular_dim: int, n_models: int = 3):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_models),
            nn.Softmax(dim=-1),
        )
        self.n_models = n_models

    def forward(
        self,
        tabular: torch.Tensor,
        predictions: torch.Tensor,  # (N, n_models)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.weight_net(tabular)                   # (N, n_models)
        fused = (weights * predictions).sum(dim=-1)         # (N,)
        return fused, weights


class DynamicFusionTrainer:
    """Trainer for dynamic weight fusion."""

    def __init__(self, tabular_dim: int, config: Dict, seed: int = 42):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)

        self.model = DynamicWeightFusion(tabular_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.criterion = nn.MSELoss()

    def fit(
        self,
        X_tab: np.ndarray,
        base_preds: np.ndarray,  # (N, 3)
        y: np.ndarray,
    ):
        X_tab_t = torch.FloatTensor(X_tab).to(self.device)
        base_preds_t = torch.FloatTensor(base_preds).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tab_t, base_preds_t, y_t)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in range(self.config["epochs"]):
            self.model.train()
            losses = []
            for xb, pb, yb in loader:
                self.optimizer.zero_grad()
                pred, _ = self.model(xb, pb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                logger.info(f"[DynamicFusion] Epoch {epoch+1}: loss={np.mean(losses):.4f}")

    def predict(
        self, X_tab: np.ndarray, base_preds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            xt = torch.FloatTensor(X_tab).to(self.device)
            pt = torch.FloatTensor(base_preds).to(self.device)
            fused, weights = self.model(xt, pt)
        return fused.cpu().numpy(), weights.cpu().numpy()


# ============================================================
# Stacking Fusion
# ============================================================
class StackingFusion:
    """Stacking ensemble with linear or MLP meta-learner."""

    def __init__(self, meta_model_type: str = "linear"):
        self.meta_model_type = meta_model_type
        if meta_model_type == "linear":
            from sklearn.linear_model import Ridge
            self.meta_model = Ridge(alpha=1.0)
        else:
            from sklearn.neural_network import MLPRegressor
            self.meta_model = MLPRegressor(
                hidden_layer_sizes=(32,), max_iter=500, random_state=42
            )

    def fit(self, base_preds: np.ndarray, y: np.ndarray):
        """base_preds: (N, 3)"""
        self.meta_model.fit(base_preds, y)

    def predict(self, base_preds: np.ndarray) -> np.ndarray:
        return self.meta_model.predict(base_preds)


# ============================================================
# MAML Meta-Learning
# ============================================================
class MAMLBaseModel(nn.Module):
    """
    Simple MLP for MAML (tabular input).
    Input:  (N, F)
    Output: (N, 1)
    """

    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning (MAML) trainer.
    Each module is a task.
    - Inner loop: task-specific gradient update
    - Outer loop: meta-gradient update
    """

    def __init__(self, input_dim: int, config: Dict, seed: int = 42):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)

        self.meta_model = MAMLBaseModel(input_dim, config["hidden_size"]).to(self.device)
        self.meta_optimizer = optim.Adam(
            self.meta_model.parameters(), lr=config["outer_lr"]
        )
        self.criterion = nn.MSELoss()

    def _inner_loop(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        """Perform inner loop gradient updates on support set."""
        task_model = copy.deepcopy(model)
        inner_opt = optim.SGD(task_model.parameters(), lr=self.config["inner_lr"])

        task_model.train()
        for _ in range(self.config["inner_steps"]):
            inner_opt.zero_grad()
            pred = task_model(support_x)
            loss = self.criterion(pred, support_y)
            loss.backward()
            inner_opt.step()

        return task_model

    def meta_train(self, tasks: List[Dict]):
        """
        tasks: list of task dicts with keys 'tabular', 'y_reg'
        """
        logger.info(f"Starting MAML training for {self.config['meta_epochs']} epochs...")

        for epoch in range(self.config["meta_epochs"]):
            meta_loss = 0.0
            self.meta_optimizer.zero_grad()

            # Accumulate gradients across tasks
            outer_losses = []
            for task in tasks:
                X = torch.FloatTensor(task["tabular"]).to(self.device)
                y = torch.FloatTensor(task["y_reg"]).to(self.device)
                n = len(y)

                # Random support/query split
                idx = torch.randperm(n)
                s_size = min(self.config["support_size"], n // 2)
                q_size = min(self.config["query_size"], n - s_size)
                sup_idx = idx[:s_size]
                qry_idx = idx[s_size: s_size + q_size]

                sup_x, sup_y = X[sup_idx], y[sup_idx]
                qry_x, qry_y = X[qry_idx], y[qry_idx]

                # Inner loop
                adapted_model = self._inner_loop(self.meta_model, sup_x, sup_y)

                # Outer loss on query set
                adapted_model.eval()
                with torch.no_grad():
                    qry_pred = adapted_model(qry_x)
                qry_loss = self.criterion(qry_pred, qry_y)
                outer_losses.append(qry_loss)
                meta_loss += qry_loss.item()

            # Meta-update using first-order approximation
            total_outer = sum(outer_losses) / len(outer_losses)
            # Re-compute with gradients for meta-update
            self.meta_optimizer.zero_grad()
            grad_loss = torch.tensor(0.0, requires_grad=True)
            for task in tasks:
                X = torch.FloatTensor(task["tabular"]).to(self.device)
                y = torch.FloatTensor(task["y_reg"]).to(self.device)
                pred = self.meta_model(X)
                task_loss = self.criterion(pred, y)
                grad_loss = grad_loss + task_loss

            (grad_loss / len(tasks)).backward()
            self.meta_optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.info(f"[MAML] Epoch {epoch+1}: meta_loss={meta_loss/len(tasks):.4f}")

    def fine_tune(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        n_steps: int = 10,
    ) -> nn.Module:
        """Fine-tune meta-model on new task support set."""
        X = torch.FloatTensor(support_x).to(self.device)
        y = torch.FloatTensor(support_y).to(self.device)
        adapted = self._inner_loop(self.meta_model, X, y)
        for _ in range(n_steps - self.config["inner_steps"]):
            opt = optim.SGD(adapted.parameters(), lr=self.config["inner_lr"])
            opt.zero_grad()
            loss = self.criterion(adapted(X), y)
            loss.backward()
            opt.step()
        return adapted

    def predict(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            xt = torch.FloatTensor(X).to(self.device)
            pred = model(xt).cpu().numpy()
        return pred

    def save(self, path: str):
        torch.save(self.meta_model.state_dict(), path)

    def load(self, path: str):
        self.meta_model.load_state_dict(torch.load(path, map_location=self.device))
