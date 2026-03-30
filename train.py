from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import StudentDataset
from evaluate import classification_metrics, regression_metrics
from importlib.util import module_from_spec, spec_from_file_location

_model_spec = spec_from_file_location("self_dynamic_fusion", "model/dynamic_fusion.py")
_model_mod = module_from_spec(_model_spec)
assert _model_spec and _model_spec.loader
_model_spec.loader.exec_module(_model_mod)
DynamicFusionModel = _model_mod.DynamicFusionModel
from preprocessing import build_classification_labels, preprocess_scores


@dataclass
class TrainConfig:
    csv_path: str = "data/student_scores.csv"
    task: Literal["regression", "classification"] = "regression"
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 200
    patience: int = 20
    seed: int = 42


def train_and_evaluate(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    prepared = preprocess_scores(cfg.csv_path)
    train_df, val_df, test_df = prepared.train_df, prepared.val_df, prepared.test_df

    label_col = prepared.target_col
    num_classes = 4
    if cfg.task == "classification":
        train_df, num_classes = build_classification_labels(train_df, prepared.target_col)
        val_df, _ = build_classification_labels(val_df, prepared.target_col)
        test_df, _ = build_classification_labels(test_df, prepared.target_col)
        label_col = "label_cls"

    train_ds = StudentDataset(train_df, prepared.exercise_cols, prepared.lab_cols, prepared.static_cols, label_col)
    val_ds = StudentDataset(val_df, prepared.exercise_cols, prepared.lab_cols, prepared.static_cols, label_col)
    test_ds = StudentDataset(test_df, prepared.exercise_cols, prepared.lab_cols, prepared.static_cols, label_col)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = DynamicFusionModel(
        static_dim=len(prepared.static_cols),
        hidden_dim=64,
        task=cfg.task,
        num_classes=num_classes,
    )
    criterion = nn.MSELoss() if cfg.task == "regression" else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_state, best_val_loss, wait = None, float("inf"), 0

    for _ in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            pred = model(batch["exercise"], batch["lab"], batch["static"])
            target = batch["label"] if cfg.task == "regression" else batch["label"].long()
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = _eval_loss(model, val_loader, criterion, cfg.task)
        if val_loss < best_val_loss:
            best_val_loss, wait = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = _predict(model, test_loader, cfg.task)
    if cfg.task == "regression":
        metrics = regression_metrics(y_true, y_pred)
    else:
        metrics = classification_metrics(y_true.astype(int), y_pred)
    return model, metrics


def _eval_loss(model, loader, criterion, task: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["exercise"], batch["lab"], batch["static"])
            target = batch["label"] if task == "regression" else batch["label"].long()
            loss = criterion(pred, target)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def _predict(model, loader, task: str):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["exercise"], batch["lab"], batch["static"])
            ys.append(batch["label"].numpy())
            ps.append(pred.numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return y_true, y_pred


if __name__ == "__main__":
    model, metrics = train_and_evaluate(TrainConfig(task="regression"))
    print("Regression metrics:", metrics)

    _, cls_metrics = train_and_evaluate(TrainConfig(task="classification"))
    print("Classification metrics:", cls_metrics)
