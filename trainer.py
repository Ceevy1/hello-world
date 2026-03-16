from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

from loss.advanced_binary_loss import loss_fn
from model import DynamicFusionEnhanced


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 15
    lambda1: float = 0.5
    lambda2: float = 0.1
    lambda3: float = 0.0
    grad_clip: float = 5.0


def _stack_batch(batch):
    x_seq = torch.stack([b.x_seq for b in batch])
    x_stat = torch.stack([b.x_stat for b in batch])
    node_idx = torch.stack([b.node_index for b in batch])
    week = torch.stack([b.week_index for b in batch])
    y = torch.stack([b.y for b in batch])
    return x_seq, x_stat, node_idx, week, y


def evaluate(model: DynamicFusionEnhanced, loader: DataLoader, node_features: torch.Tensor, edge_index: torch.Tensor, device: torch.device) -> dict[str, float]:
    model.eval()
    ys, ps, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            x_seq, x_stat, node_idx, week, y = _stack_batch(batch)
            pred, _, _ = model(x_seq.to(device), x_stat.to(device), node_idx.to(device), week.to(device), node_features.to(device), edge_index.to(device))
            y_dev = y.to(device)
            losses.append(float(F.binary_cross_entropy(pred, y_dev).detach().cpu()))
            ys.append(y.numpy())
            ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "AUC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "Accuracy": float(accuracy_score(y_true, y_hat)),
        "Precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "Recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
    }


def train_epoch(model: DynamicFusionEnhanced, loader: DataLoader, optimizer: Adam, node_features: torch.Tensor, edge_index: torch.Tensor, cfg: TrainConfig, device: torch.device) -> dict[str, float]:
    model.train()
    loss_values = []
    for batch in loader:
        x_seq, x_stat, node_idx, week, y = _stack_batch(batch)
        x_seq = x_seq.to(device)
        x_stat = x_stat.to(device)
        node_idx = node_idx.to(device)
        week = week.to(device)
        y = y.to(device)

        pred, w, _ = model(x_seq, x_stat, node_idx, week, node_features.to(device), edge_index.to(device))
        pred_early = pred
        pred_full = pred.detach()
        loss, _ = loss_fn(pred, y, w, pred_early, pred_full, lambda1=cfg.lambda1, lambda2=cfg.lambda2, lambda3=cfg.lambda3)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        loss_values.append(float(loss.detach().cpu()))
    return {"train_loss": float(np.mean(loss_values))}


def fit(model: DynamicFusionEnhanced, train_loader: DataLoader, val_loader: DataLoader, graph: tuple[torch.Tensor, torch.Tensor], cfg: TrainConfig, output_dir: str = "outputs") -> dict[str, float]:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    node_features, edge_index = graph

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    hist = []
    best_auc = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, node_features, edge_index, cfg, device)
        va = evaluate(model, val_loader, node_features, edge_index, device)
        row = {
            "epoch": epoch,
            "train_loss": tr["train_loss"],
            "val_loss": va["loss"],
            "val_AUC": va["AUC"],
            "val_Accuracy": va["Accuracy"],
            "val_Precision": va["Precision"],
            "val_Recall": va["Recall"],
            "val_F1": va["F1"],
        }
        hist.append(row)
        if np.nan_to_num(va["AUC"], nan=0.0) > best_auc:
            best_auc = np.nan_to_num(va["AUC"], nan=0.0)
            best_state = model.state_dict()

    pd.DataFrame(hist).to_csv(out / "loss_curve.csv", index=False)
    if best_state is not None:
        torch.save(best_state, out / "model_best.pth")
        model.load_state_dict(best_state)
    return {"best_val_auc": float(best_auc)}


def export_predictions(
    model: DynamicFusionEnhanced,
    test_loader: DataLoader,
    graph: tuple[torch.Tensor, torch.Tensor],
    output_dir: str = "outputs",
    student_ids: np.ndarray | None = None,
) -> dict[str, float]:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    node_features, edge_index = graph

    rows = []
    y_true, y_prob = [], []
    weights_track = []
    repr_track = []
    with torch.no_grad():
        sid = 0
        for batch in test_loader:
            x_seq, x_stat, node_idx, week, y = _stack_batch(batch)
            x_seq_d, x_stat_d, node_idx_d, week_d = x_seq.to(device), x_stat.to(device), node_idx.to(device), week.to(device)
            pred, weights, _ = model(x_seq_d, x_stat_d, node_idx_d, week_d, node_features.to(device), edge_index.to(device))
            seq_repr = model.seq(x_seq_d).cpu().numpy()
            p = pred.cpu().numpy()
            w = weights.cpu().numpy()
            yt = y.numpy()
            for i in range(len(p)):
                student_id = int(student_ids[sid]) if student_ids is not None and sid < len(student_ids) else sid
                rows.append({"student_id": student_id, "y_true": int(yt[i]), "y_pred_prob": float(p[i]), "y_pred_label": int(p[i] >= 0.5)})
                sid += 1
            y_true.append(yt)
            y_prob.append(p)
            weights_track.append(w)
            repr_track.append(seq_repr)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out / "predictions.csv", index=False)

    yt = np.concatenate(y_true)
    yp = np.concatenate(y_prob)
    fpr, tpr, _ = roc_curve(yt, yp)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(out / "roc_curve_data.csv", index=False)

    cm = confusion_matrix(yt, (yp >= 0.5).astype(int))
    np.savez(out / "confusion_matrix.npz", confusion_matrix=cm)

    w_arr = np.concatenate(weights_track, axis=0)
    pd.DataFrame({"epoch": np.arange(1, len(w_arr) + 1), "w_seq": w_arr[:, 0], "w_kg": w_arr[:, 1], "w_stat": w_arr[:, 2]}).to_csv(out / "weights_trajectory.csv", index=False)

    repr_arr = np.concatenate(repr_track, axis=0)
    dist = np.linalg.norm(repr_arr[:, None, :] - repr_arr[None, :, :], axis=-1)
    np.save(out / "representation_distances.npy", dist)

    return {
        "AUC": float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan"),
        "Accuracy": float(accuracy_score(yt, (yp >= 0.5).astype(int))),
        "Precision": float(precision_score(yt, (yp >= 0.5).astype(int), zero_division=0)),
        "Recall": float(recall_score(yt, (yp >= 0.5).astype(int), zero_division=0)),
        "F1": float(f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)),
    }
