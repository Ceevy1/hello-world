from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class JunyiModelConfig:
    n_exercises: int
    emb_dim: int = 64
    hidden_dim: int = 96
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 128
    lambda_reg: float = 1e-3


class JunyiDynamicModel(nn.Module):
    """Dynamic weighting model for Junyi logs with optional graph fusion."""

    def __init__(self, cfg: JunyiModelConfig, adj_matrix: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.exercise_emb = nn.Embedding(cfg.n_exercises, cfg.emb_dim, padding_idx=0)
        self.response_emb = nn.Embedding(2, cfg.emb_dim)
        self.cont_proj = nn.Linear(2, cfg.emb_dim)

        self.encoder = nn.LSTM(cfg.emb_dim * 2, cfg.hidden_dim, batch_first=True)
        self.weight_gen = nn.Linear(cfg.hidden_dim, 1)

        self.use_graph = adj_matrix is not None
        if self.use_graph:
            A = torch.FloatTensor(adj_matrix)
            self.register_buffer("adj", A)
            self.graph_proj = nn.Linear(cfg.emb_dim, cfg.hidden_dim)
            self.gate = nn.Sequential(nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, 1))

        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, 1))

    def forward(
        self,
        exercise_ids: torch.Tensor,
        response_ids: torch.Tensor,
        continuous: torch.Tensor,
        mask: torch.Tensor,
        target_exercise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        e = self.exercise_emb(exercise_ids)
        r = self.response_emb(response_ids)
        c = self.cont_proj(continuous)
        x = torch.cat([e + r, c], dim=-1)

        h, _ = self.encoder(x)

        attn_logits = self.weight_gen(h).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(attn_logits, dim=1)
        context = torch.sum(h * attn.unsqueeze(-1), dim=1)

        gate_val = None
        if self.use_graph and target_exercise is not None:
            ex_base = self.exercise_emb.weight
            ex_graph = self.adj @ ex_base
            g_vec = self.graph_proj(ex_graph[target_exercise])
            gate_val = torch.sigmoid(self.gate(torch.cat([context, g_vec], dim=-1)))
            context = gate_val * context + (1.0 - gate_val) * g_vec

        logits = self.head(context).squeeze(-1)
        return logits, attn, gate_val

    @staticmethod
    def loss_fn(logits: torch.Tensor, target: torch.Tensor, attn: torch.Tensor, lambda_reg: float) -> torch.Tensor:
        ce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        reg = (attn ** 2).mean()
        return ce + lambda_reg * reg


class JunyiTrainer:
    def __init__(self, model: JunyiDynamicModel, cfg: JunyiModelConfig, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, data: Dict[str, np.ndarray]) -> None:
        ds = TensorDataset(
            torch.LongTensor(data["exercise_ids"]),
            torch.LongTensor(data["response_ids"]),
            torch.FloatTensor(data["continuous"]),
            torch.FloatTensor(data["mask"]),
            torch.FloatTensor(data["target"]),
            torch.LongTensor(data["target_exercise"]),
        )
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.model.train()
        for _ in range(self.cfg.epochs):
            for ex, resp, cont, m, y, tgt_ex in dl:
                ex, resp, cont, m, y, tgt_ex = ex.to(self.device), resp.to(self.device), cont.to(self.device), m.to(self.device), y.to(self.device), tgt_ex.to(self.device)
                opt.zero_grad()
                logits, attn, _ = self.model(ex, resp, cont, m, tgt_ex)
                loss = self.model.loss_fn(logits, y, attn, self.cfg.lambda_reg)
                loss.backward()
                opt.step()

    def predict(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            ex = torch.LongTensor(data["exercise_ids"]).to(self.device)
            resp = torch.LongTensor(data["response_ids"]).to(self.device)
            cont = torch.FloatTensor(data["continuous"]).to(self.device)
            m = torch.FloatTensor(data["mask"]).to(self.device)
            tgt_ex = torch.LongTensor(data["target_exercise"]).to(self.device)
            logits, attn, _ = self.model(ex, resp, cont, m, tgt_ex)
            prob = torch.sigmoid(logits).cpu().numpy()
            return prob, attn.cpu().numpy()
