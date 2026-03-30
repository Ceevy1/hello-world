from __future__ import annotations

from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


class StudentDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        exercise_cols: list[str],
        lab_cols: list[str],
        static_cols: list[str],
        label_col: str,
    ):
        self.exercise = data[exercise_cols].to_numpy(dtype="float32")
        self.lab = data[lab_cols].to_numpy(dtype="float32")
        self.static = data[static_cols].to_numpy(dtype="float32")
        self.label = data[label_col].to_numpy(dtype="float32")

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "exercise": torch.tensor(self.exercise[idx]),
            "lab": torch.tensor(self.lab[idx]),
            "static": torch.tensor(self.static[idx]),
            "label": torch.tensor(self.label[idx]),
        }
