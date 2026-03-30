"""Pretrained model loader for multiple formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch


class PretrainedModelLoader:
    """Load pretrained model and expose embedding-friendly interfaces."""

    SUPPORTED_FORMATS = [".pkl", ".joblib", ".pt", ".pth", ".json", ".ubj"]

    def __init__(self, model_path: str, model_type: str = "auto"):
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}

    def load(self) -> "PretrainedModelLoader":
        suffix = self.model_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported model format: {suffix}")

        if suffix in [".pkl", ".joblib"]:
            self.model = joblib.load(self.model_path)
            self.model_type = "sklearn"
        elif suffix in [".pt", ".pth"]:
            self.model = torch.load(self.model_path, map_location="cpu")
            self.model_type = "pytorch"
        elif suffix in [".json", ".ubj"]:
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(str(self.model_path))
            self.model = booster
            self.model_type = "xgboost"

        self.metadata = {
            "model_path": str(self.model_path),
            "model_type": self.model_type,
        }
        return self

    def extract_feature_extractor(self) -> Any:
        if self.model_type == "pytorch" and hasattr(self.model, "features"):
            return self.model.features
        return self.model

    def get_embedding(self, X: np.ndarray) -> np.ndarray:
        if self.model_type in {"sklearn", "xgboost"}:
            return np.asarray(X)
        if self.model_type == "pytorch":
            self.model.eval()
            with torch.no_grad():
                x = torch.as_tensor(X, dtype=torch.float32)
                out = self.model(x)
                return out.detach().cpu().numpy()
        raise RuntimeError("Model not loaded")
