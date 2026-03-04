"""Unified exports for both legacy pipeline and new modular MT-HAFNet code."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .baselines import BaselineSuite, fit_predict_baselines
from .cat import CatModel
from .hafm import HAFM, FusionOutput, fuse_predictions
from .lstm import LSTMConfig, LSTMRegressor
from .xgb import XGBModel

# defaults (overwritten when legacy module exists)
SequenceDataset = TabularDataset = LSTMModel = LSTMTrainer = None
XGBoostModel = CatBoostModel = None
DynamicWeightFusion = DynamicWeightNet = DynamicFusionTrainer = None
StackingFusion = MAMLBaseModel = MAMLTrainer = None

_legacy_path = Path(__file__).resolve().parent.parent / "models.py"
if _legacy_path.exists():
    spec = importlib.util.spec_from_file_location("_legacy_models", _legacy_path)
    if spec and spec.loader:
        _legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_legacy)

        SequenceDataset = _legacy.SequenceDataset
        TabularDataset = _legacy.TabularDataset
        LSTMModel = _legacy.LSTMModel
        LSTMTrainer = _legacy.LSTMTrainer
        XGBoostModel = _legacy.XGBoostModel
        CatBoostModel = _legacy.CatBoostModel
        DynamicWeightFusion = _legacy.DynamicWeightFusion
        DynamicWeightNet = _legacy.DynamicWeightFusion
        DynamicFusionTrainer = _legacy.DynamicFusionTrainer
        StackingFusion = _legacy.StackingFusion
        MAMLBaseModel = _legacy.MAMLBaseModel
        MAMLTrainer = _legacy.MAMLTrainer

__all__ = [
    "LSTMConfig",
    "LSTMRegressor",
    "XGBModel",
    "CatModel",
    "HAFM",
    "FusionOutput",
    "fuse_predictions",
    "BaselineSuite",
    "fit_predict_baselines",
    "SequenceDataset",
    "TabularDataset",
    "LSTMModel",
    "LSTMTrainer",
    "XGBoostModel",
    "CatBoostModel",
    "DynamicWeightFusion",
    "DynamicWeightNet",
    "DynamicFusionTrainer",
    "StackingFusion",
    "MAMLBaseModel",
    "MAMLTrainer",
]
