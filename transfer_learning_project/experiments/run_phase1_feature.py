"""Phase 1: feature engineering and mapping."""

from __future__ import annotations

import joblib

from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer


def run_phase1(config: dict, logger) -> None:
    loader = DataLoader(config["custom_dataset"]["columns"])
    fe = FeatureEngineer()
    df = loader.load_custom_dataset(config["data"]["target"]["path"], anonymize=True)
    df = fe.handle_missing(df, config["feature_engineering"]["missing_strategy"])
    df = fe.remove_outliers(df, config["feature_engineering"]["outlier_method"])
    if config["feature_engineering"]["construct_derived"]:
        df = fe.construct_derived_features(df)
    joblib.dump(df, "data/processed/target_features.pkl")
    logger.info("Phase1 complete: target_features.pkl generated")
