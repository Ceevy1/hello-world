"""Phase 2: feature alignment and domain adaptation metrics."""

from __future__ import annotations

import joblib
import pandas as pd

from src.data.feature_aligner import FeatureAligner


def run_phase2(config: dict, logger) -> None:
    df = joblib.load("data/processed/target_features.pkl")
    aligner = FeatureAligner()
    aligned = aligner.align(df)
    joblib.dump(aligned, "data/aligned/aligned_features.pkl")
    pd.DataFrame(aligned).to_pickle("data/aligned/aligned_data.pkl")
    logger.info("Phase2 complete: aligned_features.pkl generated")
