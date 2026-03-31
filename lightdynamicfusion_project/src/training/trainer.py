import numpy as np
import pandas as pd

from src.training.cv_handler import SmallSampleCVHandler


class ExperimentTrainer:
    def __init__(self, random_state: int = 42):
        self.cv_handler = SmallSampleCVHandler(random_state=random_state)

    def evaluate_with_cv(self, model, df: pd.DataFrame, target_col: str):
        y = df[target_col].to_numpy(dtype=float)
        X = df.drop(columns=[target_col])
        return self.cv_handler.run_cv(model, X, y, task='regression')
