import numpy as np
from sklearn.linear_model import LinearRegression

from src.evaluation.evaluator import Evaluator


def test_evaluator_regression_metrics():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([2, 4, 6, 8], dtype=float)
    model = LinearRegression().fit(X, y)
    metrics = Evaluator(task_type="regression").evaluate(model, X, y)
    assert metrics["mae"] < 1e-6
