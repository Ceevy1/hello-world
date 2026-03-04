"""Evaluation package exports + compatibility bridge for legacy `evaluation.py`."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .metrics import regression_metrics
from .statistics import significance_tests

_legacy_path = Path(__file__).resolve().parent.parent / "evaluation.py"
if _legacy_path.exists():
    spec = importlib.util.spec_from_file_location("_legacy_evaluation", _legacy_path)
    if spec and spec.loader:
        _legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_legacy)

        compute_regression_metrics = _legacy.compute_regression_metrics
        compute_classification_metrics = _legacy.compute_classification_metrics
        SignificanceTester = _legacy.SignificanceTester
        SHAPAnalyzer = _legacy.SHAPAnalyzer
        ResultsReporter = _legacy.ResultsReporter

__all__ = [
    "regression_metrics",
    "significance_tests",
    "compute_regression_metrics",
    "compute_classification_metrics",
    "SignificanceTester",
    "SHAPAnalyzer",
    "ResultsReporter",
]
