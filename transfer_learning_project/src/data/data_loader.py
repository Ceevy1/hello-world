"""Data loading and schema validation utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load source/target datasets and provide validation/reporting helpers."""

    SUPPORTED_SUFFIXES = {".xlsx", ".csv", ".json"}

    def __init__(self, column_mapping: dict[str, str] | None = None) -> None:
        self.column_mapping = column_mapping or {}

    def load_custom_dataset(self, file_path: str, anonymize: bool = False) -> pd.DataFrame:
        """Load local custom dataset.

        Args:
            file_path: Path to .xlsx/.csv/.json local file.
            anonymize: Whether to hash `student_id` column when present.

        Returns:
            Loaded and standardized DataFrame.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        if path.suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

        if path.suffix == ".xlsx":
            df = pd.read_excel(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path, encoding="utf-8-sig")
        else:
            df = pd.read_json(path)

        if self.column_mapping:
            reverse_mapping = {v: k for k, v in self.column_mapping.items()}
            df = df.rename(columns=reverse_mapping)

        if anonymize and "student_id" in df.columns:
            df["student_id"] = df["student_id"].astype(str).map(
                lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:16]
            )
        return df

    def load_oulad_features(self, oulad_path: str) -> pd.DataFrame:
        """Load preprocessed OULAD features from pickle/joblib/csv."""
        path = Path(oulad_path)
        if path.suffix in {".pkl", ".joblib"}:
            loaded = joblib.load(path)
            return loaded if isinstance(loaded, pd.DataFrame) else pd.DataFrame(loaded)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError("OULAD特征文件需为 .pkl/.joblib/.csv")

    def validate_schema(self, df: pd.DataFrame, schema_config: dict[str, Any]) -> None:
        """Validate required columns and optional dtypes."""
        required = set(schema_config.get("required_columns", []))
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"缺失必要列: {sorted(missing)}")

        dtypes = schema_config.get("dtypes", {})
        for col, expected in dtypes.items():
            if col in df.columns and str(df[col].dtype) != expected:
                raise TypeError(f"列 {col} 类型错误, 期望={expected}, 实际={df[col].dtype}")

    def generate_data_report(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate missing-value/distribution/outlier summary."""
        numeric = df.select_dtypes(include=[np.number])
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        outlier_mask = (numeric < (q1 - 1.5 * iqr)) | (numeric > (q3 + 1.5 * iqr))

        report = {
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "missing_rate": (df.isna().mean().round(4)).to_dict(),
            "describe": numeric.describe().round(4).to_dict(),
            "outlier_rate": outlier_mask.mean().fillna(0).round(4).to_dict(),
        }
        return report

    def split_dataset(
        self,
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
        stratify_col: str | None,
        seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/val/test."""
        stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
        train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)

        stratify_train = (
            train_val[stratify_col] if stratify_col and stratify_col in train_val.columns else None
        )
        relative_val_size = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=relative_val_size,
            random_state=seed,
            stratify=stratify_train,
        )
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_data_report(report: dict[str, Any], output_path: str) -> None:
    """Persist report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
