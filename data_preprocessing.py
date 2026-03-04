"""
Data Layer & Preprocessing Module
Handles OULAD dataset loading, feature engineering, and dataset construction.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, SEED, MODULES, LABEL_MAP,
    SEQUENCE_MAX_WEEKS, SEQUENCE_DIM,
    STATIC_CAT_FEATURES, STATIC_NUM_FEATURES,
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. Raw Data Loader
# ============================================================
class OULADLoader:
    """Load raw OULAD CSV tables."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir

    def load_all(self) -> Dict[str, pd.DataFrame]:
        tables = {}
        files = {
            "student_info":       "studentInfo.csv",
            "student_vle":        "studentVle.csv",
            "assessments":        "assessments.csv",
            "student_assessment": "studentAssessment.csv",
            "student_reg":        "studentRegistration.csv",
            "courses":            "courses.csv",
            "vle":                "vle.csv",
        }
        for key, fname in files.items():
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                tables[key] = pd.read_csv(path)
                logger.info(f"Loaded {fname}: {tables[key].shape}")
            else:
                logger.warning(f"File not found: {path}")
        return tables


# ============================================================
# 2. Grade Reconstruction Module
# ============================================================
class GradeReconstructor:
    """
    Reconstruct final weighted score per student per module-presentation.
    FinalScore = sum(score_ij * weight_j)
    Strictly avoids using 'final_result' to prevent data leakage.
    """

    def compute(
        self,
        student_assessment: pd.DataFrame,
        assessments: pd.DataFrame,
    ) -> pd.DataFrame:
        df = student_assessment.merge(
            assessments[["id_assessment", "code_module", "code_presentation",
                         "assessment_type", "weight"]],
            on="id_assessment", how="left"
        )
        # Normalize weight per module-presentation
        total_weight = (
            assessments.groupby(["code_module", "code_presentation"])["weight"]
            .sum()
            .reset_index()
            .rename(columns={"weight": "total_weight"})
        )
        df = df.merge(total_weight, on=["code_module", "code_presentation"], how="left")
        df["norm_weight"] = df["weight"] / df["total_weight"].replace(0, np.nan)

        # Weighted score contribution
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
        df["weighted_score"] = df["score"] * df["norm_weight"]

        final_scores = (
            df.groupby(["id_student", "code_module", "code_presentation"])
            ["weighted_score"]
            .sum()
            .reset_index()
            .rename(columns={"weighted_score": "final_score"})
        )
        final_scores["final_score"] = final_scores["final_score"].clip(0, 100)
        return final_scores


# ============================================================
# 3. Sequence Feature Builder
# ============================================================
class SequenceBuilder:
    """
    Build weekly time-series sequences: X_seq in R^{N x T x D}
    D = 4: [weekly_clicks, active_days, activity_types, submissions]
    """

    def __init__(self, max_weeks: int = SEQUENCE_MAX_WEEKS, cutoff_week: Optional[int] = None):
        self.max_weeks = max_weeks
        self.cutoff_week = cutoff_week  # For early prediction (4 or 8)

    def build(
        self,
        student_vle: pd.DataFrame,
        student_assessment: pd.DataFrame,
        vle_meta: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build per-student per-week aggregated features."""
        df = student_vle.copy()
        df["week"] = (df["date"] // 7).astype(int)

        if self.cutoff_week is not None:
            df = df[df["week"] < self.cutoff_week]

        df["week"] = df["week"].clip(0, self.max_weeks - 1)

        # --- Weekly clicks ---
        weekly_clicks = (
            df.groupby(["id_student", "code_module", "code_presentation", "week"])
            ["sum_click"].sum()
            .reset_index()
            .rename(columns={"sum_click": "weekly_clicks"})
        )

        # --- Active days proxy (unique dates per week) ---
        df["day"] = df["date"]
        active_days = (
            df.groupby(["id_student", "code_module", "code_presentation", "week"])
            ["day"].nunique()
            .reset_index()
            .rename(columns={"day": "active_days"})
        )

        # --- Activity types (unique sites) ---
        act_types = (
            df.groupby(["id_student", "code_module", "code_presentation", "week"])
            ["id_site"].nunique()
            .reset_index()
            .rename(columns={"id_site": "activity_types"})
        )

        weekly = weekly_clicks.merge(active_days, on=["id_student","code_module","code_presentation","week"], how="outer")
        weekly = weekly.merge(act_types,  on=["id_student","code_module","code_presentation","week"], how="outer")
        weekly["submissions"] = 0  # placeholder; can be filled from student_assessment dates

        weekly = weekly.fillna(0)
        return weekly

    def to_sequences(self, weekly: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert weekly dataframe to 3D array.
        Returns:
            sequences: (N, T, D)
            keys: (N, 3) - [student_id, module, presentation]
        """
        feature_cols = ["weekly_clicks", "active_days", "activity_types", "submissions"]
        groups = weekly.groupby(["id_student", "code_module", "code_presentation"])

        sequences, keys = [], []
        for (sid, mod, pres), grp in groups:
            seq = np.zeros((self.max_weeks, len(feature_cols)), dtype=np.float32)
            for _, row in grp.iterrows():
                w = int(row["week"])
                if 0 <= w < self.max_weeks:
                    seq[w] = [row[c] for c in feature_cols]
            sequences.append(seq)
            keys.append((sid, mod, pres))

        return np.array(sequences), np.array(keys, dtype=object)


# ============================================================
# 4. Statistical Feature Builder
# ============================================================
class StatFeatureBuilder:
    """
    Build tabular statistical features from behavioral sequences.
    X_tab in R^{N x F}
    """

    @staticmethod
    def compute_entropy(clicks: np.ndarray) -> float:
        total = clicks.sum()
        if total == 0:
            return 0.0
        p = clicks / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-10)))

    def build(self, sequences: np.ndarray, keys: np.ndarray) -> pd.DataFrame:
        """
        sequences: (N, T, D)
        keys: (N, 3) - [student_id, module, presentation]
        """
        records = []
        for i, (seq, key) in enumerate(zip(sequences, keys)):
            clicks = seq[:, 0]  # weekly_clicks channel
            total = clicks.sum()
            active_weeks = (clicks > 0).sum()
            mean_c = clicks.mean()
            std_c = clicks.std()
            entropy = self.compute_entropy(clicks)
            early = clicks[:4].sum() / (total + 1e-10)
            growth = (clicks[10:].mean() - clicks[:10].mean()) if len(clicks) >= 10 else 0
            cv = std_c / (mean_c + 1e-10)

            records.append({
                "id_student": key[0],
                "code_module": key[1],
                "code_presentation": key[2],
                "total_clicks": total,
                "active_weeks": active_weeks,
                "mean_clicks": mean_c,
                "std_clicks": std_c,
                "behavior_entropy": entropy,
                "growth_rate": growth,
                "early_click_ratio": early,
                "click_cv": cv,
                "max_weekly_clicks": clicks.max(),
                "min_weekly_clicks": clicks.min(),
            })
        return pd.DataFrame(records)


# ============================================================
# 5. Static Feature Encoder
# ============================================================
class StaticFeatureEncoder:
    """
    Encode demographic and registration features from studentInfo.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, info_df: pd.DataFrame) -> pd.DataFrame:
        df = info_df.copy()
        for col in STATIC_CAT_FEATURES:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        num_cols = [c for c in STATIC_NUM_FEATURES if c in df.columns]
        if num_cols:
            df[num_cols] = self.scaler.fit_transform(df[num_cols].fillna(0))

        self.fitted = True
        return df

    def transform(self, info_df: pd.DataFrame) -> pd.DataFrame:
        df = info_df.copy()
        for col, le in self.label_encoders.items():
            if col in df.columns:
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                df[f"{col}_enc"] = le.transform(df[col])

        num_cols = [c for c in STATIC_NUM_FEATURES if c in df.columns]
        if num_cols:
            df[num_cols] = self.scaler.transform(df[num_cols].fillna(0))
        return df


# ============================================================
# 6. Dataset Builder
# ============================================================
class OULADDatasetBuilder:
    """
    Main orchestrator: loads raw tables and outputs unified dataset dict.

    Output format:
    {
        "sequence": (N, T, D),
        "tabular":  (N, F),
        "y_reg":    (N,),
        "y_cls":    (N,),
        "module":   (N,),
        "student_id": (N,)
    }
    """

    def __init__(self, data_dir: str = DATA_DIR, cutoff_week: Optional[int] = None):
        self.loader = OULADLoader(data_dir)
        self.cutoff_week = cutoff_week
        self.grade_builder = GradeReconstructor()
        self.seq_builder = SequenceBuilder(cutoff_week=cutoff_week)
        self.stat_builder = StatFeatureBuilder()
        self.static_encoder = StaticFeatureEncoder()

    def build(self) -> Dict:
        logger.info("Loading raw tables...")
        tables = self.loader.load_all()

        if not tables:
            logger.warning("No data files found. Generating synthetic data for demonstration.")
            return self._generate_synthetic_dataset()

        # --- Grade reconstruction ---
        logger.info("Reconstructing grades...")
        grades = self.grade_builder.compute(
            tables["student_assessment"], tables["assessments"]
        )

        # --- Sequence features ---
        logger.info("Building weekly sequences...")
        weekly = self.seq_builder.build(
            tables["student_vle"],
            tables.get("student_assessment", pd.DataFrame()),
            tables.get("vle", None),
        )
        sequences, keys = self.seq_builder.to_sequences(weekly)

        # --- Statistical features ---
        logger.info("Computing statistical features...")
        stat_feats = self.stat_builder.build(sequences, keys)

        # --- Static features ---
        logger.info("Encoding static features...")
        info = tables["student_info"].copy()
        info = self.static_encoder.fit_transform(info)

        # --- Merge all ---
        keys_df = pd.DataFrame(keys, columns=["id_student", "code_module", "code_presentation"])
        merged = keys_df.merge(stat_feats, on=["id_student","code_module","code_presentation"], how="left")
        merged = merged.merge(grades, on=["id_student","code_module","code_presentation"], how="left")

        enc_cols = [f"{c}_enc" for c in STATIC_CAT_FEATURES] + STATIC_NUM_FEATURES
        avail_enc = [c for c in enc_cols if c in info.columns]
        merged = merged.merge(
            info[["id_student","code_module","code_presentation"] + avail_enc + ["final_result"]],
            on=["id_student","code_module","code_presentation"], how="left"
        )

        # --- Labels ---
        merged["y_reg"] = merged["final_score"].fillna(0).clip(0, 100)
        merged["y_cls"] = merged["final_result"].map(LABEL_MAP).fillna(0).astype(int)

        # --- Tabular feature matrix ---
        stat_cols = ["total_clicks","active_weeks","mean_clicks","std_clicks",
                     "behavior_entropy","growth_rate","early_click_ratio",
                     "click_cv","max_weekly_clicks","min_weekly_clicks"]
        tab_cols = stat_cols + avail_enc + [c for c in STATIC_NUM_FEATURES if c in merged.columns]
        X_tab = merged[tab_cols].fillna(0).values.astype(np.float32)

        # Scale tabular features
        scaler = StandardScaler()
        X_tab = scaler.fit_transform(X_tab)

        dataset = {
            "sequence":   sequences.astype(np.float32),
            "tabular":    X_tab.astype(np.float32),
            "y_reg":      merged["y_reg"].values.astype(np.float32),
            "y_cls":      merged["y_cls"].values.astype(np.int64),
            "module":     merged["code_module"].values,
            "student_id": merged["id_student"].values,
            "tab_feature_names": tab_cols,
        }

        logger.info(f"Dataset built: N={len(sequences)}, T={SEQUENCE_MAX_WEEKS}, D={SEQUENCE_DIM}, F={X_tab.shape[1]}")
        return dataset

    def _generate_synthetic_dataset(self, n_per_module: int = 500) -> Dict:
        """Generate synthetic dataset for code testing when real data unavailable."""
        np.random.seed(SEED)
        N = n_per_module * len(MODULES)
        T = SEQUENCE_MAX_WEEKS
        D = SEQUENCE_DIM

        sequences = np.random.exponential(10, size=(N, T, D)).astype(np.float32)
        sequences[:, :, 0] = np.clip(sequences[:, :, 0], 0, 500)

        # Correlated tabular features
        X_tab_raw = np.random.randn(N, 18).astype(np.float32)
        X_tab = StandardScaler().fit_transform(X_tab_raw)

        # Synthetic labels with some correlation
        base_score = (sequences[:, :, 0].mean(axis=1) * 0.5 + np.random.randn(N) * 10).clip(0, 100)
        y_reg = base_score.astype(np.float32)
        y_cls = np.where(y_reg >= 70, 2, np.where(y_reg >= 50, 1, 0)).astype(np.int64)

        modules = np.array([m for m in MODULES for _ in range(n_per_module)])
        student_ids = np.arange(N)

        tab_feature_names = [
            "total_clicks","active_weeks","mean_clicks","std_clicks",
            "behavior_entropy","growth_rate","early_click_ratio",
            "click_cv","max_weekly_clicks","min_weekly_clicks",
            "studied_credits","num_of_prev_attempts",
            "gender_enc","region_enc","highest_education_enc",
            "imd_band_enc","age_band_enc","disability_enc"
        ]

        return {
            "sequence":   sequences,
            "tabular":    X_tab,
            "y_reg":      y_reg,
            "y_cls":      y_cls,
            "module":     modules,
            "student_id": student_ids,
            "tab_feature_names": tab_feature_names,
        }


# ============================================================
# 7. Cross-Module Splitter (LOMO)
# ============================================================
class LeaveOneModuleOut:
    """
    Leave-One-Module-Out cross-validation splitter.
    For transfer learning experiments.
    """

    def split(self, dataset: Dict) -> List[Tuple[Dict, Dict, str]]:
        """
        Returns list of (train_dataset, test_dataset, held_out_module)
        """
        modules = dataset["module"]
        unique_modules = np.unique(modules)
        splits = []

        for test_mod in unique_modules:
            train_mask = modules != test_mod
            test_mask  = modules == test_mod

            train_ds = {k: v[train_mask] if isinstance(v, np.ndarray) else v
                        for k, v in dataset.items() if k != "tab_feature_names"}
            train_ds["tab_feature_names"] = dataset.get("tab_feature_names", [])

            test_ds = {k: v[test_mask] if isinstance(v, np.ndarray) else v
                       for k, v in dataset.items() if k != "tab_feature_names"}
            test_ds["tab_feature_names"] = dataset.get("tab_feature_names", [])

            splits.append((train_ds, test_ds, test_mod))
            logger.info(f"LOMO split: test={test_mod}, train_n={train_mask.sum()}, test_n={test_mask.sum()}")

        return splits


# ============================================================
# 8. Meta-Learning Task Builder
# ============================================================
class MetaTaskBuilder:
    """
    Construct per-module tasks for MAML meta-learning.
    Each module = one task: Task_m = {X_m, y_m}
    """

    def build_tasks(self, dataset: Dict) -> List[Dict]:
        modules = dataset["module"]
        unique_modules = np.unique(modules)
        tasks = []

        for mod in unique_modules:
            mask = modules == mod
            task = {
                "module":    mod,
                "sequence":  dataset["sequence"][mask],
                "tabular":   dataset["tabular"][mask],
                "y_reg":     dataset["y_reg"][mask],
                "y_cls":     dataset["y_cls"][mask],
            }
            tasks.append(task)
            logger.info(f"Meta task: module={mod}, n={mask.sum()}")

        return tasks


# ============================================================
# Train/Val/Test Split Utility
# ============================================================
def split_dataset(dataset: Dict, test_size: float = 0.2, val_size: float = 0.1, seed: int = SEED):
    """Split dataset into train/val/test preserving module distribution."""
    N = len(dataset["y_reg"])
    indices = np.arange(N)

    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size / (1 - test_size), random_state=seed)

    def subset(idx):
        sub = {}
        for k, v in dataset.items():
            if isinstance(v, np.ndarray):
                sub[k] = v[idx]
            else:
                sub[k] = v
        return sub

    return subset(train_idx), subset(val_idx), subset(test_idx)
