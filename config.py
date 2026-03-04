"""
Configuration file for OULAD Multi-Modal Fusion & Transfer Learning System
"""

import os

# ==================== Paths ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

for d in [DATA_DIR, OUTPUT_DIR, LOG_DIR, MODEL_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== Random Seed ====================
SEED = 42

# ==================== Data Settings ====================
MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]

# Prediction window options
PREDICTION_WINDOWS = {
    "full": None,   # full course
    "week4": 4,
    "week8": 8,
}

# Target variable
TARGET_REG = "final_score"       # regression
TARGET_CLS = "result_label"      # classification

# ==================== Feature Settings ====================
SEQUENCE_MAX_WEEKS = 20
SEQUENCE_FEATURES = ["weekly_clicks", "active_days", "activity_types", "submissions"]
SEQUENCE_DIM = len(SEQUENCE_FEATURES)  # D = 4

STATIC_CAT_FEATURES = [
    "gender", "region", "highest_education",
    "imd_band", "age_band", "disability"
]
STATIC_NUM_FEATURES = ["studied_credits", "num_of_prev_attempts"]

# Tabular features (statistical + static)
TABULAR_FEATURES = [
    "total_clicks", "active_weeks", "mean_clicks", "std_clicks",
    "behavior_entropy", "growth_rate", "early_click_ratio",
    "click_cv", "max_weekly_clicks", "min_weekly_clicks",
    "studied_credits", "num_of_prev_attempts",
    "gender_enc", "region_enc", "highest_education_enc",
    "imd_band_enc", "age_band_enc", "disability_enc"
]

# ==================== LSTM Settings ====================
LSTM_CONFIG = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 50,
    "patience": 10,
}

# ==================== XGBoost Settings ====================
XGB_CONFIG = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": -1,
}

# ==================== CatBoost Settings ====================
CATBOOST_CONFIG = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": SEED,
    "verbose": 0,
}

# ==================== Fusion Settings ====================
FUSION_CONFIG = {
    "method": "dynamic",          # "stacking" or "dynamic"
    "meta_model": "mlp",          # for stacking: "linear" or "mlp"
    "dynamic_hidden": 32,
    "learning_rate": 1e-3,
    "epochs": 30,
}

# ==================== MAML Settings ====================
MAML_CONFIG = {
    "inner_lr": 0.01,
    "outer_lr": 0.001,
    "inner_steps": 5,
    "meta_epochs": 100,
    "support_size": 64,
    "query_size": 64,
    "hidden_size": 64,
}

# ==================== Experiment Settings ====================
CV_FOLDS = 5
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# ==================== Classification Labels ====================
# Distinction=2, Pass=1, Fail/Withdrawn=0
LABEL_MAP = {
    "Distinction": 2,
    "Pass": 1,
    "Fail": 0,
    "Withdrawn": 0,
}
NUM_CLASSES = 3
