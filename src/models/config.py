"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Model Configuration
Python  : 3.11
=========================================================

Central configuration for model training.
"""

from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==========================================================
# Directories
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Dataset Files
# ==========================================================

X_TRAIN_FILE = PROCESSED_DIR / "X_train.csv"
Y_TRAIN_FILE = PROCESSED_DIR / "y_train.csv"

X_VALID_FILE = PROCESSED_DIR / "X_valid.csv"
Y_VALID_FILE = PROCESSED_DIR / "y_valid.csv"

X_TEST_FILE = PROCESSED_DIR / "X_test.csv"
Y_TEST_FILE = PROCESSED_DIR / "y_test.csv"

# ==========================================================
# Saved Model Files
# ==========================================================

BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"

MODEL_RESULTS_FILE = MODELS_DIR / "model_results.csv"

FEATURE_IMPORTANCE_FILE = MODELS_DIR / "feature_importance.csv"

PREDICTIONS_FILE = MODELS_DIR / "predictions.csv"

# ==========================================================
# Target
# ==========================================================

TARGET_COLUMN = "units_sold"

# ==========================================================
# Training Parameters
# ==========================================================

RANDOM_STATE = 42

N_JOBS = -1

CV_FOLDS = 5

# ==========================================================
# Evaluation Metrics
# ==========================================================

METRICS = [

    "MAE",

    "MSE",

    "RMSE",

    "R2",

    "MAPE"

]

# ==========================================================
# Random Forest
# ==========================================================

RF_PARAMS = {

    "n_estimators": 300,

    "max_depth": 20,

    "random_state": RANDOM_STATE,

    "n_jobs": N_JOBS

}

# ==========================================================
# Gradient Boosting
# ==========================================================

GB_PARAMS = {

    "n_estimators": 300,

    "learning_rate": 0.05,

    "random_state": RANDOM_STATE

}

# ==========================================================
# XGBoost
# ==========================================================

XGB_PARAMS = {

    "n_estimators": 300,

    "learning_rate": 0.05,

    "max_depth": 8,

    "random_state": RANDOM_STATE

}

# ==========================================================
# LightGBM
# ==========================================================

LGBM_PARAMS = {

    "n_estimators": 300,

    "learning_rate": 0.05,

    "random_state": RANDOM_STATE

}

# ==========================================================
# CatBoost
# ==========================================================

CATBOOST_PARAMS = {

    "iterations": 300,

    "learning_rate": 0.05,

    "verbose": 0,

    "random_state": RANDOM_STATE

}