"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Configuration
Python  : 3.11
=========================================================

Central configuration file.

Contains:
• Project Paths
• Dataset Paths
• Model Parameters
• Feature Lists
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Dataset Paths
# ==========================================================

RAW_DATA_FILE = (
    RAW_DATA_DIR /
    "enterprise_supply_chain_dataset_india_2023_2025.csv"
)

FEATURE_ENGINEERED_DATA_FILE = (
    PROCESSED_DATA_DIR /
    "feature_engineered_supply_chain_dataset.csv"
)

# ==========================================================
# Train / Validation / Test Files
# ==========================================================

X_TRAIN_FILE = PROCESSED_DATA_DIR / "X_train.csv"
X_VALID_FILE = PROCESSED_DATA_DIR / "X_valid.csv"
X_TEST_FILE = PROCESSED_DATA_DIR / "X_test.csv"

Y_TRAIN_FILE = PROCESSED_DATA_DIR / "y_train.csv"
Y_VALID_FILE = PROCESSED_DATA_DIR / "y_valid.csv"
Y_TEST_FILE = PROCESSED_DATA_DIR / "y_test.csv"

# ==========================================================
# Saved Objects
# ==========================================================

SCALER_FILE = ARTIFACTS_DIR / "scaler.pkl"
LABEL_ENCODER_FILE = ARTIFACTS_DIR / "label_encoder.pkl"
FEATURE_LIST_FILE = ARTIFACTS_DIR / "feature_list.pkl"

# ==========================================================
# Target Column
# ==========================================================

TARGET_COLUMN = "units_sold"

DATE_COLUMN = "date"

# ==========================================================
# Dataset Split
# ==========================================================

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

# ==========================================================
# Numerical Features
# ==========================================================

NUMERICAL_COLUMNS = [
    "selling_price",
    "discount_percent",
    "inventory_level",
    "reorder_point",
    "safety_stock",
    "supplier_lead_time_days",
    "delivery_time_days",
    "shipping_cost",
    "procurement_cost",
    "profit_margin",
    "inventory_turnover"
]

# ==========================================================
# Label Encoding Columns
# ==========================================================

LABEL_ENCODING_COLUMNS = [
    "brand",
    "warehouse_id",
    "supplier_id",
    "promotion",
    "delivery_status"
]

# ==========================================================
# One-Hot Encoding Columns
# ==========================================================

ONE_HOT_COLUMNS = [
    "category",
    "promotion_type",
    "festival_name",
    "holiday_name",
    "payment_method",
    "weather",
    "season",
    "region",
    "state",
    "city"
]

# ==========================================================
# Feature Engineering Parameters
# ==========================================================

LAG_FEATURES = [1, 7, 14, 30]

ROLLING_WINDOWS = [7, 14, 30]