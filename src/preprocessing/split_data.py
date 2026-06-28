"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Dataset Splitting
Python  : 3.11
=========================================================

This module performs chronological train,
validation and test splitting for
time-series forecasting.

No random shuffling is used.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import (
    TARGET_COLUMN,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
)

# ==========================================================
# Logger
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# Dataset Information
# ==========================================================

def dataset_information(
    df: pd.DataFrame
) -> None:
    """
    Print dataset information.
    """

    logger.info("=" * 60)

    logger.info("Dataset Information")

    logger.info("=" * 60)

    logger.info("Rows : %d", df.shape[0])

    logger.info("Columns : %d", df.shape[1])

    logger.info("=" * 60)


# ==========================================================
# Sort Dataset
# ==========================================================

def sort_dataset(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Sort dataframe chronologically.
    """

    logger.info("Sorting dataset...")

    df = df.sort_values("date")

    df.reset_index(
        drop=True,
        inplace=True
    )

    logger.info("Dataset sorted.")

    return df


# ==========================================================
# Feature Target Split
# ==========================================================

def feature_target_split(
    df: pd.DataFrame
):
    """
    Split dataset into X and y.
    """

    logger.info(
        "Creating feature matrix..."
    )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    logger.info(
        "Feature matrix created."
    )

    return X, y


# ==========================================================
# Calculate Split Sizes
# ==========================================================

def calculate_split_sizes(
    total_rows: int
):
    """
    Calculate train, validation
    and test sizes.
    """

    train_size = int(
        total_rows * TRAIN_RATIO
    )

    valid_size = int(
        total_rows * VALID_RATIO
    )

    test_size = (
        total_rows
        - train_size
        - valid_size
    )

    logger.info(
        "Train Size : %d",
        train_size
    )

    logger.info(
        "Validation Size : %d",
        valid_size
    )

    logger.info(
        "Test Size : %d",
        test_size
    )

    return (

        train_size,

        valid_size,

        test_size

    )
# ==========================================================
# Create Training Set
# ==========================================================

def create_train_set(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int
):
    """
    Create training dataset.
    """

    logger.info("Creating Training Dataset...")

    X_train = X.iloc[:train_size].copy()

    y_train = y.iloc[:train_size].copy()

    logger.info(
        "Training Shape : %s",
        X_train.shape
    )

    return X_train, y_train


# ==========================================================
# Create Validation Set
# ==========================================================

def create_validation_set(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int,
    valid_size: int
):
    """
    Create validation dataset.
    """

    logger.info(
        "Creating Validation Dataset..."
    )

    start = train_size

    end = train_size + valid_size

    X_valid = X.iloc[start:end].copy()

    y_valid = y.iloc[start:end].copy()

    logger.info(
        "Validation Shape : %s",
        X_valid.shape
    )

    return X_valid, y_valid


# ==========================================================
# Create Test Set
# ==========================================================

def create_test_set(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int,
    valid_size: int
):
    """
    Create testing dataset.
    """

    logger.info(
        "Creating Testing Dataset..."
    )

    start = train_size + valid_size

    X_test = X.iloc[start:].copy()

    y_test = y.iloc[start:].copy()

    logger.info(
        "Testing Shape : %s",
        X_test.shape
    )

    return X_test, y_test


# ==========================================================
# Verify Dataset Shapes
# ==========================================================

def verify_shapes(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test
):
    """
    Verify dataset dimensions.
    """

    logger.info("=" * 60)

    logger.info("Dataset Shapes")

    logger.info("=" * 60)

    logger.info(
        "X Train : %s",
        X_train.shape
    )

    logger.info(
        "X Validation : %s",
        X_valid.shape
    )

    logger.info(
        "X Test : %s",
        X_test.shape
    )

    logger.info(
        "y Train : %s",
        y_train.shape
    )

    logger.info(
        "y Validation : %s",
        y_valid.shape
    )

    logger.info(
        "y Test : %s",
        y_test.shape
    )

    logger.info("=" * 60)


# ==========================================================
# Verify Date Ranges
# ==========================================================

def verify_date_ranges(
    df: pd.DataFrame,
    train_size: int,
    valid_size: int,
    date_column: str = "date"
):
    """
    Verify chronological split.
    """

    logger.info("=" * 60)

    logger.info("Date Range Verification")

    logger.info("=" * 60)

    train = df.iloc[:train_size]

    valid = df.iloc[
        train_size:
        train_size + valid_size
    ]

    test = df.iloc[
        train_size + valid_size:
    ]

    logger.info(
        "Train : %s -> %s",
        train[date_column].min(),
        train[date_column].max()
    )

    logger.info(
        "Validation : %s -> %s",
        valid[date_column].min(),
        valid[date_column].max()
    )

    logger.info(
        "Test : %s -> %s",
        test[date_column].min(),
        test[date_column].max()
    )

    logger.info("=" * 60)
# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(
    df: pd.DataFrame,
    output_path: str | Path
) -> None:
    """
    Save DataFrame to CSV.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    logger.info(
        "Saved : %s",
        output_path
    )


# ==========================================================
# Save All Split Files
# ==========================================================

def save_split_datasets(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
    output_dir
):
    """
    Save all train/validation/test datasets.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_dataset(
        X_train,
        output_dir / "X_train.csv"
    )

    save_dataset(
        X_valid,
        output_dir / "X_valid.csv"
    )

    save_dataset(
        X_test,
        output_dir / "X_test.csv"
    )

    save_dataset(
        y_train.to_frame(name=TARGET_COLUMN),
        output_dir / "y_train.csv"
    )

    save_dataset(
        y_valid.to_frame(name=TARGET_COLUMN),
        output_dir / "y_valid.csv"
    )

    save_dataset(
        y_test.to_frame(name=TARGET_COLUMN),
        output_dir / "y_test.csv"
    )

    logger.info(
        "All split datasets saved successfully."
    )


# ==========================================================
# Complete Split Pipeline
# ==========================================================

def run_split_pipeline(
    df: pd.DataFrame,
    output_dir: str | Path
):
    """
    Complete chronological splitting pipeline.
    """

    logger.info("=" * 70)
    logger.info("START DATA SPLITTING")
    logger.info("=" * 70)

    dataset_information(df)

    df = sort_dataset(df)

    X, y = feature_target_split(df)

    (
        train_size,
        valid_size,
        test_size
    ) = calculate_split_sizes(len(df))

    X_train, y_train = create_train_set(
        X,
        y,
        train_size
    )

    X_valid, y_valid = create_validation_set(
        X,
        y,
        train_size,
        valid_size
    )

    X_test, y_test = create_test_set(
        X,
        y,
        train_size,
        valid_size
    )

    verify_shapes(
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    )

    verify_date_ranges(
        df,
        train_size,
        valid_size
    )

    save_split_datasets(
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        output_dir
    )

    logger.info("=" * 70)
    logger.info("DATA SPLITTING COMPLETED")
    logger.info("=" * 70)

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    )


# ==========================================================
# Standalone Execution
# ==========================================================

if __name__ == "__main__":

    logger.info(
        "Split Data Module Loaded Successfully."
    )
